from __future__ import annotations

from typing import Any

from ..llm import (
    ALLOWED_EVIDENCE_REFS_KEY,
    APICLINICRReasoner,
    CLINICRModelUnavailable,
    CLINICRReasoner,
)
from .schemas import (
    CandidateGeneration,
    DiagnosticCandidate,
    EliminativeAssessment,
    SUPPORTED_DISORDERS,
)
from .verifier import IndependentVerifier
from .assessment import assessments_from_step1
from .information_report import build_information_report


from shared.disorder_catalog import DISORDER_COUNT, disorder_metadata
from shared.candidate_ordering import candidate_sort_key
from shared.diagnostic_knowledge import (
    DiagnosticKnowledgeNotConnected, DiagnosticKnowledgeProvider,
    get_default_knowledge_provider,
)

MIN_PRIMARY_CANDIDATE_CONFIDENCE = 0.5


DISORDER_SEVERITY_SCALE = {
    "D013": "PHQ9",
    "D011": "GAD7",
    "D018": "PCL5",
    "D016": "Y_BOCS",
    "Insomnia": "ISI",
    "D008": "MDQ",
    "D009": "MDQ",
}

SEVERITY_THRESHOLDS = {
    "PHQ9": (
        (0, 4, "normal"),
        (5, 9, "mild"),
        (10, 14, "moderate"),
        (15, 19, "moderate_severe"),
        (20, 27, "severe"),
    ),
    "GAD7": (
        (0, 4, "normal"),
        (5, 9, "mild"),
        (10, 14, "moderate"),
        (15, 21, "severe"),
    ),
}


class CLINICR:
    """Step 1 一次取证、确定性汇总，再执行排名第一候选独立终验。"""

    def __init__(
        self,
        reasoner: CLINICRReasoner | None = None,
        verifier: IndependentVerifier | None = None,
        *,
        knowledge: DiagnosticKnowledgeProvider | None = None,
    ) -> None:
        self.knowledge = knowledge if knowledge is not None else get_default_knowledge_provider()
        self.reasoner = reasoner or APICLINICRReasoner(knowledge=self.knowledge)
        self.verifier = verifier or IndependentVerifier(knowledge=self.knowledge)

    def run(
        self,
        profile: dict[str, Any],
        safety_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        known_refs = self._known_evidence_refs(profile)
        # 允许列表由规则从画像生成，覆盖任何同名输入，不能由患者或模型自行指定。
        model_profile = {
            **profile,
            ALLOWED_EVIDENCE_REFS_KEY: sorted(known_refs),
        }

        # Step 1: 完整考虑 MentalKG 23 个节点及失眠障碍，按患者证据保留候选。
        generation = self.reasoner.generate_candidates(model_profile)
        candidates = self._validate_generation(generation, known_refs)

        # Reuse the completed Step 1 records; no per-candidate ELIMINATIVE call.
        try:
            assessments = assessments_from_step1(generation, candidates, knowledge=self.knowledge)
        except (KeyError, ValueError, DiagnosticKnowledgeNotConnected) as exc:
            raise CLINICRModelUnavailable(f"Step 1证据汇总失败：{exc}") from exc
        for assessment in assessments:
            self._validate_assessment(assessment, known_refs)

        severity = self._get_severity(profile)
        severity_level = self._get_severity_level(severity)
        candidate_by_disorder = {item.disorder: item for item in candidates}
        assessment_by_disorder = {item.disorder: item for item in assessments}

        eligible = [
            candidate
            for candidate in candidates
            if assessment_by_disorder[candidate.disorder].conclusion
            == "meets_criteria"
        ]
        eligible.sort(key=candidate_sort_key)

        under_observation = [
            self._observation_payload(candidate_by_disorder[item.disorder], item)
            for item in assessments
            if item.conclusion == "watch"
        ]
        ruled_out = [
            self._ruled_out_payload(item)
            for item in assessments
            if item.conclusion == "ruled_out"
        ]

        if not eligible:
            missing = self._dedupe_missing(self._collect_missing(assessments) + [
                item.model_dump(mode="json") for item in generation.missing_dimensions
            ])
            if not missing:
                missing = [{
                    "field": "diagnostic_evidence",
                    "purpose": "当前候选均未满足核心标准，需要补充可区分诊断的证据",
                    "suggested_question": "请继续了解症状的具体表现、持续时间和功能影响。",
                }]
            return self._result(
                candidates=candidates,
                assessments=assessments,
                primary=None,
                provisional_primary=None,
                secondary=[],
                under_observation=under_observation,
                ruled_out=ruled_out,
                severity=severity,
                severity_level=severity_level,
                verifier={
                    "score": 0.0,
                    "decision": "rejected",
                    "routing": "return_to_l2",
                    "checks": [],
                    "fail_reasons": ["Step 1候选尚未满足完整必需标准及鉴别条件"],
                    "missing_dimensions": missing,
                },
                safety_context=safety_context,
                generation=generation,
            )

        top_candidate = eligible[0]
        top_assessment = assessment_by_disorder[top_candidate.disorder]
        provisional_primary = self._primary_payload(
            top_candidate, top_assessment, severity
        )

        differential_context = [
            {
                "disorder": item.disorder,
                "conclusion": item.conclusion,
                "conclusion_reason": item.conclusion_reason,
                "criteria_detail": [
                    detail.model_dump(mode="json")
                    for detail in item.criteria_detail
                ],
            }
            for item in assessments
            if item.disorder != top_candidate.disorder
        ]

        # Step 3: 独立Verifier不接收候选置信度，只评估推理质量。
        verifier = self.verifier.verify(
            top_candidate,
            top_assessment,
            model_profile,
            differential_context,
        )
        self._validate_verifier_refs(verifier, known_refs)

        if top_candidate.confidence < MIN_PRIMARY_CANDIDATE_CONFIDENCE:
            verifier = {
                **verifier,
                "decision": "rejected",
                "routing": "return_to_l2",
                "fail_reasons": list(dict.fromkeys(
                    verifier["fail_reasons"]
                    + [
                        "排名第一候选的初步置信度低于进入治疗匹配的最低门槛0.5"
                    ]
                )),
                "missing_dimensions": self._ensure_missing(
                    verifier["missing_dimensions"]
                ),
            }

        primary = (
            provisional_primary
            if verifier["routing"] in {"pass", "conditional"}
            else None
        )
        secondary = [
            self._secondary_payload(
                candidate,
                assessment_by_disorder[candidate.disorder],
            )
            for candidate in eligible[1:]
            if candidate.confidence >= MIN_PRIMARY_CANDIDATE_CONFIDENCE
        ]

        return self._result(
            candidates=candidates,
            assessments=assessments,
            primary=primary,
            provisional_primary=provisional_primary,
            secondary=secondary,
            under_observation=under_observation,
            ruled_out=ruled_out,
            severity=severity,
            severity_level=severity_level,
            verifier=verifier,
            safety_context=safety_context,
            generation=generation,
        )

    @staticmethod
    def _validate_generation(
        generation: CandidateGeneration,
        known_refs: set[str],
    ) -> list[DiagnosticCandidate]:
        considered = generation.considered_disorders
        if len(set(considered)) != DISORDER_COUNT or set(considered) != set(SUPPORTED_DISORDERS):
            raise CLINICRModelUnavailable(
                f"CLINICR Step 1没有完整且唯一地考虑规定的{DISORDER_COUNT}种障碍"
            )
        disorders = [item.disorder for item in generation.candidates]
        if len(disorders) != len(set(disorders)):
            raise CLINICRModelUnavailable("CLINICR Step 1生成了重复候选")
        for candidate in generation.candidates:
            CLINICR._assert_known_refs(
                candidate.evidence_refs,
                known_refs,
                f"Step 1候选{candidate.disorder}",
            )
        for record in generation.tree_trace:
            CLINICR._assert_known_refs(record.evidence_refs, known_refs, f"Step 1节点{record.node_id}")
        return sorted(generation.candidates, key=candidate_sort_key)

    @staticmethod
    def _validate_assessment(
        assessment: EliminativeAssessment,
        known_refs: set[str],
    ) -> None:
        codes = [item.code for item in assessment.criteria_detail]
        if len(codes) != len(set(codes)):
            raise CLINICRModelUnavailable(
                f"Step 1候选{assessment.disorder}包含重复诊断条目"
            )
        for item in assessment.criteria_detail:
            if item.status in {"met", "not_met"} and not item.evidence_refs:
                raise CLINICRModelUnavailable(
                    f"Step 1候选{assessment.disorder}的{item.code}缺少判定证据"
                )
            CLINICR._assert_known_refs(
                item.evidence_refs,
                known_refs,
                f"Step 1候选{assessment.disorder}的{item.code}",
            )
        if (
            any(item.status == "pending" for item in assessment.criteria_detail)
            and not assessment.missing_dimensions
        ):
            raise CLINICRModelUnavailable(
                f"Step 1候选{assessment.disorder}含待确认条目但没有给出L2追问"
            )

    @staticmethod
    def _validate_verifier_refs(
        verifier: dict[str, Any], known_refs: set[str]
    ) -> None:
        for check in verifier["checks"]:
            CLINICR._assert_known_refs(
                check["evidence_refs"],
                known_refs,
                f"Step 3终验维度{check['check_id']}",
            )

    @staticmethod
    def _assert_known_refs(
        refs: list[str], known_refs: set[str], context: str
    ) -> None:
        unknown = [ref for ref in refs if ref not in known_refs]
        if unknown:
            raise CLINICRModelUnavailable(
                f"{context}引用了患者画像中不存在的证据：{unknown}"
            )

    @staticmethod
    def _known_evidence_refs(profile: dict[str, Any]) -> set[str]:
        refs: set[str] = set()
        scale_keys = {
            "PHQ9", "GAD7", "PCL5", "Y_BOCS", "ISI", "MDQ",
            "ADNM", "RRS", "AAQ_II", "PHQ15",
        }
        for key in scale_keys:
            if key in profile and profile[key] is not None:
                refs.add(f"{key}={profile[key]}")
            items = profile.get(f"{key}_items")
            if isinstance(items, dict):
                for item, score in items.items():
                    refs.add(f"{key}_{item}={score}")

        for collection in (
            "key_symptoms_extracted",
            "emotions",
            "cognitive_patterns",
            "key_narratives",
            "functional_impairments",
        ):
            for item in profile.get(collection) or []:
                if isinstance(item, dict):
                    refs.update(str(ref) for ref in item.get("sources") or [] if ref)

        for node in profile.get("evidence_nodes") or []:
            if isinstance(node, dict) and node.get("evidence_id"):
                refs.add(str(node["evidence_id"]))
        return refs

    @staticmethod
    def _primary_payload(
        candidate: DiagnosticCandidate,
        assessment: EliminativeAssessment,
        severity: dict[str, str],
    ) -> dict[str, Any]:
        detail = [CLINICR._criterion_payload(item) for item in assessment.criteria_detail]
        met = [item["code"] for item in detail if item["status"] == "met"]
        scale = DISORDER_SEVERITY_SCALE.get(candidate.disorder)
        return {
            **disorder_metadata(candidate.disorder),
            "confidence": candidate.confidence,
            "severity": severity.get(scale, "unknown"),
            "evidence": candidate.core_evidence_summary,
            "dsm5_criteria_met": met,
            "dsm5_criteria_detail": detail,
        }

    @staticmethod
    def _secondary_payload(
        candidate: DiagnosticCandidate,
        assessment: EliminativeAssessment,
    ) -> dict[str, Any]:
        return {
            **disorder_metadata(candidate.disorder),
            "confidence": candidate.confidence,
            "evidence": assessment.conclusion_reason,
        }

    @staticmethod
    def _observation_payload(
        candidate: DiagnosticCandidate,
        assessment: EliminativeAssessment,
    ) -> dict[str, Any]:
        return {
            **disorder_metadata(candidate.disorder),
            "confidence": candidate.confidence,
            "reason": assessment.conclusion_reason,
            "missing_dimensions": [
                item.model_dump(mode="json") for item in assessment.missing_dimensions
            ],
        }

    @staticmethod
    def _ruled_out_payload(assessment: EliminativeAssessment) -> dict[str, Any]:
        refs = [
            ref
            for item in assessment.criteria_detail
            if item.status == "not_met"
            for ref in item.evidence_refs
        ]
        return {
            **disorder_metadata(assessment.disorder),
            "reason": assessment.conclusion_reason,
            "evidence_ref": ", ".join(dict.fromkeys(refs)),
        }

    @staticmethod
    def _criterion_payload(item: Any) -> dict[str, Any]:
        status = "unknown" if item.status == "pending" else item.status
        return {
            "code": item.code,
            "name": item.name,
            "status": status,
            "met": item.status == "met",
            "evidence_refs": item.evidence_refs,
        }

    @staticmethod
    def _collect_missing(
        assessments: list[EliminativeAssessment],
    ) -> list[dict[str, str]]:
        items = [
            item.model_dump(mode="json")
            for assessment in assessments
            for item in assessment.missing_dimensions
        ]
        return CLINICR._dedupe_missing(items)

    @staticmethod
    def _ensure_missing(items: list[dict[str, str]]) -> list[dict[str, str]]:
        if items:
            return items
        return [{
            "field": "diagnostic_evidence",
            "purpose": "提高候选诊断的初步支持度",
            "suggested_question": "请继续了解核心症状的具体表现、持续时间和功能影响。",
        }]

    @staticmethod
    def _dedupe_missing(items: list[dict[str, str]]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in items:
            if item["field"] in seen:
                continue
            seen.add(item["field"])
            result.append(item)
        return result

    @staticmethod
    def _get_severity(profile: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for scale, thresholds in SEVERITY_THRESHOLDS.items():
            score = profile.get(scale)
            if score is None:
                result[scale] = "unknown"
                continue
            result[scale] = next(
                (level for low, high, level in thresholds if low <= score <= high),
                "unknown",
            )
        return result

    @staticmethod
    def _get_severity_level(severity: dict[str, str]) -> int:
        if any(value == "severe" for value in severity.values()):
            return 3
        if any(value in {"moderate", "moderate_severe"} for value in severity.values()):
            return 2
        return 1

    def _result(
        self,
        *,
        candidates: list[DiagnosticCandidate],
        assessments: list[EliminativeAssessment],
        primary: dict[str, Any] | None,
        provisional_primary: dict[str, Any] | None,
        secondary: list[dict[str, Any]],
        under_observation: list[dict[str, Any]],
        ruled_out: list[dict[str, Any]],
        severity: dict[str, str],
        severity_level: int,
        verifier: dict[str, Any],
        safety_context: dict[str, Any] | None,
        generation: CandidateGeneration | None = None,
    ) -> dict[str, Any]:
        evidence_refs: list[dict[str, Any]] = []
        tree_sources = {
            record.node_id: record.source_refs
            for record in generation.tree_trace
        } if generation else {}
        for candidate in candidates:
            evidence_refs.append({
                "type": "CLINICR-step1",
                **disorder_metadata(candidate.disorder),
                "source": "L2 ClinicalIntake",
                "threshold": (
                    f"symptom_match={candidate.symptom_match.matched}/"
                    f"{candidate.symptom_match.total}; confidence={candidate.confidence}"
                ),
                "evidence_refs": candidate.evidence_refs,
            })
        for assessment in assessments:
            for criterion in assessment.criteria_detail:
                sources = tree_sources.get(criterion.code)
                if not sources:
                    knowledge = self.knowledge.get_candidate(assessment.disorder)
                    sources = [knowledge.get("criterion_source_refs", {}).get(
                        criterion.code, knowledge["source_ref"],
                    )]
                evidence_refs.append({
                    "type": "DSM-5-criterion",
                    **disorder_metadata(assessment.disorder),
                    "criterion_code": criterion.code,
                    "criterion_name": criterion.name,
                    "status": (
                        "unknown" if criterion.status == "pending" else criterion.status
                    ),
                    "evidence_refs": criterion.evidence_refs,
                    "source_doc": sources[0],
                })
        if provisional_primary:
            evidence_refs.append({
                "type": "CLINICR-step3",
                "disorder": provisional_primary["disorder"],
                "source": "Reward Model Verifier",
                "threshold": (
                    f"score={verifier['score']}; route={verifier['routing']}"
                ),
                "evidence_refs": [
                    ref
                    for check in verifier["checks"]
                    for ref in check["evidence_refs"]
                ],
            })

        flags = list((safety_context or {}).get("flags", []))
        routing_flag = {
            "conditional": "provisional_diagnosis_needs_more_information",
            "return_to_l2": "insufficient_diagnostic_evidence",
        }.get(verifier["routing"])
        if routing_flag:
            flags.append(routing_flag)

        if primary:
            summary = (
                f"CLINICR复用Step 1证据并终验得到{disorder_metadata(primary['disorder'])['display_name']}，"
                f"候选置信度={primary['confidence']}，"
                f"Verifier={verifier['score']}（{verifier['decision']}）。"
            )
        else:
            summary = (
                "CLINICR终验未达到进入疗法匹配的最低要求，"
                "返回L2补充指定信息。"
            )

        result = {
            "diagnosis": {
                "primary": primary,
                "secondary": secondary,
                "under_observation": under_observation,
                "ruled_out": ruled_out,
            },
            "provisional_diagnosis": provisional_primary,
            "candidate_confidence": (
                provisional_primary["confidence"] if provisional_primary else 0.0
            ),
            "verifier_score": verifier["score"],
            "verifier_decision": verifier["decision"],
            "verifier_routing": verifier["routing"],
            "verifier_checks": verifier["checks"],
            "verifier_fail_reasons": verifier["fail_reasons"],
            "missing_dimensions": verifier["missing_dimensions"],
            "severity": severity,
            "severity_level": severity_level,
            "considered_disorders": list(SUPPORTED_DISORDERS),
            "diagnostic_standard": "DSM-5",
            "candidate_generation_method": "dsm5_decision_tree" if generation and generation.tree_trace else "external_reasoner",
            "tree_trace": [item.model_dump(mode="json") for item in generation.tree_trace] if generation else [],
            "candidate_coverage": generation.candidate_coverage if generation else {},
            "candidate_missing_dimensions": [item.model_dump(mode="json") for item in generation.missing_dimensions] if generation else [],
            "candidates": [{**item.model_dump(mode="json"), **disorder_metadata(item.disorder)} for item in candidates],
            "checked": [
                {**item.model_dump(mode="json"), "source": "step1_tree_records"}
                for item in assessments
            ],
            "evidence_refs": evidence_refs,
            "confidence": verifier["score"],
            "flags": list(dict.fromkeys(flags)),
            "summary": summary,
        }
        result.update(build_information_report(result, knowledge=self.knowledge))
        if primary is None:
            readiness = result["diagnostic_readiness"]
            if readiness["status"] == "insufficient_information":
                result["summary"] = (
                    f"信息不足：{readiness['counts']['insufficient_information']}个候选仍缺必需信息，"
                    f"另有{readiness['counts']['known_blocker']}个候选存在已知不满足条件。"
                    "保留原诊断门槛，返回L2补充信息；pending不表示病症不存在。"
                )
            elif readiness["status"] == "known_blocker":
                result["summary"] = "当前候选存在已知不满足条件，未形成最终诊断；按原路由复核，不将其等同于资料缺失。"
            elif readiness["status"] == "supported":
                result["summary"] = "候选必需判据已满足，但终验或置信度门未放行；不形成最终诊断，按原路由处理。"
        return result

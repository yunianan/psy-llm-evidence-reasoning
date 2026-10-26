"""A0: patient evidence and display names only; never instantiate the G1 engine."""
import json
import math

from layers.layer3.diagnosis.schemas import CandidateGeneration, DiagnosticCandidate, MissingDimension
from layers.layer3.llm.client import (
    ALLOWED_EVIDENCE_REFS_KEY, _parse_json_object, _canonicalize_payload_evidence_refs,
    _assert_payload_evidence_refs, _request_structured_output_with_one_retry,
)
from shared.disorder_catalog import SUPPORTED_DISORDERS, disorder_metadata

PROMPT_VERSION = "a0-no-knowledge-v1"
METHOD = "no_knowledge_baseline"


def known_patient_refs(profile):
    # Reuse only the engine's pure patient-reference whitelist, never CLINICR.run.
    from layers.layer3.diagnosis.clinicr import CLINICR
    return sorted(CLINICR._known_evidence_refs(profile))


class A0NoKnowledge:
    def __init__(self, request, *, model, knowledge=None):
        self.request, self.model = request, model
        # knowledge deliberately remains unused, including while assembling prompts.

    def generate_candidates(self, profile):
        clinical = {k: v for k, v in profile.items() if k != ALLOWED_EVIDENCE_REFS_KEY}
        allowed = known_patient_refs(clinical)
        catalog = [{"disorder": key, "label": disorder_metadata(key)["label"]} for key in SUPPORTED_DISORDERS]
        prompt = (
            "你是无外部知识库的候选生成基线。只根据患者提供的事实和下列展示标签生成0至24个候选。"
            "信息不足允许空列表，不强制保留候选；不下最终诊断。"
            "只能引用患者证据白名单中的ref；没有可引用依据就不输出该候选。"
            "symptom_match只表示画像已陈述的不同线索数量，不表示诊断标准总数；"
            "total至少1，matched不得超过total。confidence为0至1的数值，不是临床概率。"
            "label逐字使用展示名。decision_paths必须为空。不补造患者事实。"
            "缺失问题写入missing_dimensions，字段为field/purpose/suggested_question。"
            "只返回JSON对象，根字段只能是candidates和missing_dimensions。候选字段为"
            "disorder/label/symptom_match{matched,total}/core_evidence_summary/evidence_refs/confidence/decision_paths。\n"
            f"展示标签：{json.dumps(catalog, ensure_ascii=False)}\n"
            f"允许的患者证据ref：{json.dumps(allowed, ensure_ascii=False)}\n"
            f"患者画像：{json.dumps(clinical, ensure_ascii=False)}\n"
        )

        def request(text):
            return self.request(text, model=self.model, temperature=0, response_format={"type": "json_object"},
                                extra_body={"enable_thinking": True})

        def validate(raw):
            payload = _canonicalize_payload_evidence_refs(_parse_json_object(raw), clinical, allowed)
            _assert_payload_evidence_refs(payload, allowed)
            if not set(payload) <= {"candidates", "missing_dimensions"} or not isinstance(payload.get("candidates"), list):
                raise ValueError("A0 requires a candidates list and no knowledge/tree fields")
            if any(not isinstance(c, dict) or type(c.get("confidence")) not in (int, float)
                   or not math.isfinite(c["confidence"]) or not 0 <= c["confidence"] <= 1 for c in payload["candidates"]):
                raise ValueError("A0 confidence must be a finite number, not bool")
            candidates = [DiagnosticCandidate.model_validate(c) for c in payload["candidates"]]
            ids = [c.disorder for c in candidates]
            if len(ids) != len(set(ids)) or len(ids) > len(SUPPORTED_DISORDERS):
                raise ValueError("Duplicate or excessive A0 candidates")
            names = {c["disorder"]: c["label"] for c in catalog}
            for candidate in candidates:
                if candidate.label != names[candidate.disorder] or candidate.decision_paths:
                    raise ValueError("A0 requires fixed display labels and empty paths")
                if any(ref not in allowed for ref in candidate.evidence_refs):
                    raise ValueError("Unknown patient evidence")
            missing = [MissingDimension.model_validate(x) for x in payload.get("missing_dimensions", [])]
            return CandidateGeneration(
                considered_disorders=list(SUPPORTED_DISORDERS), candidates=candidates,
                tree_trace=[], missing_dimensions=missing,
                candidate_coverage={key: "candidate" if key in ids else "not_triggered" for key in SUPPORTED_DISORDERS},
            )

        return _request_structured_output_with_one_retry(
            request, validate, prompt,
            prompt + "\n上次JSON或引用不合法。按同一契约重新输出，不添加候选数量，不补造引用。",
        )


def generate_candidates_no_knowledge(case_id, profile, *, request, model, knowledge=None):
    if not isinstance(case_id, str) or not case_id.strip() or not isinstance(profile, dict):
        raise ValueError("case_id and profile required")
    return A0NoKnowledge(request, model=model, knowledge=knowledge).generate_candidates(profile)

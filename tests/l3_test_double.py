from __future__ import annotations

from typing import Any

from layers.layer3.diagnosis.schemas import (
    CandidateGeneration,
    DiagnosticCandidate,
    EliminativeAssessment,
    RewardModelAssessment,
    SUPPORTED_DISORDERS,
)


def _known_ref(profile: dict[str, Any]) -> str:
    items = profile.get("PHQ9_items") or {}
    if "item2" in items:
        return f"PHQ9_item2={items['item2']}"
    for key in ("PHQ9", "GAD7", "PCL5", "Y_BOCS", "ISI", "MDQ"):
        if key in profile:
            return f"{key}={profile[key]}"
    nodes = profile.get("evidence_nodes") or []
    if nodes:
        return str(nodes[0]["evidence_id"])
    raise ValueError("受控L3模型需要至少一个可引用的输入证据")


class ControlledCLINICRReasoner:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []

    def generate_candidates(self, profile: dict[str, Any]) -> CandidateGeneration:
        from tests.tree_test_support import controlled_generation

        self.generate_calls.append(profile)
        required = {"item1", "item2", "item3", "item4", "item6"}
        complete = ["D013"] if required <= set(profile.get("PHQ9_items") or {}) else []
        return controlled_generation(
            ["D013", "D011", "D020", "D009"], ref=_known_ref(profile),
            complete_labels=complete,
            confidences={"D013": .75, "D011": .45, "D020": .35, "D009": .15},
        )


class ControlledRewardModelVerifier:
    def __init__(self, score: float = 0.87) -> None:
        self.score = score
        self.calls: list[dict[str, Any]] = []

    def score_reasoning(
        self,
        profile: dict[str, Any],
        candidate_without_confidence: dict[str, Any],
        assessment: EliminativeAssessment,
        differential_context: list[dict[str, Any]],
    ) -> RewardModelAssessment:
        self.calls.append({
            "candidate": candidate_without_confidence,
            "assessment": assessment.model_dump(mode="json"),
            "differential_context": differential_context,
        })
        ref = _known_ref(profile)
        needs_more = self.score < 0.8
        return RewardModelAssessment.model_validate({
            "score": self.score,
            "dimensions": [
                {
                    "dimension": dimension,
                    "score": self.score,
                    "reason": "受控终验维度评分",
                    "evidence_refs": [ref] if dimension == "evidence_sufficiency" else [],
                }
                for dimension in (
                    "evidence_sufficiency",
                    "logic_coherence",
                    "differential_coverage",
                    "no_reasoning_leaps",
                )
            ],
            "fail_reasons": (
                ["推理质量低于最低门槛"] if self.score < 0.5 else []
            ),
            "missing_dimensions": ([{
                "field": "diagnostic_context.additional_detail",
                "purpose": "补充终验所需的诊断信息",
                "suggested_question": "这些症状持续多久，并造成了哪些具体影响？",
            }] if needs_more else []),
        })


def controlled_clinicr(score: float = 0.87):
    from layers.layer3.diagnosis.clinicr import CLINICR
    from layers.layer3.diagnosis.verifier import IndependentVerifier
    from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider

    reasoner = ControlledCLINICRReasoner()
    verifier_model = ControlledRewardModelVerifier(score=score)
    engine = CLINICR(
        reasoner=reasoner,
        verifier=IndependentVerifier(model=verifier_model),
        knowledge=JsonDiagnosticKnowledgeProvider(),
    )
    return engine, reasoner, verifier_model

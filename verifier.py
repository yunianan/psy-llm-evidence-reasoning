from __future__ import annotations

from typing import Any

from .api import APIRewardModelVerifier
from ._support import VerifierModelUnavailable
from .interfaces import RewardModelVerifier
from .contracts import DiagnosticCandidate, EliminativeAssessment


VERIFIER_PASS_THRESHOLD = 0.8
VERIFIER_CONDITIONAL_THRESHOLD = 0.5
REQUIRED_DIMENSIONS = {
    "evidence_sufficiency",
    "logic_coherence",
    "differential_coverage",
    "no_reasoning_leaps",
}


class IndependentVerifier:
    """Step 3：不读取候选置信度，独立评价排名第一诊断的推理质量。"""

    def __init__(self, model: RewardModelVerifier | None = None) -> None:
        self.model = model or APIRewardModelVerifier()

    def verify(
        self,
        candidate: DiagnosticCandidate,
        assessment: EliminativeAssessment,
        profile: dict[str, Any],
        differential_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate_without_confidence = candidate.model_dump(
            mode="json", exclude={"confidence": True, "decision_paths": {
                "__all__": {"decisions": {"__all__": {"confidence"}}}
            }}
        )
        result = self.model.score_reasoning(
            profile,
            candidate_without_confidence,
            assessment,
            differential_context,
        )

        dimensions = [item.dimension for item in result.dimensions]
        if set(dimensions) != REQUIRED_DIMENSIONS or len(dimensions) != 4:
            raise VerifierModelUnavailable(
                "CLINICR Step 3输出必须且只能包含四个质量维度"
            )

        missing = self._dedupe_missing(
            [item.model_dump(mode="json") for item in result.missing_dimensions]
        )
        fail_reasons = list(dict.fromkeys(result.fail_reasons))

        if result.score >= VERIFIER_PASS_THRESHOLD:
            decision = "confirmed"
            routing = "pass"
        elif result.score >= VERIFIER_CONDITIONAL_THRESHOLD:
            if not missing:
                raise VerifierModelUnavailable(
                    "暂定诊断必须给出供L2继续追问的具体缺失信息"
                )
            decision = "provisional"
            routing = "conditional"
        else:
            if not missing or not fail_reasons:
                raise VerifierModelUnavailable(
                    "拒绝诊断必须同时说明拒绝原因和需补充的信息"
                )
            decision = "rejected"
            routing = "return_to_l2"

        checks = [
            {
                "check_id": item.dimension,
                "status": "pass" if item.score >= VERIFIER_PASS_THRESHOLD else "fail",
                "score": item.score,
                "weight": 0.25,
                "reasons": [item.reason],
                "evidence_refs": item.evidence_refs,
            }
            for item in result.dimensions
        ]
        return {
            "score": result.score,
            "decision": decision,
            "routing": routing,
            "checks": checks,
            "fail_reasons": fail_reasons,
            "missing_dimensions": missing,
        }

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

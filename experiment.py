"""独立运行 Step 3，并形成可保存、可与 L3 对照的实验结果。"""
from typing import Any

from ._support import ALLOWED_EVIDENCE_REFS_KEY, _assert_payload_evidence_refs
from .contracts import RewardModelAssessment, VerificationCase
from .evidence import known_evidence_refs
from .interfaces import RewardModelVerifier
from .verifier import IndependentVerifier


class ReplayScorer:
    """回放已有评分响应，只测试解析与判定，不模拟真实评分能力。"""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = RewardModelAssessment.model_validate(response)

    def score_reasoning(self, profile, candidate_without_confidence, assessment, differential_context):
        return self.response.model_copy(deep=True)


def verify_case(case: VerificationCase, scorer: RewardModelVerifier) -> dict[str, Any]:
    if case.candidate.disorder != case.assessment.disorder:
        raise ValueError("候选与标准核对记录的疾病 ID 不一致")
    allowed_refs = sorted(known_evidence_refs(case.profile))
    profile = {**case.profile, ALLOWED_EVIDENCE_REFS_KEY: allowed_refs}
    verdict = IndependentVerifier(model=scorer).verify(
        case.candidate, case.assessment, profile, case.differential_context,
    )
    # 原系统在 CLINICR 中完成此项检查；独立入口保留相同的引用守卫。
    _assert_payload_evidence_refs(verdict, allowed_refs)
    accepted = verdict["routing"] in {"pass", "conditional"}
    return {
        "case_id": case.case_id,
        **verdict,
        "candidate_confidence": case.candidate.confidence,
        "confidence": verdict["score"],
        "reviewed_candidate": case.candidate.model_dump(mode="json"),
        "diagnosis": {
            "primary": case.candidate.model_dump(mode="json") if accepted else None,
            "secondary": case.secondary_candidates if accepted else [],
        },
        "secondary_verification": "not_performed",
    }

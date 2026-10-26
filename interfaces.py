"""自有奖励模型只需实现该评分接口，不需要依赖候选生成器。"""
from typing import Any, Protocol

from .contracts import EliminativeAssessment, RewardModelAssessment


class RewardModelVerifier(Protocol):
    def score_reasoning(
        self,
        profile: dict[str, Any],
        candidate_without_confidence: dict[str, Any],
        assessment: EliminativeAssessment,
        differential_context: list[dict[str, Any]],
    ) -> RewardModelAssessment:
        ...

"""Fail closed on incomplete screening denials before tree pruning."""
from __future__ import annotations

import logging
from typing import Any

from .schemas import TreeNodeJudgment

logger = logging.getLogger(__name__)
SCREEN_DENIAL_POLICY_VERSION = "screen-denial-scope-v1"


def validate_screen_denial_scope(
    judgment: TreeNodeJudgment, *, stage: str,
) -> tuple[TreeNodeJudgment, dict[str, Any] | None]:
    if stage != "screening" or judgment.status != "not_met" or judgment.denial_scope == "full":
        return judgment, None
    event = {
        "event": "screen_denial_scope_downgrade",
        "node_id": judgment.node_id,
        "original_status": judgment.status,
        "effective_status": "pending",
        "denial_scope": judgment.denial_scope,
        "evidence_refs": list(judgment.evidence_refs),
        "policy_version": SCREEN_DENIAL_POLICY_VERSION,
    }
    logger.warning("screen_denial_scope_downgrade", extra=event)
    from ..llm.observability import emit
    emit(event["event"], **{k: v for k, v in event.items() if k != "event"})
    return judgment.model_copy(update={
        "status": "pending",
        "rationale": (
            f"screen_denial_scope_downgrade：denial_scope={judgment.denial_scope}，"
            f"未确认对整个入口的明确否认，保留待查。原判断依据：{judgment.rationale}"
        ),
    }), event

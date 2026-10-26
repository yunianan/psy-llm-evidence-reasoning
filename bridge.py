"""将已保存的 L3 数据打包为实验输入；不运行或导入 Layer 3。"""
from typing import Any

from .contracts import VerificationCase


def case_from_layer3(
    *,
    case_id: str,
    profile: dict[str, Any],
    result: dict[str, Any],
    source_knowledge: dict[str, Any] | None = None,
) -> VerificationCase:
    diagnosis = result.get("diagnosis") or {}
    selected = result.get("provisional_diagnosis") or diagnosis.get("primary")
    if not selected:
        raise ValueError("L3 结果没有选定的第一候选，无法提取 Step 3 输入")
    disorder = selected["disorder"]
    candidates = [item for item in result.get("candidates", []) if item.get("disorder") == disorder]
    checked = [item for item in result.get("checked", []) if item.get("disorder") == disorder]
    if len(candidates) != 1 or len(checked) != 1:
        raise ValueError("L3 结果缺少唯一的第一候选或其标准核对记录")
    assessment = {key: value for key, value in checked[0].items()
                  if key in {"disorder", "criteria_detail", "conclusion", "conclusion_reason", "missing_dimensions"}}
    candidate = {key: value for key, value in candidates[0].items()
                 if key in {"disorder", "label", "symptom_match", "core_evidence_summary", "evidence_refs", "confidence", "decision_paths"}}
    differential = [
        {key: item[key] for key in ("disorder", "conclusion", "conclusion_reason", "criteria_detail")}
        for item in result.get("checked", []) if item.get("disorder") != disorder
    ]
    return VerificationCase(
        case_id=case_id, profile=profile, candidate=candidate, assessment=assessment,
        differential_context=differential, source_knowledge=source_knowledge,
        secondary_candidates=diagnosis.get("secondary", []),
    )

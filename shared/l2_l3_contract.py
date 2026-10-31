from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RequiredDiagnosticField:
    """L2 必须采集、L3 才能开始诊断的共享字段定义。"""

    field: str
    missing_label: str
    purpose: str
    suggested_question: str

    @property
    def evidence_target(self) -> str:
        return f"diagnostic_context.{self.field}"


L2_L3_REQUIRED_DIAGNOSTIC_FIELDS: tuple[RequiredDiagnosticField, ...] = (
    RequiredDiagnosticField(
        field="symptom_duration_days",
        missing_label="关键症状持续时间",
        purpose="确认主要困扰的持续时间",
        suggested_question="这些最主要的困扰大约持续多久了？",
    ),
    RequiredDiagnosticField(
        field="functional_impairment",
        missing_label="功能影响",
        purpose="确认困扰是否影响工作、学习、人际或日常生活",
        suggested_question="这些情况目前对工作、学习、人际或日常生活造成了哪些影响？",
    ),
    RequiredDiagnosticField(
        field="mania_or_hypomania_history",
        missing_label="躁狂或轻躁狂史",
        purpose="补充鉴别双相障碍所需的既往情绪高涨与精力变化信息",
        suggested_question=(
            "在这次低落之前或更早，是否有过连续几天异常兴奋、精力明显增多，"
            "或者睡得很少却不觉得疲倦的时期？"
        ),
    ),
    RequiredDiagnosticField(
        field="substance_or_medical_cause",
        missing_label="物质、药物或躯体疾病影响",
        purpose="补充 DSM 排除物质、药物或躯体疾病所致症状需要的信息",
        suggested_question=(
            "这些变化是否可能与饮酒、其他物质、正在使用的药物，"
            "或者身体疾病有关？"
        ),
    ),
    RequiredDiagnosticField(
        field="psychotic_symptoms",
        missing_label="精神病性症状筛查",
        purpose="补充鉴别精神病性症状所需的信息",
        suggested_question=(
            "最近是否出现过别人听不到的声音、明显感觉自己正被伤害或监视，"
            "或者难以分辨现实的经历？"
        ),
    ),
)

REQUIRED_DIAGNOSTIC_FIELD_BY_NAME = {
    item.field: item for item in L2_L3_REQUIRED_DIAGNOSTIC_FIELDS
}
REQUIRED_DIAGNOSTIC_FIELD_BY_LABEL = {
    item.missing_label: item for item in L2_L3_REQUIRED_DIAGNOSTIC_FIELDS
}


def requirement_value(context: Any, field: str):
    get = context.get if isinstance(context, Mapping) else lambda name, default=None: getattr(context, name, default)
    if field == "symptom_duration_days":
        return get("duration_evidence") or get(field)
    return get(field)


def missing_required_context_fields(context: Any) -> list[RequiredDiagnosticField]:
    """检查 L2 DiagnosticContext；False 与 0 都是有效且必须保留的回答。"""

    return [
        item
        for item in L2_L3_REQUIRED_DIAGNOSTIC_FIELDS
        if requirement_value(context, item.field) is None
    ]


def missing_required_profile_fields(
    profile: Mapping[str, Any],
) -> list[RequiredDiagnosticField]:
    """检查已经转换为 L3 输入字典的同一组字段。"""

    return [
        item
        for item in L2_L3_REQUIRED_DIAGNOSTIC_FIELDS
        if requirement_value(profile, item.field) is None
    ]


def missing_required_profile_evidence(
    profile: Mapping[str, Any],
) -> list[RequiredDiagnosticField]:
    """检查 L3 输入是否携带每个必填字段对应的 L2 证据节点。"""

    targets = {
        node.get("target_path")
        for node in profile.get("evidence_nodes", [])
        if isinstance(node, Mapping)
    }
    return [
        item
        for item in L2_L3_REQUIRED_DIAGNOSTIC_FIELDS
        if requirement_value(profile, item.field) is not None and item.evidence_target not in targets
        and not (item.field == "symptom_duration_days" and _duration_refs_valid(profile))
    ]


def _duration_refs_valid(profile: Mapping[str, Any]) -> bool:
    nodes = {n.get("evidence_id"): n for n in profile.get("evidence_nodes", []) if isinstance(n, Mapping)}
    records = profile.get("duration_evidence") or []
    return bool(records) and all(isinstance(r, Mapping) and r.get("evidence_refs")
        and all(ref in nodes and str(r.get("raw_span", "")) in str(nodes[ref].get("quote", ""))
                for ref in r["evidence_refs"]) for r in records)


def as_missing_dimension(item: RequiredDiagnosticField) -> dict[str, str]:
    return {
        "field": item.field,
        "purpose": item.purpose,
        "suggested_question": item.suggested_question,
    }

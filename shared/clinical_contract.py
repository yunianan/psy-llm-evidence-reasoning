"""L2/L3共享事实与补采协议；医学判据留在知识来源中。"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "l2-l3-clinical-v2"
TEMPORAL_POLICY_VERSION = "duration-interval-v1"
FactAttribute = Literal["existence", "onset", "duration", "frequency", "severity", "impact", "trigger", "denial", "clarification"]
STRUCTURED_FIELD_NAMES = (
    "symptom_duration_days", "functional_impairment", "mania_or_hypomania_history", "substance_or_medical_cause",
    "differential_diagnoses_assessed", "evidence_conflicts", "diagnostic_details", "relapse_count",
    "rhythm_disturbance", "exposure_avoidant", "ritual_behaviors", "insomnia_chief_complaint", "chronic_months",
    "interpersonal_conflicts", "interpersonal_impairment", "single_focus", "psychotic_symptoms", "substance_active",
    "severe_dissociation", "motivation_importance", "motivation_confidence", "motivation_readiness",
    "scale_consent", "scale_answer_score", "confirmation_decision",
)
StructuredFieldName = Literal[*STRUCTURED_FIELD_NAMES]


class DurationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptom_id: str = ""
    scope: str = "unknown"
    raw_span: str = Field(min_length=1)
    min_days: float | None = Field(default=None, ge=0)
    max_days: float | None = Field(default=None, ge=0)
    unit: Literal["day", "week", "month", "year", "relative", "unknown"] = "unknown"
    quantity: float | None = Field(default=None, ge=0)
    approx: bool = False
    approximation: Literal["none", "around", "more", "under", "near", "colloquial"] = "none"
    anchor: str = "relative_to_interview"
    interview_date: date | None = None
    pattern: Literal["continuous", "episodic", "unknown"] = "unknown"
    duration_kind: Literal["overall", "episode", "unknown"] = "overall"
    parse_status: Literal["ok", "vague", "failed", "unresolved"] = "failed"
    source: Literal["L2", "L3_writeback", "legacy_adapter"] = "L2"
    evidence_refs: list[str] = Field(default_factory=list)
    policy_version: str = TEMPORAL_POLICY_VERSION

    @model_validator(mode="after")
    def valid_interval(self):
        if self.min_days is not None and self.max_days is not None and self.min_days > self.max_days:
            raise ValueError("病程区间上下界颠倒")
        if self.parse_status == "ok" and self.min_days is None and self.max_days is None:
            raise ValueError("成功解析的病程必须有区间")
        return self


class TemporalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["met", "not_met", "unknown", "met_approx"]
    threshold_days: float | None = None
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)
    suggested_question: str = "大概从几月份开始的？期间一直如此，还是反复发作？"
    policy_version: str = TEMPORAL_POLICY_VERSION


class ClinicalFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_id: str
    symptom_id: str = ""
    attribute: FactAttribute
    state: Literal["present", "absent", "unknown", "conflict"] = "unknown"
    value: Any = None
    raw_span: str
    evidence_refs: list[str] = Field(default_factory=list)
    duration: DurationEvidence | None = None


class FollowUpTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    field: str
    node_id: str = ""
    symptom_id: str = ""
    attribute: FactAttribute = "clarification"
    purpose: str = ""
    suggested_question: str
    priority: str = "remaining"
    required_by: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    expected_by_candidate: dict[str, str] = Field(default_factory=dict)
    status: Literal["pending", "answered", "unresolved", "conflict"] = "pending"
    answer_evidence_refs: list[str] = Field(default_factory=list)


def follow_up_task(item: dict[str, Any]) -> FollowUpTask:
    field = str(item.get("field") or "diagnostic_evidence")
    node_id = str(item.get("node_id") or field.removeprefix("decision_tree."))
    attribute = item.get("attribute") or ("duration" if "duration" in field else "clarification")
    return FollowUpTask(task_id=str(item.get("task_id") or field), field=field, node_id=node_id,
        symptom_id=str(item.get("symptom_id") or ""), attribute=attribute,
        purpose=str(item.get("purpose") or ""),
        suggested_question=str(item.get("suggested_question") or "请补充这方面的信息。"),
        priority=str(item.get("priority") or "remaining"), required_by=list(item.get("required_by") or []),
        source_refs=list(item.get("source_refs") or []), expected_by_candidate=dict(item.get("expected_by_candidate") or {}))

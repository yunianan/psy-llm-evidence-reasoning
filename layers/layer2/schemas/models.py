from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from shared.clinical_contract import CONTRACT_VERSION, ClinicalFact, DurationEvidence, FollowUpTask, StructuredFieldName
from shared.clinical_profile import ClinicalProfile


MIStage = Literal["engaging", "focusing", "evoking", "planning"]
InputType = Literal[
    "free_narrative",
    "focus_selection",
    "scale_consent",
    "scale_answer",
    "goal_statement",
    "motivation_rating",
    "profile_confirmation",
    "correction",
    "refusal",
    "pause",
    "off_topic",
]
ExpectedDialogueInput = Literal[
    "auto",
    "free_text",
    "focus_selection",
    "scale_consent",
    "scale_answer",
    "patient_goals",
    "motivation",
    "profile_confirmation",
    "symptom_duration_days",
    "functional_impairment",
    "mania_or_hypomania_history",
    "substance_or_medical_cause",
    "psychotic_symptoms",
    "differential_diagnoses_assessed",
    "evidence_conflicts",
]
ScaleName = Literal["PHQ-9", "GAD-7", "PCL-5", "Y-BOCS", "ISI", "MDQ"]
ConfirmationDecision = Literal["continue", "revise", "pause"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SemanticCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    evidence_quote: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value", "evidence_quote")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("候选值和证据原话不能为空")
        return value


class StructuredFieldCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: StructuredFieldName
    value: Any
    evidence_quote: str

    @model_validator(mode="after")
    def field_value_type(self):
        # 自然语言病程合法保留，后续确定性解析，不能因类型不符静默丢弃。
        if self.field == "symptom_duration_days" and not (
            isinstance(self.value, (str, dict)) or type(self.value) is int
        ):
            raise ValueError("病程必须为原文、结构化病程或整数兼容值")
        if self.field in ClinicalProfile.model_fields and self.field != "symptom_duration_days":
            ClinicalProfile.model_validate({self.field: self.value})
        if self.field.startswith("motivation_") and not (type(self.value) is int and 0 <= self.value <= 10):
            raise ValueError("动机评分必须为0至10整数")
        if self.field == "scale_consent" and type(self.value) is not bool:
            raise ValueError("量表同意必须为布尔值")
        if self.field == "scale_answer_score" and type(self.value) is not int:
            raise ValueError("量表逐题答案必须为整数")
        if self.field == "confirmation_decision" and self.value not in ("continue", "revise", "pause"):
            raise ValueError("画像确认值无效")
        return self

    @field_validator("field", "evidence_quote")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("结构化字段和证据原话不能为空")
        return value


class CorrectionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_path: str
    action: Literal["remove", "replace"]
    value: Any = None
    evidence_quote: str

    @field_validator("target_path", "evidence_quote")
    @classmethod
    def correction_text_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("修正目标和证据原话不能为空")
        return value


class SemanticAnalysis(BaseModel):
    """语义分析 API 的严格输出；这里的内容尚未成为患者事实。"""

    model_config = ConfigDict(extra="forbid")

    input_type: InputType
    symptoms: list[SemanticCandidate] = Field(default_factory=list)
    emotions: list[SemanticCandidate] = Field(default_factory=list)
    cognitive_patterns: list[SemanticCandidate] = Field(default_factory=list)
    patient_goals: list[SemanticCandidate] = Field(default_factory=list)
    key_narratives: list[SemanticCandidate] = Field(default_factory=list)
    functional_impairments: list[SemanticCandidate] = Field(default_factory=list)
    focus_topics: list[SemanticCandidate] = Field(default_factory=list)
    change_talk: list[SemanticCandidate] = Field(default_factory=list)
    sustain_talk: list[SemanticCandidate] = Field(default_factory=list)
    possible_scale_domains: list[ScaleName] = Field(default_factory=list)
    structured_fields: list[StructuredFieldCandidate] = Field(default_factory=list)
    corrections: list[CorrectionCandidate] = Field(default_factory=list)
    clarification_needed: list[str] = Field(default_factory=list)


class EvidenceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_type: Literal["patient_message", "scale_item", "benchmark_narrative"]
    message_id: str
    turn_index: int = Field(ge=1)
    quote: str
    target_path: str
    created_by: Literal["llm_candidate_rule_verified", "scale_rule", "offline_source_adapter"]
    parent_ids: list[str] = Field(default_factory=list)


class SymptomEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptom: str
    sources: list[str] = Field(min_length=1)
    patient_quotes: list[str] = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    attribute_facts: list[ClinicalFact] = Field(default_factory=list)


class ProfileFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    sources: list[str] = Field(min_length=1)
    patient_quotes: list[str] = Field(min_length=1)


class MotivationScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    importance: int | None = Field(default=None, ge=0, le=10)
    confidence: int | None = Field(default=None, ge=0, le=10)
    readiness: int | None = Field(default=None, ge=0, le=10)
    sources: list[str] = Field(default_factory=list)


class ScaleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_name: ScaleName
    status: Literal["not_started", "pending_consent", "active", "completed", "declined", "skipped"] = "not_started"
    items: dict[int, int] = Field(default_factory=dict)
    score: int | None = None
    threshold: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    skip_reason: str = ""


class PendingScaleItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_name: ScaleName
    item_number: int = Field(ge=1)
    prompt: str
    min_score: int
    max_score: int
    anchors: dict[int, str]


class DiagnosticContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptom_duration_days: int | None = Field(default=None, ge=0)
    duration_evidence: list[DurationEvidence] = Field(default_factory=list)
    functional_impairment: bool | None = None
    mania_or_hypomania_history: bool | None = None
    substance_or_medical_cause: bool | None = None
    differential_diagnoses_assessed: list[str] | None = None
    evidence_conflicts: list[dict[str, Any]] | None = None
    relapse_count: int | None = Field(default=None, ge=0)
    rhythm_disturbance: bool | None = None
    exposure_avoidant: bool | None = None
    ritual_behaviors: bool | None = None
    insomnia_chief_complaint: bool | None = None
    chronic_months: int | None = Field(default=None, ge=0)
    interpersonal_conflicts: int | None = Field(default=None, ge=0)
    interpersonal_impairment: bool | None = None
    single_focus: bool | None = None
    psychotic_symptoms: bool | None = None
    substance_active: bool | None = None
    severe_dissociation: bool | None = None
    diagnostic_details: dict[str, Any] = Field(default_factory=dict)
    sources: dict[str, list[str]] = Field(default_factory=dict)


class L2SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    contract_version: str = CONTRACT_VERSION
    interview_date: str | None = None
    clinical_facts: list[ClinicalFact] = Field(default_factory=list)
    follow_up_tasks: list[FollowUpTask] = Field(default_factory=list)
    current_stage: MIStage = "engaging"
    turn_index: int = 0
    safety_passport: dict[str, Any]
    active_safety_controls: list[str] = Field(default_factory=list)
    messages: list[dict[str, str]] = Field(default_factory=list)
    consumed_l1_message_ids: list[str] = Field(default_factory=list)
    symptoms: dict[str, SymptomEvidence] = Field(default_factory=dict)
    emotions: list[ProfileFact] = Field(default_factory=list)
    cognitive_patterns: list[ProfileFact] = Field(default_factory=list)
    patient_goals: list[ProfileFact] = Field(default_factory=list)
    key_narratives: list[ProfileFact] = Field(default_factory=list)
    functional_impairments: list[ProfileFact] = Field(default_factory=list)
    focus_topics: list[ProfileFact] = Field(default_factory=list)
    change_talk: list[ProfileFact] = Field(default_factory=list)
    sustain_talk: list[ProfileFact] = Field(default_factory=list)
    motivation: MotivationScores = Field(default_factory=MotivationScores)
    scale_results: dict[str, ScaleResult] = Field(default_factory=dict)
    pending_scale_name: ScaleName | None = None
    evidence_nodes: list[EvidenceNode] = Field(default_factory=list)
    diagnostic_context: DiagnosticContext = Field(default_factory=DiagnosticContext)
    confirmation_status: Literal["not_requested", "pending", "confirmed", "revision_requested", "paused"] = "not_requested"
    missing_dimensions: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class StageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_before: MIStage
    stage_after: MIStage
    transition_reason: str
    response_goal: str
    oars_techniques: list[Literal["open_question", "affirmation", "reflection", "complex_reflection", "summary"]]
    awaiting: ExpectedDialogueInput = "auto"
    required_question: str = ""


class L2Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_before: MIStage
    stage_after: MIStage
    input_type: InputType
    triggered_scales: list[ScaleName] = Field(default_factory=list)
    pending_scale_item: PendingScaleItem | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    transition_reason: str
    response_goal: str


class DialogueResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reply: str
    awaiting: str | None = None
    l2_turn: L2Turn | None = None
    error_code: str | None = None
    confirmation_decision: ConfirmationDecision | None = None


class ClinicalIntake(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_id: Literal["L2"] = "L2"
    timestamp: str = Field(default_factory=utc_now)
    session_id: str
    contract_version: str = CONTRACT_VERSION
    interview_date: str | None = None
    clinical_facts: list[ClinicalFact] = Field(default_factory=list)
    follow_up_tasks: list[FollowUpTask] = Field(default_factory=list)
    current_stage: MIStage
    key_symptoms_extracted: list[SymptomEvidence] = Field(default_factory=list)
    emotions: list[ProfileFact] = Field(default_factory=list)
    cognitive_patterns: list[ProfileFact] = Field(default_factory=list)
    patient_goals: list[str] = Field(default_factory=list)
    patient_goal_facts: list[ProfileFact] = Field(default_factory=list)
    key_narratives: list[ProfileFact] = Field(default_factory=list)
    functional_impairments: list[ProfileFact] = Field(default_factory=list)
    focus_topics: list[str] = Field(default_factory=list)
    focus_topic_facts: list[ProfileFact] = Field(default_factory=list)
    change_talk: list[ProfileFact] = Field(default_factory=list)
    sustain_talk: list[ProfileFact] = Field(default_factory=list)
    motivation: MotivationScores
    scale_results: dict[str, ScaleResult] = Field(default_factory=dict)
    diagnostic_context: DiagnosticContext
    evidence_nodes: list[EvidenceNode] = Field(default_factory=list)
    information_sufficiency: Literal["insufficient", "sufficient"]
    missing_dimensions: list[str] = Field(default_factory=list)
    patient_confirmation: Literal["not_requested", "pending", "confirmed", "revision_requested", "paused"]
    safety_controls: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    l2_validation: dict[str, Any] = Field(default_factory=dict)

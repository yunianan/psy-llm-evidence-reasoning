from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InstructionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

Substage = Literal["L4-L1", "L4-L2", "L4-L3", "L4-L4"]


class RoleInstruction(InstructionModel):
    """L1 角色定义（全会话复用）。"""

    identity: str
    style: str
    boundaries: list[str] = Field(default_factory=list)


class DiagnosticAnchor(InstructionModel):
    """L3 生成的会话级锚点；L4 只消费，不重新推导。"""

    diagnostic_standard: Literal["WHO ICD-10"] = "WHO ICD-10"
    primary: str = ""
    primary_label: str = ""
    comorbidities: list[str] = []
    excluded: list[str] = []
    core_beliefs: list[str] = []
    plugins: list[str] = []
    plugin_triggers: list[dict[str, Any]] = []
    strategy_adjustments: dict[str, Any] = {}
    contraindications: list[str] = []
    confidence: float | None = None
    candidate_confidence: float | None = 0.0
    verifier_score: float | None = None
    therapy_type: str = ""
    verifier_decision: str = ""
    diagnosis_details: dict[str, Any] = Field(default_factory=dict)
    scale_scores: dict[str, float] = Field(default_factory=dict)
    cognitive_patterns: list[dict[str, Any]] = Field(default_factory=list)
    triggering_situations: list[dict[str, Any]] = Field(default_factory=list)
    key_narratives: list[dict[str, Any]] = Field(default_factory=list)
    patient_goals: list[str] = Field(default_factory=list)
    allowed_techniques: dict[str, str] = Field(default_factory=dict)
    wording_constraint: str = "使用初步印象措辞，禁止确诊"


class SourceQuote(InstructionModel):
    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ProgressObservation(InstructionModel):
    kind: Literal["patient_response", "homework", "unresolved"]
    interpretation: str = Field(min_length=1)
    evidence: SourceQuote


class TurnInstruction(InstructionModel):
    """L3 本轮动态指令（每轮重新生成）。"""

    goal: str
    technique: str
    forbidden: list[str] = []
    link_previous: str = ""
    plugin_guidance: str | None = None
    force_substage: Substage | None = None
    session_id: str = ""
    turn_number: int = Field(default=0, ge=0)
    source_message: str = ""
    technique_code: str = ""
    plugin_steps: dict[str, str] = Field(default_factory=dict)
    evidence: list[SourceQuote] = Field(default_factory=list)
    observations: list[ProgressObservation] = Field(default_factory=list)
    previous_goal_met: bool = Field(default=False, strict=True)
    patient_ready: bool = Field(default=False, strict=True)
    rationale: str = ""


class BoundaryInstruction(InstructionModel):
    """L4 防越界指令（每轮复用）。"""

    absolute_bans: list[str] = []
    safety_trigger: str = ""


class TreatmentInstructions(InstructionModel):
    """会话建立时 L3 为空；每次生成回复前必须有绑定本轮的 L3。"""

    L1: RoleInstruction
    L2: DiagnosticAnchor
    L3: TurnInstruction | None = None
    L4: BoundaryInstruction


class TreatmentTurnRecord(InstructionModel):
    instruction: TurnInstruction
    user_message: str
    reply: str
    fallback_used: bool = False


class SafetyPassport(BaseModel):
    """L1 安全通行证中 L4 需要的部分（每轮复用）。"""

    risk_level: int = 1
    sensitive_topics: list[str] = []
    flags: list[str] = []

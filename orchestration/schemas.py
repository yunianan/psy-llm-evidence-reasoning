from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from layers.layer2.schemas import ClinicalIntake, ExpectedDialogueInput
from layers.layer4.schemas.l3_report import DiagnosticReport


PipelineRoute = Literal[
    "L1_SAFETY_ASSESSMENT",
    "L1_SAFETY_HOLD",
    "L2_INTERVIEW",
    "L2_SCALE",
    "L2_DIAGNOSTIC_CONTEXT",
    "L3_PROCESSING",
    "RETURN_TO_L2",
    "MANUAL_REVIEW",
    "AWAITING_PATIENT_CONFIRMATION",
    "L4",
    "PAUSED_BY_PATIENT",
    "STOPPED",
]

ConfirmationDecision = Literal["continue", "revise", "pause"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PatientConfirmation(BaseModel):
    """患者可见的脱敏确认卡，不包含内部评分、规则或证据编号。"""

    model_config = ConfigDict(extra="forbid")

    title: str
    profile_summary: list[str] = Field(default_factory=list)
    scale_summary: list[str] = Field(default_factory=list)
    initial_impression: str
    disclaimer: str
    next_step: str
    actions: list[Literal["continue", "revise", "pause"]] = [
        "continue", "revise", "pause"
    ]


class OrchestrationOutcome(BaseModel):
    """患者/API 可见的统一输出；刻意排除开发者画像和内部临床对象。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    route: PipelineRoute
    reply: str
    waiting_for: str | None = None
    terminal: bool = False
    confirmation: PatientConfirmation | None = None


class PipelineSession(BaseModel):
    """只保存在编排层的全局状态。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    route: PipelineRoute = "L2_INTERVIEW"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    l2_session_id: str | None = None
    l4_session_id: str | None = None
    expected_input: ExpectedDialogueInput = "auto"
    deferred_l2_reply: str = ""
    deferred_l2_expected_input: ExpectedDialogueInput | None = None
    l1_passport: dict[str, Any] | None = None
    clinical_intake: ClinicalIntake | None = None
    diagnostic_report: DiagnosticReport | None = None
    l3_raw: dict[str, Any] | None = None
    pending_l5_review: dict[str, Any] | None = None
    patient_confirmation: PatientConfirmation | None = None
    stopped_reason: str = ""
    internal_flags: list[str] = Field(default_factory=list)
    latest_developer_profile_path: str | None = None

    def touch(self) -> None:
        self.updated_at = utc_now()

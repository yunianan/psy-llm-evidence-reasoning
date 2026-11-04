from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from layers.layer4.schemas.instructions import DiagnosticAnchor
from shared.schemas.treatment_instructions import TreatmentInstructions


class ICD10EvidenceQuote(BaseModel):
    message_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    source_role: Literal["patient", "collateral", "clinical_record"]


class ICD10CriterionEvidence(BaseModel):
    node_id: str
    answer: Literal["yes", "no", "unknown"]
    source: dict[str, Any]
    question: str = ""
    original_clause: str = ""
    evidence: list[ICD10EvidenceQuote] = Field(default_factory=list)
    calculation: dict[str, Any] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class DiagnosticDisplay(BaseModel):
    name: str = ""
    label: str = ""
    display_group: str = ""
    display_group_label: str = ""
    display_name: str = ""
    knowledge_source: str = ""


class PrimaryDiagnosis(DiagnosticDisplay):
    disorder: str
    confidence: float | None = 0.0
    severity: str = ""
    evidence: str = ""
    criteria_evidence: list[ICD10CriterionEvidence] = Field(default_factory=list)


class SecondaryDiagnosis(DiagnosticDisplay):
    disorder: str
    confidence: float | None = 0.0
    evidence: str = ""
    criteria_evidence: list[ICD10CriterionEvidence] = Field(default_factory=list)


class RuledOutDiagnosis(DiagnosticDisplay):
    disorder: str
    reason: str = ""
    evidence_ref: str = ""


class Diagnosis(BaseModel):
    primary: PrimaryDiagnosis | None = None
    secondary: list[SecondaryDiagnosis] = []
    ruled_out: list[RuledOutDiagnosis] = []


class RecommendedTherapy(BaseModel):
    primary: str = ""
    addon: list[str] = []
    plugin_triggers: list[dict[str, Any]] = Field(default_factory=list)
    strategy_adjustments: dict[str, Any] = Field(default_factory=dict)
    contraindications: list[str] = []


class ICD10ClinicalReview(BaseModel):
    """Input from the trusted clinical-review caller for one tree run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tree_session_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    primary_code: str = Field(min_length=1)
    secondary_codes: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    severity_level: int = Field(ge=1, le=3, strict=True)
    therapy: RecommendedTherapy


class L3ValidationCheck(BaseModel):
    rule_id: str
    status: Literal["pass", "fail"]
    detail: str = ""


class L3ValidationReceipt(BaseModel):
    passed: bool
    route: str
    checks: list[L3ValidationCheck] = Field(default_factory=list)
    payload_hash: str
    policy_version: str


class DiagnosticReport(BaseModel):
    """接口三。将经复核的 WHO ICD-10 诊断与治疗指令交给 L4。"""

    diagnostic_standard: Literal["WHO ICD-10"]
    icd10_catalog_id: str = ""
    icd10_review: ICD10ClinicalReview | None = None
    candidate_generation_method: Literal["icd10_tree"] = "icd10_tree"
    layer_id: str = "L3"
    timestamp: str = ""
    diagnosis: Diagnosis = Diagnosis()
    verifier_score: None = None
    verifier_decision: str = ""
    candidate_confidence: float | None = 0.0
    verifier_routing: str = ""
    recommended_therapy: RecommendedTherapy = RecommendedTherapy()
    severity_level: int = 1
    l3_validation: L3ValidationReceipt | None = None
    instructions: dict[str, Any] = Field(default_factory=dict)

    def to_anchor(self) -> DiagnosticAnchor:
        return TreatmentInstructions.model_validate(self.instructions).L2

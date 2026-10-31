from typing import Any, Literal

from pydantic import BaseModel, Field

from layers.layer4.schemas.instructions import DiagnosticAnchor
from shared.schemas.treatment_instructions import TreatmentInstructions


class DSMCriterionEvidence(BaseModel):
    code: str
    name: str
    status: Literal["met", "not_met", "unknown"]
    met: bool
    evidence_refs: list[str] = Field(default_factory=list)


class DiagnosticDisplay(BaseModel):
    name: str = ""
    label: str = ""
    display_group: str = ""
    display_group_label: str = ""
    display_name: str = ""
    knowledge_source: str = ""


class PrimaryDiagnosis(DiagnosticDisplay):
    disorder: str
    confidence: float = 0.0
    severity: str = ""
    evidence: str = ""
    dsm5_criteria_met: list[str] = []
    dsm5_criteria_detail: list[DSMCriterionEvidence] = Field(default_factory=list)


class SecondaryDiagnosis(DiagnosticDisplay):
    disorder: str
    confidence: float = 0.0
    evidence: str = ""


class RuledOutDiagnosis(DiagnosticDisplay):
    disorder: str
    reason: str = ""
    evidence_ref: str = ""


class UnderObservationDiagnosis(DiagnosticDisplay):
    disorder: str
    confidence: float = 0.0
    reason: str = ""
    missing_dimensions: list[dict] = Field(default_factory=list)


class Diagnosis(BaseModel):
    primary: PrimaryDiagnosis | None = None
    secondary: list[SecondaryDiagnosis] = []
    under_observation: list[UnderObservationDiagnosis] = Field(default_factory=list)
    ruled_out: list[RuledOutDiagnosis] = []


class RecommendedTherapy(BaseModel):
    primary: str = ""
    addon: list[str] = []
    plugin_triggers: list[dict[str, Any]] = Field(default_factory=list)
    strategy_adjustments: dict[str, Any] = Field(default_factory=dict)
    contraindications: list[str] = []


class TherapyRouting(BaseModel):
    primary: str = ""
    addon: list[str] = []
    plugin_triggers: list[dict[str, Any]] = Field(default_factory=list)
    strategy_adjustments: dict[str, Any] = Field(default_factory=dict)
    l4_substage_priority: list[str] = []


class StructuredLabels(BaseModel):
    therapy_routing: TherapyRouting = TherapyRouting()
    wording_constraint: str = ""
    risk_monitoring: list[str] = []


class EvidenceRefIn(BaseModel):
    type: str = ""
    disorder: str = ""
    criteria_met: str = ""
    source_doc: str = ""
    threshold: str = ""
    therapy: str = ""
    source: str = ""
    criterion_code: str = ""
    criterion_name: str = ""
    status: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class MissingDimension(BaseModel):
    field: str
    purpose: str = ""
    suggested_question: str = ""


class VerifierCheck(BaseModel):
    check_id: str
    status: Literal["pass", "fail"]
    score: float
    weight: float
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


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
    """接口三：L3 → L4 DiagnosticReport（诊断报告）。未知字段忽略，真实 L3 完整 JSON 可解析。"""

    diagnostic_standard: Literal["DSM-5"] = "DSM-5"
    supported_disorders: list[str] = Field(default_factory=list)
    diagnostic_catalog: list[dict[str, Any]] = Field(default_factory=list)
    considered_disorders: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    checked: list[dict[str, Any]] = Field(default_factory=list)
    candidate_generation_method: str = "external_reasoner"
    tree_trace: list[dict[str, Any]] = Field(default_factory=list)
    candidate_coverage: dict[str, str] = Field(default_factory=dict)
    candidate_missing_dimensions: list[dict[str, str]] = Field(default_factory=list)
    diagnostic_readiness: dict[str, Any] = Field(default_factory=dict)
    information_collection_plan: dict[str, Any] = Field(default_factory=dict)
    duration_evidence: list[dict[str, Any]] = Field(default_factory=list)
    temporal_audit: list[dict[str, Any]] = Field(default_factory=list)
    layer_id: str = "L3"
    timestamp: str = ""
    diagnosis: Diagnosis = Diagnosis()
    provisional_diagnosis: PrimaryDiagnosis | None = None
    verifier_score: float = 0.0
    verifier_decision: str = ""
    candidate_confidence: float = 0.0
    verifier_routing: str = ""
    verifier_checks: list[VerifierCheck] = Field(default_factory=list)
    verifier_fail_reasons: list[str] = Field(default_factory=list)
    recommended_therapy: RecommendedTherapy = RecommendedTherapy()
    severity_level: int = 1
    clinical_notes_for_L4: str = ""
    summary: str = ""
    structured_labels: StructuredLabels = StructuredLabels()
    evidence_refs: list[EvidenceRefIn] = []
    confidence: float = 0.0
    flags: list[str] = []
    instruction_to_next: str = ""
    missing_dimensions: list[MissingDimension] = []
    l3_validation: L3ValidationReceipt | None = None
    instructions: dict[str, Any] = Field(default_factory=dict)

    def to_anchor(self) -> DiagnosticAnchor:
        return TreatmentInstructions.model_validate(self.instructions).L2

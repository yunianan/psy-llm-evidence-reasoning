"""L2序列化与L3两种入口共用同一患者画像类型。"""
from typing import Any, Literal
from datetime import date
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator
from shared.clinical_contract import CONTRACT_VERSION, ClinicalFact, DurationEvidence, FollowUpTask


class ClinicalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    interview_date: date | None = None
    clinical_facts: list[ClinicalFact] = Field(default_factory=list)
    duration_evidence: list[DurationEvidence] = Field(default_factory=list)
    follow_up_tasks: list[FollowUpTask] = Field(default_factory=list)
    temporal_audit: list[dict[str, Any]] = Field(default_factory=list)
    PHQ9: float | None = Field(default=None, ge=0, le=27)
    GAD7: float | None = Field(default=None, ge=0, le=21)
    MDQ: float | None = Field(default=None, ge=0, le=13)
    PCL5: float | None = Field(default=None, ge=0, le=80)
    ISI: float | None = Field(default=None, ge=0, le=28)
    Y_BOCS: float | None = Field(default=None, ge=0, le=40)
    ADNM: float | None = Field(default=None, ge=0)
    RRS: float | None = Field(default=None, ge=0)
    AAQ_II: float | None = Field(default=None, ge=0)
    PHQ15: float | None = Field(default=None, ge=0, le=30)
    PHQ9_items: dict[str, float] = Field(default_factory=dict)
    GAD7_items: dict[str, float] = Field(default_factory=dict)
    MDQ_items: dict[str, float] = Field(default_factory=dict)
    PCL5_items: dict[str, float] = Field(default_factory=dict)
    ISI_items: dict[str, float] = Field(default_factory=dict)
    Y_BOCS_items: dict[str, float] = Field(default_factory=dict)
    ADNM_items: dict[str, float] = Field(default_factory=dict)
    RRS_items: dict[str, float] = Field(default_factory=dict)
    AAQ_II_items: dict[str, float] = Field(default_factory=dict)
    PHQ15_items: dict[str, float] = Field(default_factory=dict)
    relapse_count: StrictInt | None = Field(default=None, ge=0)
    chronic_months: StrictInt | None = Field(default=None, ge=0)
    interpersonal_conflicts: StrictInt | None = Field(default=None, ge=0)
    symptom_duration_days: StrictInt | None = Field(default=None, ge=0)
    functional_impairment: StrictBool | None = None
    mania_or_hypomania_history: StrictBool | None = None
    substance_or_medical_cause: StrictBool | None = None
    rhythm_disturbance: StrictBool | None = None
    exposure_avoidant: StrictBool | None = None
    ritual_behaviors: StrictBool | None = None
    insomnia_chief_complaint: StrictBool | None = None
    interpersonal_impairment: StrictBool | None = None
    single_focus: StrictBool | None = None
    psychotic_symptoms: StrictBool | None = None
    substance_active: StrictBool | None = None
    severe_dissociation: StrictBool | None = None
    differential_diagnoses_assessed: list[str] | None = None
    evidence_conflicts: list[dict[str, Any]] | None = None
    diagnostic_details: dict[str, Any] = Field(default_factory=dict)
    key_symptoms_extracted: list[dict[str, Any]] = Field(default_factory=list)
    emotions: list[dict[str, Any]] = Field(default_factory=list)
    cognitive_patterns: list[dict[str, Any]] = Field(default_factory=list)
    key_narratives: list[dict[str, Any]] = Field(default_factory=list)
    functional_impairments: list[dict[str, Any]] = Field(default_factory=list)
    patient_goals: list[str] = Field(default_factory=list)
    motivation: dict[str, Any] = Field(default_factory=dict)
    evidence_nodes: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def canonical_fact_sources(self):
        nodes = {n.get("evidence_id"): n for n in self.evidence_nodes if n.get("evidence_id")}
        for fact in [*self.duration_evidence, *self.clinical_facts]:
            if not fact.evidence_refs or any(ref not in nodes or fact.raw_span not in nodes[ref].get("quote", "")
                                             for ref in fact.evidence_refs):
                raise ValueError("共享事实必须绑定存在且包含原文片段的患者证据")
        return self

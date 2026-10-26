from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


from shared.disorder_catalog import DISORDER_COUNT, SUPPORTED_DISORDERS

SupportedDisorder = Literal[*SUPPORTED_DISORDERS]


class MissingDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    suggested_question: str = Field(min_length=1)


class TreeNodeJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    status: Literal["met", "not_met", "pending"]
    rationale: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    symptom_ids: list[str] = Field(
        default_factory=list,
        description="仅 met 可携带正症状 ID；pending/not_met 必须为 []，不把未满足节点计为正症状",
    )
    confidence: float = Field(default=0.5, ge=0, le=1)
    denial_scope: Literal["full", "partial", "none", "unknown"] = Field(
        default="unknown",
        description="screening 入口的否认范围；full 仅表示整体明确否认，非 screening 使用 unknown",
    )

    @model_validator(mode="after")
    def positive_symptoms_require_met(self) -> "TreeNodeJudgment":
        if self.status != "met" and self.symptom_ids:
            raise ValueError("pending/not_met must have symptom_ids=[]; positive symptoms require met")
        return self


class TreeDecisionRecord(TreeNodeJudgment):
    question: str
    source_refs: list[str]
    stage: Literal["screening", "anchor", "exclusion", "criterion"]


class CandidateDecisionPath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str
    tree_refs: list[str]
    provisional: bool
    decisions: list[TreeDecisionRecord]


class SymptomMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched: int = Field(ge=0)
    total: int = Field(ge=1)

    @model_validator(mode="after")
    def matched_cannot_exceed_total(self) -> "SymptomMatch":
        if self.matched > self.total:
            raise ValueError("matched 不能大于 total")
        return self


class DiagnosticCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disorder: SupportedDisorder
    label: str = Field(min_length=1)
    symptom_match: SymptomMatch
    core_evidence_summary: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    decision_paths: list[CandidateDecisionPath] = Field(default_factory=list)


class CandidateGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    considered_disorders: list[SupportedDisorder] = Field(min_length=DISORDER_COUNT, max_length=DISORDER_COUNT)
    candidates: list[DiagnosticCandidate] = Field(default_factory=list, max_length=DISORDER_COUNT)
    tree_trace: list[TreeDecisionRecord] = Field(default_factory=list)
    missing_dimensions: list[MissingDimension] = Field(default_factory=list)
    candidate_coverage: dict[str, str] = Field(default_factory=dict)


class CriterionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal["met", "not_met", "pending"]
    rationale: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class EliminativeAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disorder: SupportedDisorder
    criteria_detail: list[CriterionAssessment] = Field(min_length=1)
    conclusion: Literal["meets_criteria", "watch", "ruled_out"]
    conclusion_reason: str = Field(min_length=1)
    missing_dimensions: list[MissingDimension] = Field(default_factory=list)


class VerifierDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Literal[
        "evidence_sufficiency",
        "logic_coherence",
        "differential_coverage",
        "no_reasoning_leaps",
    ]
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class RewardModelAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    dimensions: list[VerifierDimension] = Field(min_length=4, max_length=4)
    fail_reasons: list[str] = Field(default_factory=list)
    missing_dimensions: list[MissingDimension] = Field(default_factory=list)

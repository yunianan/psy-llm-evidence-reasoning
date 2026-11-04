import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from layers.layer4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from layers.layer4.schemas.l3_report import DiagnosticReport
from layers.layer4.schemas.l5_context import DiagnosisContext, L5Context, RiskContext
from layers.layer4.schemas.review import ReviewResult
from layers.layer4.schemas.summary import EvidenceRef, StructuredSummary
from shared.schemas.treatment_instructions import TreatmentInstructions

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_role_instruction():
    r = RoleInstruction(identity="CBT治疗师", style="温暖")
    assert r.identity == "CBT治疗师"
    with pytest.raises(ValidationError):
        RoleInstruction()  # 缺少必填字段


def test_turn_instruction_force_substage_enum():
    t = TurnInstruction(goal="g", technique="t", force_substage="L4-L2")
    assert t.force_substage == "L4-L2"
    with pytest.raises(ValidationError):
        TurnInstruction(goal="g", technique="t", force_substage="L4-L5")


def test_anchor_therapy_type():
    a = DiagnosticAnchor(primary="F40.2", therapy_type="CBT")
    assert a.therapy_type == "CBT"


def test_safety_passport_defaults():
    s = SafetyPassport()
    assert s.risk_level == 1
    assert s.sensitive_topics == []


def test_review_result_passed():
    ok = ReviewResult(harm="pass", boundary="pass", quality="pass")
    assert ok.passed is True
    bad = ReviewResult(harm="fail", boundary="pass", quality="pass",
                       fail_reason="x", fix_instruction="y")
    assert bad.passed is False


def test_l5_context_substage_enum():
    instructions = TreatmentInstructions(
        L1=RoleInstruction(identity="CBT治疗师", style="温暖"),
        L2=DiagnosticAnchor(primary="F40.2", therapy_type="CBT"),
        L3=TurnInstruction(goal="探索困扰", technique="反映"),
        L4=BoundaryInstruction(),
    )
    ctx = L5Context(l4_raw_output="你好", l4_substage="L4-L3",
                    user_message="我很难过", history=[], instructions=instructions,
                    therapy_type="CBT",
                    diagnosis_context=DiagnosisContext(primary="F40.2"),
                    risk_context=RiskContext())
    assert ctx.l4_substage == "L4-L3"
    with pytest.raises(ValidationError):
        L5Context(l4_raw_output="你好", l4_substage="L4-L9",
                  user_message="我很难过", history=[], instructions=instructions,
                  therapy_type="CBT",
                  diagnosis_context=DiagnosisContext(),
                  risk_context=RiskContext())


def test_structured_summary_defaults():
    s = StructuredSummary(timestamp="2026-08-12T00:00:00Z",
                          summary="s",
                          structured_labels={},
                          evidence_refs=[EvidenceRef(source="a", finding="b", basis="c")],
                          confidence=0.85)
    assert s.layer_id == "L4"
    assert s.flags == []


def test_diagnostic_report_parses_sample_and_ignores_extra():
    raw = json.loads((EXAMPLES / "sample_diagnostic_report.json").read_text(encoding="utf-8"))
    raw["unused_example_field"] = "ignored"
    report = DiagnosticReport.model_validate(raw)
    assert report.diagnosis.primary.disorder == "F40.2"
    assert report.diagnosis.primary.confidence == 0.63
    assert report.recommended_therapy.primary == "CBT"
    assert report.recommended_therapy.addon == ["CBT-I"]
    assert report.verifier_routing == "pass"
    anchor = report.to_anchor()
    assert anchor.therapy_type == "CBT"
    assert "CBT-I" in anchor.plugins
    assert anchor.contraindications == report.recommended_therapy.contraindications
    assert anchor.primary_label == report.diagnosis.primary.label
    assert "unused_example_field" not in report.model_dump()

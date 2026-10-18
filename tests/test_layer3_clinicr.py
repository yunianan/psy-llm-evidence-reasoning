import pytest

from layers.layer3.api import diagnose
from layers.layer3.llm import ALLOWED_EVIDENCE_REFS_KEY, CLINICRModelUnavailable
from layers.layer3.diagnosis.schemas import SUPPORTED_DISORDERS
from layers.layer4.schemas.l3_report import DiagnosticReport
from tests.l3_test_double import (
    ControlledCLINICRReasoner,
    controlled_clinicr,
)


def _profile() -> dict:
    return {
        "PHQ9": 16,
        "PHQ9_items": {
            "item1": 3,
            "item2": 3,
            "item3": 2,
            "item4": 2,
            "item5": 0,
            "item6": 2,
            "item7": 0,
            "item8": 0,
            "item9": 0,
        },
        "key_symptoms_extracted": [{
            "symptom": "depressed_mood",
            "sources": ["patient_utterance_L008"],
        }],
        "symptom_duration_days": 21,
        "functional_impairment": True,
        "mania_or_hypomania_history": False,
        "substance_or_medical_cause": False,
        "psychotic_symptoms": False,
        "differential_diagnoses_assessed": ["D009", "D020"],
        "evidence_conflicts": [],
        "evidence_nodes": [
            {
                "evidence_id": f"ev-{field}",
                "target_path": f"diagnostic_context.{field}",
            }
            for field in (
                "symptom_duration_days",
                "functional_impairment",
                "mania_or_hypomania_history",
                "substance_or_medical_cause",
                "psychotic_symptoms",
            )
        ],
    }


def test_step1_considers_exactly_twenty_four_disorders_and_keeps_a_relevant_subset():
    engine, reasoner, verifier = controlled_clinicr()

    result = engine.run(_profile())

    assert set(reasoner.generate_calls[0].keys()) >= {"PHQ9", "PHQ9_items"}
    assert set(reasoner.generate_calls[0][ALLOWED_EVIDENCE_REFS_KEY]) >= {
        "PHQ9=16", "PHQ9_item2=3", "patient_utterance_L008"
    }
    assert 1 <= len(result["candidates"]) <= len(SUPPORTED_DISORDERS)
    assert set(SUPPORTED_DISORDERS) == set(
        reasoner.generate_candidates(_profile()).considered_disorders
    )
    assert not hasattr(reasoner, "evaluate_candidate")
    assert len(verifier.calls) == 1


def test_allowed_evidence_refs_are_rule_generated_not_accepted_from_input():
    engine, reasoner, _ = controlled_clinicr()
    profile = _profile() | {ALLOWED_EVIDENCE_REFS_KEY: ["invented_ref"]}

    engine.run(profile)

    observed = reasoner.generate_calls[0][ALLOWED_EVIDENCE_REFS_KEY]
    assert "invented_ref" not in observed
    assert "PHQ9=16" in observed


def test_candidate_assessments_reuse_step1_without_a_second_model_call():
    class NoSecondPass(ControlledCLINICRReasoner):
        def evaluate_candidate(self, *args):
            pytest.fail("ELIMINATIVE must not run after Step 1")

    engine, _, verifier = controlled_clinicr()
    reasoner = NoSecondPass()
    engine.reasoner = reasoner
    result = engine.run(_profile())
    assert len(reasoner.generate_calls) == 1
    assert [item["disorder"] for item in result["checked"]] == ["D013", "D011", "D020", "D009"]
    assert all(item["source"] == "step1_tree_records" for item in result["checked"])
    assert len(verifier.calls) == 1


def test_step1_preserves_low_confidence_differentials_and_binds_source_evidence():
    engine, _, _ = controlled_clinicr()

    result = engine.run(_profile())

    primary = result["diagnosis"]["primary"]
    assert primary["disorder"] == "D013"
    detail = {item["code"]: item for item in primary["dsm5_criteria_detail"]}
    core = detail["criterion.D013.depressive_episode/must_include_one_of"]
    assert core["status"] == "met"
    assert core["evidence_refs"] == ["PHQ9_item2=3"]
    assert primary["dsm5_criteria_met"]


    candidate_confidences = {
        item["disorder"]: item["confidence"] for item in result["candidates"]
    }
    assert candidate_confidences["D020"] == 0.35
    assert candidate_confidences["D009"] == 0.15
    assert {item["disorder"] for item in result["diagnosis"]["under_observation"]} == {
        "D011", "D020", "D009"
    }
    assert result["diagnosis"]["ruled_out"] == []


def test_no_candidate_reaches_verifier_when_step1_is_only_provisional():
    engine, reasoner, verifier = controlled_clinicr()

    result = engine.run({"PHQ9": 20})

    assert len(reasoner.generate_calls) == 1
    assert verifier.calls == []
    assert result["diagnosis"]["primary"] is None
    assert result["verifier_decision"] == "rejected"
    assert result["verifier_routing"] == "return_to_l2"
    assert result["missing_dimensions"]


@pytest.mark.parametrize(
    ("score", "routing", "decision", "has_primary"),
    [
        (0.80, "pass", "confirmed", True),
        (0.79, "conditional", "provisional", True),
        (0.50, "conditional", "provisional", True),
        (0.49, "return_to_l2", "rejected", False),
    ],
)
def test_step3_thresholds_control_routing(score, routing, decision, has_primary):
    engine, _, _ = controlled_clinicr(score=score)

    result = engine.run(_profile())

    assert result["verifier_score"] == score
    assert result["verifier_routing"] == routing
    assert result["verifier_decision"] == decision
    assert (result["diagnosis"]["primary"] is not None) is has_primary
    if routing != "pass":
        assert result["missing_dimensions"]


def test_unknown_model_evidence_is_rejected_before_diagnosis():
    class HallucinatingReasoner(ControlledCLINICRReasoner):
        def generate_candidates(self, profile):
            output = super().generate_candidates(profile)
            output.candidates[0].evidence_refs = ["invented_patient_fact"]
            return output

    engine, _, verifier_model = controlled_clinicr()
    engine.reasoner = HallucinatingReasoner()

    with pytest.raises(CLINICRModelUnavailable, match="不存在的证据"):
        engine.run(_profile())
    assert verifier_model.calls == []


def test_diagnose_reuses_step1_before_verifier_and_therapy():
    engine, _, verifier = controlled_clinicr(score=0.87)

    raw = diagnose(_profile(), clinicr=engine)
    report = DiagnosticReport.model_validate(raw)

    assert report.diagnosis.primary is not None
    assert report.diagnosis.primary.disorder == "D013"
    assert report.candidate_confidence == 0.75
    assert report.verifier_score == 0.87
    assert report.verifier_decision == "confirmed"
    assert report.verifier_routing == "pass"
    assert report.recommended_therapy.primary == "CBT+BA"
    assert report.l3_validation is not None and report.l3_validation.passed
    assert len(verifier.calls) == 1


def test_l3_shared_input_contract_returns_to_l2_before_calling_model():
    class MustNotRun:
        def run(self, profile, safety_context=None):
            raise AssertionError("输入契约不完整时不得调用CLINICR模型")

    report = diagnose(
        {
            "PHQ9": 17,
            "symptom_duration_days": 42,
            "functional_impairment": True,
            "evidence_nodes": [
                {
                    "evidence_id": "duration",
                    "target_path": "diagnostic_context.symptom_duration_days",
                },
                {
                    "evidence_id": "function",
                    "target_path": "diagnostic_context.functional_impairment",
                },
            ],
        },
        clinicr=MustNotRun(),
    )

    assert report["verifier_routing"] == "return_to_l2"
    assert report["instructions"]["L2"]["action"] == "collect_missing_dimensions"
    assert {item["field"] for item in report["missing_dimensions"]} == {
        "mania_or_hypomania_history",
        "substance_or_medical_cause",
        "psychotic_symptoms",
    }
    assert report["diagnosis"]["primary"] is None
    assert report["l3_validation"]["passed"] is False
    assert report["l3_validation"]["route"] == "reject"


def test_low_verifier_score_returns_actionable_feedback_to_l2():
    engine, _, _ = controlled_clinicr(score=0.49)

    report = diagnose(_profile(), clinicr=engine)

    assert report["diagnosis"]["primary"] is None
    assert report["verifier_routing"] == "return_to_l2"
    assert report["instructions"]["L2"]["action"] == "collect_missing_dimensions"
    assert report["verifier_fail_reasons"]
    assert report["missing_dimensions"]

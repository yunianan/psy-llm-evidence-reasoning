"""Required-check coverage and follow-up grading, not a new diagnostic score."""
from copy import deepcopy

import pytest

from layers.layer3.api import DiagnoseResponse, build_blocked_report
from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.diagnosis.information_report import build_information_report
from layers.layer3.diagnosis.verifier import IndependentVerifier
from layers.layer3.output_validator import L3OutputValidator
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from tests.l3_test_double import ControlledRewardModelVerifier
from tests.test_diagnostic_information_report import raw_report
from tests.tree_test_support import controlled_generation


COUNT = "criterion.D013.depressive_episode/min_count"
DURATION = "criterion.D013.depressive_episode/min_duration"


def grade(report, knowledge):
    return build_information_report(report, knowledge=knowledge)["diagnostic_readiness"]


def test_ratios_have_explicit_denominators_and_separate_core_from_all_requirements():
    report, knowledge = raw_report(complete=("D013",), overrides={COUNT: "pending", DURATION: "not_met"})
    before = deepcopy(report)
    item = grade(report, knowledge)["candidates"][0]
    coverage = item["evidence_coverage"]
    assert coverage["required"] == {
        "total": 7, "confirmed_count": 5, "unknown_count": 1, "contradicted_count": 1,
        "confirmed_ratio": 5 / 7, "unknown_ratio": 1 / 7, "contradicted_ratio": 1 / 7,
    }
    assert coverage["core"] == {
        "total": 2, "confirmed_count": 1, "unknown_count": 1, "contradicted_count": 0,
        "confirmed_ratio": .5, "unknown_ratio": .5, "contradicted_ratio": 0,
    }
    assert not coverage["core_requirements_confirmed"]
    assert not coverage["retained_with_unknowns_only"]
    assert item["disposition"] == "needs_review"
    assert report == before


def test_unknown_core_downgrades_to_follow_up_despite_high_anchor_confidence():
    report, knowledge = raw_report(complete=("D013",), overrides={COUNT: "pending"})
    report["candidates"][0]["confidence"] = .99
    readiness = grade(report, knowledge)
    item = readiness["candidates"][0]
    assert item["anchor_confidence"] == .99
    assert item["disposition_label"] == "待追问"
    assert readiness["candidate_groups"] == {
        "evidence_ready": [], "needs_follow_up": ["D013"], "needs_review": [],
    }
    assert item["evidence_coverage"]["retained_with_unknowns_only"]
    assert report["candidate_coverage"]["D013"] == "provisional_candidate"


def test_confirmed_core_does_not_bypass_unknown_duration():
    report, knowledge = raw_report(complete=("D013",), overrides={DURATION: "pending"})
    item = grade(report, knowledge)["candidates"][0]
    assert item["evidence_coverage"]["core"]["confirmed_ratio"] == 1
    assert item["evidence_coverage"]["core_requirements_confirmed"]
    assert item["disposition"] == "needs_follow_up"


@pytest.mark.parametrize("status,state,disposition", [
    ("not_met", "confirmed", "evidence_ready"),
    ("met", "contradicted", "needs_review"),
    ("pending", "unknown", "needs_follow_up"),
])
def test_guard_counts_against_expected_state(status, state, disposition):
    # A live contradictory guard would already prune this candidate. Exercise
    # report grading on a retained/archived candidate without changing traversal.
    report, knowledge = raw_report(complete=("D013",))
    record = next(r for r in report["candidates"][0]["decision_paths"][0]["decisions"]
                  if r["node_id"] == "history.mania")
    record["status"] = status
    item = grade(report, knowledge)["candidates"][0]
    check = next(c for c in item["evidence_coverage"]["checks"] if c["node_id"] == "history.mania")
    assert check["expected_status"] == "not_met"
    assert check["state"] == state
    assert item["disposition"] == disposition


def test_optional_core_conditions_never_enter_denominator():
    report, knowledge = raw_report(("D008",), complete=("D008",), overrides={
        "criterion.D008.depressive_episode/min_count": "pending",
        "criterion.D008.depressive_episode/must_include_one_of": "not_met",
    })
    item = grade(report, knowledge)["candidates"][0]
    assert item["disposition"] == "evidence_ready"
    assert item["evidence_coverage"]["core"]["total"] == 2
    assert item["evidence_coverage"]["core"]["confirmed_ratio"] == 1
    assert not any("depressive_episode" in c["node_id"] for c in item["evidence_coverage"]["checks"])


@pytest.mark.parametrize("raw_status", ["met", "not_met"])
def test_known_status_without_patient_refs_is_unknown_not_proof_or_refutation(raw_status):
    report, knowledge = raw_report(complete=("D013",))
    record = next(r for r in report["candidates"][0]["decision_paths"][0]["decisions"] if r["node_id"] == COUNT)
    record.update(status=raw_status, evidence_refs=[])
    item = grade(report, knowledge)["candidates"][0]
    check = next(c for c in item["evidence_coverage"]["checks"] if c["node_id"] == COUNT)
    assert check["state"] == "unknown"
    assert check["unknown_reason"] == "patient_evidence_refs_missing"
    assert item["disposition"] == "needs_follow_up"
    assert not item["known_blockers"]


def test_absent_record_is_in_denominator_and_not_sourced_from_global_trace():
    report, knowledge = raw_report(complete=("D013",))
    path = report["candidates"][0]["decision_paths"][0]
    path["decisions"] = [r for r in path["decisions"] if r["node_id"] != COUNT]
    item = grade(report, knowledge)["candidates"][0]
    assert item["evidence_coverage"]["required"]["total"] == 7
    assert item["evidence_coverage"]["core"]["unknown_count"] == 1
    assert next(c for c in item["evidence_coverage"]["checks"] if c["node_id"] == COUNT)["unknown_reason"] == "record_missing"


def test_report_fields_survive_api_serialization_and_keep_all_candidates_in_groups():
    report, knowledge = raw_report(("D013", "D011", "D003", "D002", "D001"))
    report["diagnostic_readiness"] = grade(report, knowledge)
    report["verifier_routing"] = "return_to_l2"
    report["verifier_score"] = 0
    report["verifier_decision"] = "rejected"
    report["candidate_confidence"] = 0
    report["summary"] = "fixture"
    report["evidence_refs"] = []
    report["flags"] = []
    report["severity_level"] = 1
    payload = build_blocked_report(report)
    payload["l3_validation"] = L3OutputValidator().validate(payload)
    parsed = DiagnoseResponse.model_validate(payload)
    assert parsed.diagnostic_readiness == report["diagnostic_readiness"]
    assert parsed.diagnostic_readiness["candidate_groups"]["needs_follow_up"] == ["D001", "D002", "D003", "D011", "D013"]
    assert len(parsed.candidates) == 5


def test_annotation_cannot_change_reward_model_input_or_final_routing(monkeypatch):
    generation = controlled_generation(["D013"], complete_labels=["D013"])

    class Reasoner:
        def generate_candidates(self, profile):
            return generation

    def run():
        model = ControlledRewardModelVerifier()
        result = CLINICR(reasoner=Reasoner(), verifier=IndependentVerifier(model=model),
                         knowledge=JsonDiagnosticKnowledgeProvider()).run({"evidence_nodes": [{"evidence_id": "ev-mood"}]})
        return result, model.calls

    annotated, actual_calls = run()
    monkeypatch.setattr("layers.layer3.diagnosis.clinicr.build_information_report", lambda *a, **k: {})
    baseline, previous_calls = run()
    assert actual_calls == previous_calls
    assert len(actual_calls) == 1
    assert {k: v for k, v in annotated.items() if k not in {"diagnostic_readiness", "information_collection_plan"}} == baseline
    assert annotated["diagnostic_readiness"]["candidate_groups"]["evidence_ready"] == ["D013"]

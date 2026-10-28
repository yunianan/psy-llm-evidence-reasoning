"""P2.1 ranking invariants; controlled evidence, never clinical gold or APIs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.diagnosis.information_report import build_information_report
from layers.layer3.diagnosis.verifier import IndependentVerifier
from shared.candidate_ordering import candidate_sort_key
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from tests.l3_test_double import ControlledRewardModelVerifier
from tests.tree_test_support import controlled_generation
from tools.prediction_contract import normalize_for_protocol


@pytest.mark.parametrize("as_object", [False, True])
def test_baseline_order_is_independent_of_input_order_for_equal_scores(as_object):
    candidates = [{"disorder": "Insomnia", "confidence": .8},
                  {"disorder": "D013", "confidence": .8},
                  {"disorder": "D011", "confidence": .9},
                  {"disorder": "D001", "confidence": .8}]
    if as_object:
        candidates = [SimpleNamespace(**c) for c in candidates]
    before = deepcopy(candidates)
    ranked = sorted(candidates, key=candidate_sort_key)
    ids = [c.disorder if as_object else c["disorder"] for c in ranked]
    assert ids == ["D011", "D001", "D013", "Insomnia"]
    assert sorted(reversed(candidates), key=candidate_sort_key) == ranked
    assert candidates == before


def test_confidence_is_not_rounded_or_overridden_by_clinical_signals():
    candidates = [{"disorder": "D001", "confidence": .8, "symptom_match": {"matched": 9, "total": 9},
                   "severity": "severe"},
                  {"disorder": "D013", "confidence": .8000000000000002,
                   "symptom_match": {"matched": 1, "total": 9}, "severity": "unknown"}]
    assert [c["disorder"] for c in sorted(candidates, key=candidate_sort_key)] == ["D013", "D001"]
    candidates[1]["confidence"] = .8
    assert [c["disorder"] for c in sorted(reversed(candidates), key=candidate_sort_key)] == ["D001", "D013"]


def test_traversal_order_is_preserved_while_engine_report_and_scoring_agree():
    generation = controlled_generation(["D001", "D002", "D013"],
                                      confidences={"D001": .8, "D002": .9, "D013": .8})
    before = generation.model_dump(mode="json")
    assert [c.disorder for c in generation.candidates] == ["D001", "D002", "D013"]
    candidates = CLINICR._validate_generation(generation, {"ev-mood"})
    expected = ["D002", "D001", "D013"]
    assert [c.disorder for c in candidates] == expected
    assert generation.model_dump(mode="json") == before
    report = {**before, "candidates": [c.model_dump(mode="json") for c in candidates],
              "checked": [], "final_diagnoses": []}
    original = deepcopy(report)
    normalized = normalize_for_protocol({"case_id": "engineering", "report": report})
    info = build_information_report(report, knowledge=JsonDiagnosticKnowledgeProvider())
    assert [c["disorder"] for c in normalized["candidates"]] == expected
    assert info["information_collection_plan"]["candidate_order"] == expected
    assert report == original


def test_engine_ties_are_explicit_even_when_injected_generation_is_reversed():
    generation = controlled_generation(["D001", "D002"])
    generation.candidates.reverse()
    before = generation.model_dump(mode="json")
    ranked = CLINICR._validate_generation(generation, {"ev-mood"})
    assert [c.disorder for c in ranked] == ["D001", "D002"]
    assert generation.model_dump(mode="json") == before


def test_reward_verifier_receives_existing_contract_without_candidate_confidence():
    generation = controlled_generation(["D001", "D002"], complete_labels=["D001", "D002"])

    class Reasoner:
        def generate_candidates(self, profile):
            return generation.model_copy(deep=True)

    scorer = ControlledRewardModelVerifier()
    result = CLINICR(reasoner=Reasoner(), verifier=IndependentVerifier(model=scorer),
                     knowledge=JsonDiagnosticKnowledgeProvider()).run(
                         {"evidence_nodes": [{"evidence_id": "ev-mood"}]})
    assert len(scorer.calls) == 1
    assert scorer.calls[0]["candidate"]["disorder"] == "D001"
    assert "confidence" not in scorer.calls[0]["candidate"]
    assert all("confidence" not in node for path in scorer.calls[0]["candidate"]["decision_paths"]
               for node in path["decisions"])
    assert result["verifier_routing"] == "pass"
    assert {c["check_id"] for c in result["verifier_checks"]} == {
        "evidence_sufficiency", "logic_coherence", "differential_coverage", "no_reasoning_leaps"}

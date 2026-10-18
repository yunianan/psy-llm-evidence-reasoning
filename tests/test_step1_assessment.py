import pytest

from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.diagnosis.decision_tree import criterion_nodes
from layers.layer3.diagnosis.verifier import IndependentVerifier
from layers.layer3.llm import CLINICRModelUnavailable
from tests.l3_test_double import ControlledRewardModelVerifier
from tests.tree_test_support import controlled_generation


def run_generation(generation):
    class SinglePassReasoner:
        def generate_candidates(self, profile):
            return generation

        def evaluate_candidate(self, *args):
            pytest.fail("Duplicate diagnostic judgment must not be called")

    verifier = ControlledRewardModelVerifier()
    result = CLINICR(
        reasoner=SinglePassReasoner(), verifier=IndependentVerifier(model=verifier),
    ).run({"evidence_nodes": [{"evidence_id": "ev-mood"}]})
    return result, verifier


def test_standard_records_and_exact_sources_reach_report_without_rejudgment():
    generation = controlled_generation(["D013"], complete_labels=["D013"])
    result, verifier = run_generation(generation)
    assert result["diagnosis"]["primary"]["disorder"] == "D013"
    assert len(verifier.calls) == 1
    records = {row.node_id: row for row in generation.tree_trace if row.stage == "criterion"}
    details = {row["code"]: row for row in result["checked"][0]["criteria_detail"]}
    refs = {row["criterion_code"]: row for row in result["evidence_refs"]
            if row["type"] == "DSM-5-criterion"}
    for code, record in records.items():
        assert details[code]["status"] == record.status
        assert details[code]["rationale"] == record.rationale
        assert details[code]["evidence_refs"] == record.evidence_refs
        assert refs[code]["source_doc"] == record.source_refs[0]
    assert result["checked"][0]["source"] == "step1_tree_records"


@pytest.mark.parametrize("status", ["not_met", "pending"])
def test_unmet_or_unknown_required_standard_remains_provisional(status):
    code = "criterion.D013.depressive_episode/min_duration"
    generation = controlled_generation(
        ["D013"], complete_labels=["D013"], overrides={code: status},
    )
    result, verifier = run_generation(generation)
    assert result["candidates"][0]["decision_paths"][0]["provisional"]
    assert result["checked"][0]["conclusion"] == "watch"
    detail = next(row for row in result["checked"][0]["criteria_detail"] if row["code"] == code)
    assert detail["status"] == status
    assert result["diagnosis"]["primary"] is None
    assert result["verifier_routing"] == "return_to_l2"
    assert result["missing_dimensions"]
    assert verifier.calls == []


def test_unknown_guard_cannot_be_promoted_by_complete_symptom_criteria():
    generation = controlled_generation(
        ["D013"], complete_labels=["D013"], overrides={"history.mania": "pending"},
    )
    result, verifier = run_generation(generation)
    assert result["checked"][0]["conclusion"] == "watch"
    assert any(row["field"] == "decision_tree.history.mania" for row in result["missing_dimensions"])
    assert verifier.calls == []


def test_negative_exclusion_fact_satisfies_a_negative_guard_requirement():
    generation = controlled_generation(["D013"], complete_labels=["D013"])
    assert next(row for row in generation.tree_trace if row.node_id == "history.mania").status == "not_met"
    result, _ = run_generation(generation)
    guard = next(row for row in result["checked"][0]["criteria_detail"] if row["code"] == "history.mania")
    assert guard["status"] == "met"
    assert "not_met" in guard["name"]


def test_unknown_optional_group_does_not_block_complete_required_criteria():
    optional = next(code for code, node in criterion_nodes("D008").items() if not node["mandatory"])
    generation = controlled_generation(
        ["D008"], complete_labels=["D008"], overrides={optional: "pending"},
    )
    result, verifier = run_generation(generation)
    assert result["checked"][0]["conclusion"] == "meets_criteria"
    assert result["diagnosis"]["primary"]["disorder"] == "D008"
    assert len(verifier.calls) == 1


def test_missing_record_is_not_inferred_from_a_non_provisional_flag():
    generation = controlled_generation(["D013"], complete_labels=["D013"])
    path = generation.candidates[0].decision_paths[0]
    path.decisions = [row for row in path.decisions if row.node_id != "criterion.D013.depressive_episode/min_count"]
    assert path.provisional is False
    result, verifier = run_generation(generation)
    assert result["checked"][0]["conclusion"] == "watch"
    assert result["diagnosis"]["primary"] is None
    assert verifier.calls == []


def test_bare_candidate_without_step1_records_requires_more_information():
    generation = controlled_generation(["D013"], complete_labels=["D013"])
    generation.candidates[0].decision_paths = []
    result, verifier = run_generation(generation)
    assert result["checked"][0]["conclusion"] == "watch"
    assert result["missing_dimensions"]
    assert verifier.calls == []


def test_step1_pruned_branch_is_reported_as_excluded_without_new_judgment():
    generation = controlled_generation(
        ["D013"], complete_labels=["D013"], overrides={"history.mania": "met"},
    )
    assert not generation.candidates
    result, verifier = run_generation(generation)
    assert result["diagnosis"]["ruled_out"][0]["disorder"] == "D013"
    assert result["checked"][0]["conclusion"] == "ruled_out"
    assert result["diagnosis"]["ruled_out"][0]["evidence_ref"] == "ev-mood"
    assert verifier.calls == []


def test_reused_candidate_path_still_enforces_patient_evidence_whitelist():
    generation = controlled_generation(["D013"], complete_labels=["D013"])
    path = generation.candidates[0].decision_paths[0]
    # Separate the candidate path from the top-level trace to test the summary boundary.
    path.decisions = [row.model_copy(deep=True) for row in path.decisions]
    next(row for row in path.decisions if row.stage == "criterion").evidence_refs = ["invented"]
    with pytest.raises(CLINICRModelUnavailable, match="不存在的证据"):
        run_generation(generation)

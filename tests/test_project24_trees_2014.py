"""Source scope, explicit subtype boundaries, comorbidity and graph execution.

These are engineering fixtures, not estimates of clinical sensitivity or PPV.
"""
import json

import pytest

from paper_dtree.entries import EntryProfile
from paper_dtree.models import Bundle, load_bundle
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / "knowledge/paper_dtree/first2014_project24"
from tools.verify_project24_trees_2014 import adhd_witnesses, run_scripted, verify


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(DATA / "refined_bundle.json")


def names(result):
    catalog = json.loads((DATA / "label_catalog.json").read_text(encoding="utf-8"))
    lookup = {r["name"]: r["id"] for r in catalog}
    return [lookup[n] for n in result["raw_diagnoses"]]


def execute(bundle, family, positives):
    tid = "p24." + family
    return run_scripted(bundle, [tid], {(tid, n): True for n in positives})[0]


MDE = ["mde_core", "mde_associated", "mde_significance"]
MANIA = ["mania_mood_energy", "mania_duration", "mania_symptoms", "mania_severity"]
ADHD_AE = ["a_duration", "development", "onset", "c_settings", "d_impairment"]


@pytest.mark.parametrize("family,positive,expected", [
    ("mdd", MDE, ["D013"]),
    ("mdd", MDE + ["psychosis_history", "psychosis_mood_only"], ["D015"]),
    ("mdd", MDE + ["ever_mania_hypomania"], []),
    ("mdd", MDE + ["medical_cause"], []),
    ("mdd", MDE + ["substance_cause"], []),
    ("mdd", MDE + ["psychosis_history"], []),
    ("mdd", MDE + ["psychosis_history", "independent_mood_comorbidity"], ["D013"]),
    ("bipolar1", MANIA, ["D008"]),
    ("bipolar1", MANIA + ["psychosis_history", "psychosis_mood_only"], ["D010"]),
    ("bipolar1", MANIA + ["psychosis_history"], []),
    ("bipolar1", MANIA + ["psychosis_history", "independent_mood_comorbidity"], ["D008"]),
    ("bipolar2", MANIA + MDE, ["D009"]),
    ("bipolar2", MANIA + MDE + ["ever_mania"], []),
    ("pdd", ["chronic_mood", "associated", "mdd_comorbidity", "personality_comorbidity"], ["D014"]),
    ("pdd", ["chronic_mood", "associated", "psychosis_only"], []),
    ("schizoaffective", ["overlap", "psychosis_without_mood", "majority"], []),
    ("schizoaffective", ["overlap", "psychosis_without_mood", "majority", "bipolar_type_confirmed"], ["D006"]),
    ("schizoaffective", ["overlap", "psychosis_without_mood", "majority", "depressive_type_confirmed"], ["D007"]),
    ("schizophrenia", ["active_phase", "duration", "function", "mood_proportion"], ["D005"]),
    ("schizophrenia", ["active_phase", "duration", "function", "mood_proportion", "autism_distinction"], []),
    ("asd", ["trauma", "duration", "clinical_significance"], []),
    ("asd", ["trauma", "trauma_pattern", "duration", "clinical_significance"], ["D019"]),
    ("insomnia", ["sleep_complaint", "opportunity", "frequency", "distress"], ["Insomnia"]),
    ("insomnia", ["sleep_complaint", "opportunity", "frequency", "distress", "other_condition_explains"], []),
    ("insomnia", ["sleep_complaint", "opportunity", "frequency", "distress", "breathing_sleep_only"], []),
    ("adjustment", ["stressor_reaction", "excess"], []),
    ("adjustment", ["stressor_reaction", "excess", "timing_confirmed"], ["D020"]),
    ("adjustment", ["stressor_reaction", "excess", "timing_confirmed", "other_explains_same_symptoms"], []),
])
def test_source_interpretation_boundaries(bundle, family, positive, expected):
    assert names(execute(bundle, family, positive)) == expected


def test_all_nodes_both_exits_and_all_24_labels_are_executable(bundle):
    report = verify(bundle)
    assert report["branch_probes_passed"] == 450
    assert report["labels_with_reachable_emission"] == 24
    assert report["shared_frontier_verified"] and report["all_screens_before_nodes"]
    assert report["network_calls"] == report["model_calls"] == 0


def test_all_adhd_presentation_paths_require_ae_before_emission(bundle):
    tree = next(t for t in bundle.trees if t.id == "p24.adhd")
    required = {"a_duration": "yes", "development": "yes",
                "a_opposition_only": "no", "onset": "yes", "c_settings": "yes", "d_impairment": "yes",
                "e_psychosis_only": "no", "e_better_explanation": "no"}
    emissions = set()
    for (node_id, answer_bool), path in adhd_witnesses(tree).items():
        node = tree.nodes[node_id]
        current = {key: "yes" if value else "no" for key, value in path}
        branch = node.yes if answer_bool else node.no
        for diagnosis in branch.diagnoses:
            emissions.add(diagnosis)
            assert all(current.get(key) == value for key, value in required.items())
            assert current.get("a1_gate") == "yes" or current.get("a2_gate") == "yes"
            assert all(f"a{domain}_{i}" in current for domain in (1, 2) for i in range(1, 10))
    assert len(emissions) == 3


@pytest.mark.parametrize("adult,a1,a2,expected", [
    (False, 5, 5, []), (False, 6, 6, ["D001"]), (False, 6, 5, ["D002"]), (False, 5, 6, ["D003"]),
    (True, 4, 4, []), (True, 5, 5, ["D001"]), (True, 5, 4, ["D002"]), (True, 4, 5, ["D003"]),
    (False, 1, 0, []), (True, 9, 9, ["D001"]),
])
def test_adhd_two_domains_counted_independently_with_age_boundary(bundle, adult, a1, a2, expected):
    positive = ADHD_AE + (["age_17_or_over"] if adult else [])
    positive += [f"a1_{i}" for i in range(1, a1 + 1)]
    positive += [f"a2_{i}" for i in range(1, a2 + 1)]
    result = execute(bundle, "adhd", positive)
    assert names(result) == expected
    trace = {n["node_id"]: n for n in result["tree_trace"]}
    assert all(f"a{domain}_{i}" in trace for domain in (1, 2) for i in range(1, 10))
    assert trace["a1_gate"]["calculation"]["symptom_count"] == a1
    assert trace["a1_gate"]["calculation"]["threshold"] == (5 if adult else 6)
    assert len(trace["a1_gate"]["calculation"]["inputs"]) == 9


@pytest.mark.parametrize("invalid", ["wrong_domain", "missing_age"])
def test_adhd_calculation_rejects_invalid_inputs(invalid):
    raw = json.loads((DATA / "refined_bundle.json").read_text(encoding="utf-8"))
    tree = next(t for t in raw["trees"] if t["id"] == "p24.adhd")
    calculation = tree["nodes"]["a1_gate"]["calculation"]
    if invalid == "wrong_domain":
        calculation["symptom_nodes"][0] = "a2_1"
    else:
        calculation["age_node"] = "missing_age"
    with pytest.raises(ValueError):
        Bundle.parse(raw)


@pytest.mark.parametrize("failed_gate", ["a_duration", "development", "onset", "c_settings", "d_impairment"])
def test_adhd_presentation_never_emitted_when_a_required_gate_is_no(bundle, failed_gate):
    positive = [n for n in ADHD_AE if n != failed_gate] + [f"a1_{i}" for i in range(1, 7)]
    assert names(execute(bundle, "adhd", positive)) == []


@pytest.mark.parametrize("exclusion", ["a_opposition_only", "e_psychosis_only", "e_better_explanation"])
def test_adhd_presentation_never_emitted_when_exclusion_is_yes(bundle, exclusion):
    positive = ADHD_AE + [exclusion] + [f"a1_{i}" for i in range(1, 7)]
    assert names(execute(bundle, "adhd", positive)) == []


def test_unknown_adhd_item_pauses_without_counting_it_as_no_and_resumes(bundle):
    from paper_dtree.engine import PAIRWISE_POLICY, RetryPolicy
    from paper_dtree.followup import FollowUpRunner
    from paper_dtree.models import Response
    def decided(quote, message_id="m1"):
        return json.dumps({"answer": "yes", "follow_up": "", "evidence": [{"message_id": message_id, "quote": quote}]})
    unknown = json.dumps({"answer": "unknown", "follow_up": "近半年是否经常粗心或忽略细节？", "evidence": []})
    class Responses:
        def __init__(self):
            self.answers = iter(["yes"] + ["no"] * 18 + [decided("持续半年"), decided("20岁"), unknown,
                                                           decided("近半年我经常粗心大意。", "m2"), unknown])
            self.calls = 0
        def complete(self, request):
            self.calls += 1
            return Response(next(self.answers), model="fixture")
    api = Responses()
    runner = FollowUpRunner(bundle, api, model="fixture", pairwise_policy=PAIRWISE_POLICY,
                            retry=RetryPolicy(max_attempts=1, initial_seconds=0))
    messages = [{"role": "patient", "message_id": "m1", "text": "我这些困难持续半年，我今年20岁。"}]
    runner.set_patient_evidence(messages, [])
    first = runner.run(messages[0]["text"], case_id="c")
    assert first["status"] == "awaiting_patient" and first["follow_up"]["node_id"] == "a1_1"
    assert not any(n["node_id"] in {"a1_1", "a1_gate"} for n in first["tree_trace"])
    messages.append({"role": "patient", "message_id": "m2", "text": "近半年我经常粗心大意。"})
    runner.set_patient_evidence(messages, [])
    second = runner.run("\n".join(m["text"] for m in messages), case_id="c")
    assert second["status"] == "awaiting_patient" and second["follow_up"]["node_id"] == "a1_2"
    assert next(n for n in second["tree_trace"] if n["node_id"] == "a1_1")["answer"] == "yes"
    assert not any(n["node_id"] == "a1_gate" for n in second["tree_trace"])
    assert api.calls == 24  # 19 screens + 2 determinate + unknown + resolved + next unknown


def test_mdd_pdd_and_independent_insomnia_can_all_survive_and_get_all_pairs(bundle):
    selected = ["p24.mdd", "p24.pdd", "p24.insomnia"]
    positives = {"p24.mdd": MDE, "p24.pdd": ["chronic_mood", "associated", "mdd_comorbidity"],
                 "p24.insomnia": ["sleep_complaint", "opportunity", "frequency", "distress"]}
    script = {(tid, node): True for tid, nodes in positives.items() for node in nodes}
    result, events = run_scripted(bundle, selected, script)
    assert set(names(result)) == {"D013", "D014", "Insomnia"}
    assert len(result["pairwise_comparisons"]) == 3
    assert len({(r["tree_id"], r["node_id"]) for r in result["tree_trace"]}) == len(result["tree_trace"])
    stages = [r["stage"] for r in events if r["event"] == "request_attempt"]
    assert stages[:19] == ["screening"] * 19
    assert all(s == "pairwise" for s in stages[stages.index("pairwise"):])
    # It starts the roots of every selected tree before a second question in one.
    assert [r["tree_id"] for r in result["tree_trace"][:3]] == selected


def test_insufficient_positive_evidence_is_not_output_or_proof_of_absence(bundle):
    result = execute(bundle, "mdd", [])
    assert names(result) == []
    assert any("不等于已证明不患病" in r["text"] for r in result["non_diagnostic_outcomes"])


@pytest.mark.skip(reason="Full source excerpts are not included in the public export")
def test_reproducibility_scope_sources_and_full_table_coverage():
    raw, profile, catalog, report = make_artifacts()
    assert json.loads((DATA / "refined_bundle.json").read_text(encoding="utf-8")) == raw
    assert json.loads((DATA / "entry_profile.json").read_text(encoding="utf-8")) == profile
    assert json.loads((DATA / "label_catalog.json").read_text(encoding="utf-8")) == catalog
    assert json.loads((DATA / "source_audit.json").read_text(encoding="utf-8")) == report
    assert len(raw["trees"]) == 19 and len(raw["diagnoses"]) == 24
    assert len(report["nodes"]) == len(raw["refinements"]) == 225
    assert raw["stage"] == "refined" and raw["coverage"] == "partial"
    assert not report["independent_clinical_review"]
    assert not report["author_refined_supplement_recovered"]
    assert all(n["source_anchor"] and (n["source"]["pdf_pages"] or n["source"]["document"] == "DSM5_2013_user_excerpt") for n in report["nodes"])
    adhd = {n["node_id"]: n for n in report["nodes"] if n["tree_id"] == "p24.adhd"}
    assert adhd["a_opposition_only"]["source"]["print_pages"] == [59]
    assert adhd["onset"]["source"]["print_pages"] == [60]
    assert adhd["onset"]["source"]["pdf_pages"] == []
    assert "solely a manifestation" in adhd["a_opposition_only"]["question"]
    assert report["primary_differential_rows"] == 171
    assert report["rows_with_executable_links"] + report["context_only_rows"] == 171
    assert {r["table_id"] for r in report["primary_table_row_coverage"]}.__len__() == 18
    assert len(EntryProfile.parse(profile).entries) == 24
    assert all(t.startswith("p24.") for e in profile["entries"] for t in e["tree_ids"])
    gaps = json.loads((DATA / "source_gaps.json").read_text(encoding="utf-8"))
    assert {"adhd_additional_review", "schizoaffective_subtype", "adjustment_timing"} <= {g["id"] for g in gaps}


@pytest.mark.skip(reason="Full source excerpts are not included in the public export")
def test_native_table_extraction_does_not_mix_columns_or_treat_footnotes_as_rows():
    source = json.loads((DATA / "source_evidence.json").read_text(encoding="utf-8"))
    assert len(source["primary_table_rows"]) == 189
    assert sum(r["kind"] == "definition" for r in source["primary_table_rows"]) == 18
    for r in source["primary_table_rows"]:
        assert not r["left"].startswith("a ")
        if r["table_id"] in {"3.3.1", "3.3.2"}:
            assert "Is char" not in r["left"] and "Requir" not in r["left"] and "In con" not in r["left"]
    assert len(source["primary_table_columns"]) == 43


def test_cli_default_bundle_and_profile_are_the_current_matching_forest(capsys):
    from paper_dtree.__main__ import DEFAULT_TREE_BUNDLE, main
    from paper_dtree.entries import DEFAULT_ENTRY_PROFILE

    assert DEFAULT_TREE_BUNDLE == DATA / "refined_bundle.json"
    assert DEFAULT_ENTRY_PROFILE == DATA / "entry_profile.json"
    assert main(["inspect"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["bundle_id"] == "First2014-project24-local-refined-adhd-presentations-v4"
    assert report["stage"] == "refined" and report["trees"] == 19 and report["nodes"] == 225
    profile = EntryProfile.load(DEFAULT_ENTRY_PROFILE)
    actual = load_bundle(DEFAULT_TREE_BUNDLE)
    assert {t for e in profile.entries for t in e.tree_ids} == {t.id for t in actual.trees}


def test_cli_default_run_advances_current_tree_without_explicit_bundle(tmp_path, monkeypatch):
    from paper_dtree import transport
    from paper_dtree.__main__ import main
    from tests.test_paper_pipeline import catalog_data
    from tools.verify_project24_trees_2014 import ControlledTransport

    actual = load_bundle(DATA / "refined_bundle.json")
    profile = EntryProfile.load(DATA / "entry_profile.json")
    api = ControlledTransport(actual, ["entry.D013"],
                              {("p24.mdd", node): True for node in MDE}, entry_profile=profile)
    monkeypatch.setattr(transport, "OpenAICompatibleTransport", lambda **kwargs: api)
    (tmp_path / "catalog.json").write_text(json.dumps(catalog_data()), encoding="utf-8")
    (tmp_path / "case.txt").write_text("Scripted engineering fixture", encoding="utf-8")
    output = tmp_path / "run"
    code = main(["run", "--catalog", str(tmp_path / "catalog.json"),
                 "--case-file", str(tmp_path / "case.txt"), "--case-id", "fixture", "--method", "dt",
                 "--model", "controlled-only", "--pairwise-policy", "all_pairs_union_eliminations_v1",
                 "--reconciliation-db", str(tmp_path / "reviews.sqlite"), "--output-dir", str(output)])
    assert code == 2  # Fixture vocabulary needs reconciliation; no fabricated medical scoring.
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    model = saved["stages"][1]["payload"]
    assert model["status"] == "success" and not model["pending_entry_trees"]
    assert model["raw_diagnoses"] == ["Major Depressive Disorder"]
    assert {r["tree_id"] for r in model["tree_trace"]} == {"p24.mdd"}
    assert json.loads((output / "bundle_snapshot.json").read_text(encoding="utf-8"))["id"] == actual.id

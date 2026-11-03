"""Selected project scope: 9 groups, 24 symptom screens, no implicit diagnoses."""
import copy
import json

import pytest

from paper_dtree.entries import DEFAULT_ENTRY_PROFILE, EntryProfile
from paper_dtree.engine import DecisionTreeRunner, PAIRWISE_POLICY, RetryPolicy
from paper_dtree.models import load_bundle
from paper_dtree.pipeline import SevenStagePipeline
from paper_dtree.postprocess import Catalog, ReconciliationStore
from tests.test_paper_dtree import Scripted, bundle_data, node, tree
from tests.test_paper_pipeline import catalog_data
from tools.build_project_entry_profile import build_profile, pool, read


def make_profile(*bindings):
    return EntryProfile.parse({
        "id": "entry-fixture", "standard": "DSM-5", "review_status": "source_verified",
        "groups": [{"id": "G", "label": "fixture group"}],
        "entries": [{"id": f"E{i}", "group_id": "G", "disorder_id": f"label-{i}",
                     "name": f"label name {i}", "label": f"label {i}", "question": f"entry-screen-{i}",
                     "source_refs": ["fixture"], "tree_ids": list(targets),
                     "symptoms": [{"id": f"S{i}", "description": "fixture symptom", "source_ref": "fixture"}]}
                    for i, targets in enumerate(bindings)],
    })


def make_runner(profile, transport, bundle=None):
    from paper_dtree.models import Bundle

    if bundle is None:
        bundle = Bundle.parse(bundle_data(tree("T1", {"a": node("a", yes={"diagnoses": ["Dx-A"]})}),
                                          tree("T2", {"b": node("b", yes={"diagnoses": ["Dx-B"]})})))
    return DecisionTreeRunner(bundle, transport, model="controlled-only", pairwise_policy=PAIRWISE_POLICY,
                              retry=RetryPolicy(max_attempts=1, initial_seconds=0), entry_profile=profile)


def test_selected_profile_copies_exact_nine_groups_and_twenty_four_labels():
    profile = EntryProfile.load(DEFAULT_ENTRY_PROFILE)
    assert len(profile.groups) == 9 and len(profile.entries) == 24
    assert {entry.disorder_id for entry in profile.entries} == {f"D{i:03d}" for i in range(1, 24)} | {"Insomnia"}
    counts = {g["id"]: sum(e.group_id == g["id"] for e in profile.entries) for g in profile.groups}
    assert counts == {"attention": 3, "psychotic": 4, "bipolar": 3, "anxiety": 2, "sleep": 1,
                      "depressive": 3, "obsessive": 2, "stress": 3, "eating": 3}
    assert {s["id"] for e in profile.entries for s in e.symptoms} == {f"S{i:03d}" for i in range(1, 88)}


@pytest.mark.skip(reason="Source-based entry reconstruction requires the local research archive")
def test_exported_profile_is_reproducible_and_preserves_source_symptom_wording():
    data = read(DEFAULT_ENTRY_PROFILE)
    assert data == build_profile()
    root = DEFAULT_ENTRY_PROFILE.parents[3]
    mental = root.parent / "external/MentalBench/resources/knowledge_graph/EN"
    supplement = root.parent / "external/MentalBench/resources/dsm5_supplement/EN"
    criteria = read(mental / "diagnostic_criteria.json")
    criteria.update(read(supplement / "diagnostic_criteria.json"))
    symptoms = {}
    for source_root in (mental, supplement):
        for file in (source_root / "symptom").glob("*.json"):
            symptoms.update(read(file))
    for entry in data["entries"]:
        assert {s["id"] for s in entry["symptoms"]} == pool(criteria[entry["disorder_id"]])
        for symptom in entry["symptoms"]:
            assert symptom["description"] == symptoms[symptom["id"]]["description"]
            assert symptom["description"] in entry["question"]
        assert '"min_count":' not in entry["question"] and '"confidence":' not in entry["question"]


def test_all_entry_screens_finish_before_any_tree_node_and_positive_labels_are_not_emitted():
    profile = make_profile(("T1",), ("T2",))
    api = Scripted("yes", "yes", "yes", "yes", '["Dx-A","Dx-B"]')
    model = make_runner(profile, api)
    result = model.run("case", case_id="c")
    contents = [r.messages[1]["content"] for r in api.requests]
    assert [t.rsplit("The question is as follows: ", 1)[-1] for t in contents[:4]] == [
        "entry-screen-0.", "entry-screen-1.", "a.", "b."]
    assert result["selected_entry_ids"] == ["E0", "E1"] and result["selected_groups"] == ["G"]
    assert result["candidates_before_pairwise"] == ["Dx-A", "Dx-B"]
    assert "label-0" not in result["raw_diagnoses"] and "label-1" not in result["raw_diagnoses"]
    assert len(result["screening"]) == 2 and len(result["pairwise_comparisons"]) == 1


def test_shared_target_tree_is_not_rerun_or_counted_as_multiple_label_votes():
    model = make_runner(make_profile(("T1",), ("T1",)), Scripted("yes", "yes", "yes"))
    result = model.run("case", case_id="c")
    assert len(result["selected_entry_ids"]) == 2
    assert len(result["tree_trace"]) == 1 and result["raw_diagnoses"] == ["Dx-A"]
    assert len(result["candidate_sources"]["Dx-A"]) == 1


def test_all_twenty_four_negative_screens_produce_no_diagnosis_or_extra_group_queries():
    profile = EntryProfile.load(DEFAULT_ENTRY_PROFILE)
    api = Scripted(*(["no"] * 24))
    model = make_runner(profile, api)
    result = model.run("case", case_id="c")
    assert len(api.requests) == len(result["screening"]) == 24
    assert result["status"] == "success" and result["raw_diagnoses"] == []
    assert result["tree_trace"] == result["selected_groups"] == result["pending_entry_trees"] == []


def test_positive_entry_with_missing_source_trees_is_pending_not_a_diagnosis_or_miss():
    profile = EntryProfile.load(DEFAULT_ENTRY_PROFILE)
    api = Scripted("yes", *(["no"] * 23))
    model = make_runner(profile, api)
    result = SevenStagePipeline(Catalog.parse(catalog_data()), ReconciliationStore(), runner=model).run(
        "case", case_id="c", reference=["Dx-A"], reference_source="engineering fixture")
    assert result["status"] == "awaiting_tree_content" and "scoring" not in result
    stage2 = result["stages"][1]["payload"]
    assert len(stage2["screening"]) == 24 and stage2["selected_entry_ids"] == ["entry.D001"]
    assert {r["target_tree_id"] for r in stage2["pending_entry_trees"]} == {"p24.adhd"}
    assert "raw_diagnoses" not in stage2 and stage2["candidates_before_pairwise"] == []
    assert all(s["status"] == "not_started" for s in result["stages"][2:])


def test_an_available_selected_tree_still_advances_but_missing_targets_prevent_final_comparison():
    model = make_runner(make_profile(("T1", "missing")), Scripted("yes", "yes"))
    result = model.run("case", case_id="c")
    assert result["status"] == "awaiting_tree_content"
    assert result["candidates_before_pairwise"] == ["Dx-A"]
    assert result["pending_entry_trees"][0]["target_tree_id"] == "missing"
    assert "raw_diagnoses" not in result


def test_base_arm_keeps_direct_prompting_without_entry_or_group_screening():
    api = Scripted('{"diagnoses":["Dx-A"]}')
    model = make_runner(EntryProfile.load(DEFAULT_ENTRY_PROFILE), api)
    result = model.run("case", case_id="c", method="base")
    assert len(api.requests) == 1 and result["screening"] == []
    assert result["raw_diagnoses"] == ["Dx-A"]


def test_profile_runtime_does_not_reconnect_old_knowledge_or_tree_executor(monkeypatch):
    from shared.knowledge_sources import MentalKGSource, InsomniaSupplementSource
    from layers.layer3.llm import client

    def forbidden(*args, **kwargs):
        pytest.fail("Entry runtime must use exported data, not the old knowledge/executor")

    monkeypatch.setattr(MentalKGSource, "get_candidate", forbidden)
    monkeypatch.setattr(InsomniaSupplementSource, "get_candidate", forbidden)
    monkeypatch.setattr(client, "LegacyDSM5TreeReasoner", forbidden)
    profile = EntryProfile.load(DEFAULT_ENTRY_PROFILE)
    result = make_runner(profile, Scripted(*(["no"] * 24))).run("case", case_id="c")
    assert result["status"] == "success"


@pytest.mark.parametrize("mutation", ["duplicate_entry", "missing_group", "empty_source", "empty_symptoms", "empty_routes"])
def test_invalid_entry_profiles_are_rejected(mutation):
    data = read(DEFAULT_ENTRY_PROFILE)
    if mutation == "duplicate_entry": data["entries"].append(copy.deepcopy(data["entries"][0]))
    if mutation == "missing_group": data["entries"][0]["group_id"] = "missing"
    if mutation == "empty_source": data["entries"][0]["source_refs"] = []
    if mutation == "empty_symptoms": data["entries"][0]["symptoms"] = []
    if mutation == "empty_routes": data["entries"][0]["tree_ids"] = []
    with pytest.raises(ValueError): EntryProfile.parse(data)


def test_profile_and_bundle_standards_cannot_silently_differ():
    data = read(DEFAULT_ENTRY_PROFILE)
    data["standard"] = "DSM-5-TR"
    with pytest.raises(ValueError, match="standards must match"):
        make_runner(EntryProfile.parse(data), Scripted())


def test_inspect_entries_reports_the_selected_default_without_reading_medical_tree_nodes(capsys):
    from paper_dtree.__main__ import main

    assert main(["inspect-entries"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["entry_count"] == 24 and len(result["groups"]) == 9


def test_cli_default_uses_selected_entry_profile_and_snapshots_it(tmp_path, monkeypatch):
    from paper_dtree.__main__ import main
    from paper_dtree import transport
    from paper_dtree.models import Bundle

    api = Scripted(*(["no"] * 24))
    monkeypatch.setattr(transport, "OpenAICompatibleTransport", lambda **kwargs: api)
    bundle = bundle_data(tree("T", {"q": node("q")}))
    for filename, payload in (("bundle.json", bundle), ("catalog.json", catalog_data())):
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "case.txt").write_text("controlled case", encoding="utf-8")
    output = tmp_path / "run"
    code = main(["run", "--bundle", str(tmp_path / "bundle.json"), "--catalog", str(tmp_path / "catalog.json"),
                 "--case-file", str(tmp_path / "case.txt"), "--case-id", "c", "--method", "dt",
                 "--model", "controlled-only", "--pairwise-policy", PAIRWISE_POLICY,
                 "--reconciliation-db", str(tmp_path / "reviews.sqlite"), "--output-dir", str(output)])
    assert code == 2  # Awaiting reference, not failed and not fabricated scoring.
    assert len(api.requests) == 24
    snapshot = json.loads((output / "entry_profile_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["id"] == EntryProfile.load(DEFAULT_ENTRY_PROFILE).id
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert saved["stages"][0]["payload"]["entry_profile_id"] == snapshot["id"]

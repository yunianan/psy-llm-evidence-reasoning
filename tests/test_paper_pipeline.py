"""Seven-stage experiment behavior with fictional labels and controlled responses."""
import copy
import json

import pytest

from paper_dtree.pipeline import SevenStagePipeline, STAGES
from paper_dtree.postprocess import Catalog, ReconciliationStore, reconcile
from paper_dtree.scoring import paired_macro, score_case
from tests.test_paper_dtree import Scripted, bundle_data, node, runner, tree


def catalog_data():
    return {"id": "fictional-v1", "standard": "DSM-5", "review_status": "source_verified",
            "source_ref": "fictional-test-only", "diagnoses": [
                {"name": name, "category": category, "kind": kind, "source_ref": "fixture"}
                for name, category, kind in [("Dx-A", "Cat-1", "dsm"), ("Dx-B", "Cat-1", "dsm"),
                                             ("Dx-C", "Cat-2", "dsm"), ("Dx-Other", "Cat-1", "dsm"),
                                             ("Focus-Z", "Other Conditions", "z_code")]],
            "simplifications": [{"input": "Dx-B, mild", "output": "Dx-B", "rule": 1, "source_ref": "fixture"}]}


def review(name="unmatched-output", target="Dx-A", rule=3):
    return {"input": name, "target": target, "appendix_rule": rule,
            "reviewer": "human-fixture", "reviewed_at": "2026-11-01", "reason": "explicit engineering review"}


def make_pipeline(*answers, data=None, record=None):
    api = Scripted(*answers)
    data = data or bundle_data(tree("T", {"q": node("q", yes={"diagnoses": ["Dx-A"]})}))
    model = runner(data, api)
    pipeline = SevenStagePipeline(Catalog.parse(catalog_data()), ReconciliationStore(),
                                 runner=model, record_stage=record)
    return pipeline, api


def test_complete_seven_stages_and_gold_labels_never_enter_the_model():
    pipeline, api = make_pipeline('{"diagnoses":["Dx-A","Dx-B, mild","Dx-C"]}')
    result = pipeline.run("raw vignette", case_id="case", method="base",
                          reference=["Dx-A"], reference_source="author-reference-fixture")
    assert [(s["number"], s["name"]) for s in result["stages"]] == list(STAGES)
    assert all(s["status"] == "completed" for s in result["stages"])
    assert result["status"] == "completed"
    assert len(api.requests) == 1
    prompt = json.dumps(api.requests[0].messages)
    assert "author-reference-fixture" not in prompt and '"reference"' not in prompt
    assert "reference" not in result["stages"][0]["payload"]
    assert result["stages"][2]["payload"]["responses"][0]["response"]["text"] == '{"diagnoses":["Dx-A","Dx-B, mild","Dx-C"]}'
    assert result["scoring"]["diagnosis"]["TP"] == 1
    assert result["scoring"]["diagnosis"]["FP"] == 2
    assert result["scoring"]["diagnosis"]["FN"] == 0
    assert result["scoring"]["category"]["TP"] == 1 and result["scoring"]["category"]["FP"] == 1


def test_dt_subworkflow_is_entirely_inside_model_stage_before_simplification():
    data = bundle_data(tree("T", {"q": node("q", yes={"diagnoses": ["Dx-A", "Dx-B"]})}))
    pipeline, api = make_pipeline("yes", "yes", '["Dx-A"]', data=data)
    result = pipeline.run("case", case_id="case", reference=["Dx-A"], reference_source="author-fixture")
    model = result["stages"][1]["payload"]
    assert model["candidates_before_pairwise"] == ["Dx-A", "Dx-B"]
    assert model["raw_diagnoses"] == ["Dx-A"]
    assert result["stages"][2]["payload"]["diagnoses"] == ["Dx-A"]
    assert len(result["stages"][2]["payload"]["responses"]) == 3
    assert result["scoring"]["diagnosis"]["FP"] == 0 and len(api.requests) == 3


def test_six_simplification_rules_in_order_with_merging_and_traceable_sources():
    data = catalog_data()
    data["simplifications"] += [
        {"rule": 1, "input": "variant-with-modifier", "output": "family-2", "source_ref": "fixture/r1"},
        {"rule": 2, "input": "family-2", "output": "Dx-A", "source_ref": "fixture/r2"},
        *[{"rule": n, "input": f"family-{n}", "output": "Dx-A", "source_ref": f"fixture/r{n}"}
          for n in range(3, 7)]]
    simplified = Catalog.parse(data).simplify(["variant-with-modifier", "family-3", "family-4", "family-5", "family-6", "Dx-A"])
    assert len(simplified["items"]) == 1
    assert simplified["items"][0]["name"] == "Dx-A"
    assert {s["rule"] for s in simplified["items"][0]["steps"]} == set(range(1, 7))
    assert [s["rule"] for s in simplified["rule_trace"]] == list(range(1, 7))
    assert len(simplified["items"][0]["raw_names"]) == 6


def test_unknown_strings_are_preserved_not_guessed_or_deleted_by_simplification():
    result = Catalog.parse(catalog_data()).simplify(["unseen variant, potentially meaningful", "unseen variant, potentially meaningful"])
    assert [r["name"] for r in result["items"]] == ["unseen variant, potentially meaningful"]
    assert not result["items"][0]["steps"]


def test_fuzzy_match_uses_actual_wratio_and_exact_match_is_automatic():
    from rapidfuzz.fuzz import WRatio
    catalog = Catalog.parse(catalog_data())
    matched = catalog.fuzzy_match(catalog.simplify(["Dx-A", "unmatched-output"]))
    assert matched["algorithm"] == "rapidfuzz.fuzz.WRatio"
    assert matched["items"][0]["best_score"] == 100 and matched["items"][0]["automatic_match"] == "Dx-A"
    assert matched["items"][1]["best_score"] == max(WRatio("unmatched-output", name) for name in catalog.entries)
    assert matched["items"][1]["status"] == "requires_reconciliation"


def test_exactly_99_is_reviewed_and_multiple_candidates_over_99_are_ambiguous():
    data = catalog_data()
    data["diagnoses"] = [{"name": "a" * 100, "category": "Cat", "kind": "dsm", "source_ref": "fixture"}]
    catalog = Catalog.parse(data)
    row = catalog.fuzzy_match(catalog.simplify(["a" * 99 + "b"]))["items"][0]
    assert row["best_score"] == 99 and row["automatic_match"] is None
    data["diagnoses"] = [
        {"name": name, "category": "Cat", "kind": "dsm", "source_ref": "fixture"}
        for name in ["a" * 201, "a" * 200 + "b"]]
    catalog = Catalog.parse(data)
    row = catalog.fuzzy_match(catalog.simplify(["a" * 201]))["items"][0]
    assert len(row["matches_above_99"]) == 2 and row["automatic_match"] is None


def test_unresolved_matching_stops_scoring_and_has_a_review_queue():
    pipeline, _ = make_pipeline('{"diagnoses":["unmatched-output"]}')
    result = pipeline.run("case", case_id="case", method="base", reference=["Dx-A"], reference_source="author")
    assert result["status"] == "awaiting_reconciliation" and "scoring" not in result
    assert result["stages"][5]["payload"]["predictions"]["pending"][0]["input"] == "unmatched-output"
    assert result["stages"][6]["status"] == "not_started"


def test_human_review_persists_and_resume_never_repeats_model_requests(tmp_path):
    pipeline, api = make_pipeline('{"diagnoses":["unmatched-output"]}')
    path = tmp_path / "reviews.sqlite"
    pipeline.store = ReconciliationStore(path)
    saved = pipeline.run("case", case_id="case", method="base", reference=["Dx-A"], reference_source="author")
    pipeline.store.add(pipeline.catalog, review())
    pipeline.store.close()
    store = ReconciliationStore(path)
    resumed = SevenStagePipeline(pipeline.catalog, store).resume(saved)
    assert len(api.requests) == 1 and resumed["status"] == "completed"
    assert resumed["scoring"]["diagnosis"]["TP"] == 1
    assert resumed["stages"][5]["payload"]["predictions"]["accepted"][0]["resolution"] == "reused_review"
    store.close()


@pytest.mark.parametrize("rule", [1, 7])
def test_non_dsm_is_retained_and_scored_as_false_positive(rule):
    pipeline, _ = make_pipeline('{"diagnoses":["Dx-A","unmatched-output"]}')
    pipeline.store.add(pipeline.catalog, review(target="unmatched-output", rule=rule))
    result = pipeline.run("case", case_id="case", method="base", reference=["Dx-A"], reference_source="author")
    assert result["scoring"]["diagnosis"]["FP_list"] == ["unmatched-output"]
    assert result["scoring"]["category"]["FP_list"] == ["Non-DSM"]


@pytest.mark.parametrize("rule,target", [(2, "Focus-Z"), (3, "Dx-A"), (4, "Dx-A"), (5, "Dx-Other"), (6, "Dx-A")])
def test_all_english_appendix_reconciliation_rules_are_reusable(rule, target):
    catalog, store = Catalog.parse(catalog_data()), ReconciliationStore()
    decision = store.add(catalog, review(target=target, rule=rule))
    assert store.add(catalog, review(target=target, rule=rule)) == decision
    matched = catalog.fuzzy_match(catalog.simplify(["unmatched-output"]))
    result = reconcile(catalog, matched, store)
    assert result["accepted"][0]["name"] == target and result["status"] == "completed"
    assert result["accepted"][0]["review"]["appendix_rule"] == rule


def test_reconciliation_is_scoped_to_standard_and_catalog_and_cannot_silently_change():
    catalog, store = Catalog.parse(catalog_data()), ReconciliationStore()
    store.add(catalog, review())
    with pytest.raises(ValueError, match="Conflicting"):
        store.add(catalog, review(target="Dx-B"))
    other = catalog_data()
    other["id"] = "fictional-v2"
    assert store.get(Catalog.parse(other), "unmatched-output") is None
    with pytest.raises(ValueError, match="Z-code"):
        store.add(catalog, review(name="another", target="Dx-A", rule=2))
    with pytest.raises(ValueError, match="retained"):
        store.add(catalog, review(name="another", target="Dx-A", rule=1))


def test_no_reference_means_awaiting_reference_not_perfect_or_zero_score():
    pipeline, _ = make_pipeline('{"diagnoses":["Dx-A"]}')
    saved = pipeline.run("case", case_id="case", method="base")
    assert saved["status"] == "awaiting_reference" and "scoring" not in saved
    completed = SevenStagePipeline(pipeline.catalog, pipeline.store).resume(
        saved, reference=["Dx-A"], reference_source="author")
    assert completed["scoring"]["diagnosis"]["F1"] == 1


def test_reference_receives_same_simplification_as_predictions():
    pipeline, _ = make_pipeline('{"diagnoses":["Dx-B"]}')
    result = pipeline.run("case", case_id="case", method="base", reference=["Dx-B, mild"], reference_source="author")
    assert result["scoring"]["diagnosis"]["TP_list"] == ["Dx-B"]
    assert result["stages"][3]["payload"]["reference"]["items"][0]["steps"][0]["rule"] == 1


def test_reference_can_itself_require_review_and_does_not_get_special_gold_matching():
    pipeline, _ = make_pipeline('{"diagnoses":["Dx-A"]}')
    result = pipeline.run("case", case_id="case", method="base", reference=["unmatched-output"], reference_source="author")
    assert result["status"] == "awaiting_reconciliation"
    assert result["stages"][5]["payload"]["reference"]["pending"]
    assert "scoring" not in result


def test_legitimate_empty_prediction_is_successful_and_counts_false_negatives():
    pipeline, _ = make_pipeline('{"diagnoses":[]}')
    result = pipeline.run("case", case_id="case", method="base", reference=["Dx-A"], reference_source="author")
    assert result["status"] == "completed"
    assert result["scoring"]["diagnosis"]["FN_list"] == ["Dx-A"]
    assert result["scoring"]["diagnosis"]["PPV"] == result["scoring"]["diagnosis"]["F1"] == 0


def test_failed_model_has_seven_stage_records_without_fabricated_raw_diagnoses_or_scores():
    pipeline, _ = make_pipeline(ConnectionError("first"), ConnectionError("second"))
    result = pipeline.run("case", case_id="case", method="base", reference=["Dx-A"], reference_source="author")
    assert result["status"] == "failed" and "scoring" not in result
    assert len(result["stages"]) == 7
    assert result["stages"][1]["status"] == "failed"
    assert all(s["status"] == "not_started" for s in result["stages"][2:])


def test_paired_main_metrics_use_common_successful_cases_and_macro_f1():
    pipeline, _ = make_pipeline('{"diagnoses":["Dx-A"]}', '{"diagnoses":[]}', '{"diagnoses":["Dx-A","Dx-B"]}')
    one = pipeline.run("case", case_id="one", method="base", reference=["Dx-A"], reference_source="author")
    two = pipeline.run("case", case_id="two", method="base", reference=["Dx-A"], reference_source="author")
    extra = pipeline.run("case", case_id="extra", method="base", reference=["Dx-A"], reference_source="author")
    result = paired_macro({"base": [one, two, extra], "dt": [one, two, {"case_id": "extra", "status": "failed"}]})
    assert result["n"] == 2 and result["common_successful_case_ids"] == ["one", "two"]
    assert result["macro"]["base"]["diagnosis"]["F1"] == 0.5
    assert result["excluded"]["dt"] == [{"case_id": "extra", "status": "failed"}]
    assert result["excluded"]["base"] == [{"case_id": "extra", "status": "successful_but_not_common"}]
    with pytest.raises(ValueError, match="Duplicate case"):
        paired_macro({"base": [one, one], "dt": [one]})


def test_catalog_standard_and_reference_source_are_checked_before_model_requests():
    pipeline, api = make_pipeline()
    with pytest.raises(ValueError, match="reference source"):
        pipeline.run("case", case_id="case", method="base", reference=["Dx-A"])
    assert not api.requests
    data = catalog_data()
    data["standard"] = "DSM-5-TR"
    with pytest.raises(ValueError, match="standards must match"):
        SevenStagePipeline(Catalog.parse(data), ReconciliationStore(), runner=pipeline.runner)


def test_resume_rejects_failed_or_differently_versioned_model_results():
    pipeline, _ = make_pipeline('{"diagnoses":["Dx-A"]}')
    saved = pipeline.run("case", case_id="case", method="base")
    altered = copy.deepcopy(saved)
    altered["catalog_id"] = "changed"
    with pytest.raises(ValueError, match="same standard"):
        pipeline.resume(altered)
    altered = copy.deepcopy(saved)
    altered["stages"][1]["status"] = "failed"
    with pytest.raises(ValueError, match="completed saved"):
        pipeline.resume(altered)


def test_cli_import_and_offline_resume_save_seven_separate_stage_files(tmp_path):
    from paper_dtree.__main__ import main

    pipeline, api = make_pipeline('{"diagnoses":["unmatched-output"]}')
    saved = pipeline.run("case", case_id="case", method="base", reference=["Dx-A"], reference_source="author")
    catalog_file, result_file, review_file = (tmp_path / name for name in ("catalog.json", "model.json", "review.json"))
    for path, payload in [(catalog_file, catalog_data()), (result_file, saved), (review_file, [review()])]:
        path.write_text(json.dumps(payload), encoding="utf-8")
    db, output = tmp_path / "reviews.sqlite", tmp_path / "continued"
    assert main(["import-reviews", "--catalog", str(catalog_file), "--reconciliation-db", str(db),
                 "--decisions-file", str(review_file)]) == 0
    assert main(["resume", "--input-result", str(result_file), "--catalog", str(catalog_file),
                 "--reconciliation-db", str(db), "--output-dir", str(output)]) == 0
    assert len(list(output.glob("0?_*.json"))) == 7
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["scoring"]["diagnosis"]["TP"] == 1 and len(api.requests) == 1


@pytest.mark.parametrize("response,expected_code,expected_status", [
    ('{"diagnoses":["Dx-A"]}', 0, "completed"),
    ('{"diagnoses":["unmatched-output"]}', 2, "awaiting_reconciliation"),
])
def test_run_cli_uses_all_seven_stages_and_preserves_raw_responses(tmp_path, monkeypatch, response, expected_code, expected_status):
    from paper_dtree.__main__ import main
    from paper_dtree.engine import PAIRWISE_POLICY
    from paper_dtree import transport

    api = Scripted(response)
    monkeypatch.setattr(transport, "OpenAICompatibleTransport", lambda **kwargs: api)
    files = {"bundle.json": bundle_data(tree("T", {"q": node("q")})),
             "catalog.json": catalog_data(),
             "gold.json": {"case_id": "c", "diagnoses": ["Dx-A"], "source_ref": "author"}}
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "case.txt").write_text("fictional raw vignette", encoding="utf-8")
    output = tmp_path / "run"
    code = main(["run", "--bundle", str(tmp_path / "bundle.json"),
                 "--entry-profile", "bundle",
                 "--catalog", str(tmp_path / "catalog.json"), "--reference-file", str(tmp_path / "gold.json"),
                 "--case-file", str(tmp_path / "case.txt"), "--case-id", "c", "--method", "base",
                 "--model", "fixture-model", "--pairwise-policy", PAIRWISE_POLICY,
                 "--reconciliation-db", str(tmp_path / "reviews.sqlite"), "--output-dir", str(output)])
    assert code == expected_code and len(api.requests) == 1
    assert len(list(output.glob("0?_*.json"))) == 7
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert saved["status"] == expected_status
    assert json.loads((output / "03_raw_response.json").read_text(encoding="utf-8"))["payload"]["responses"][0]["response"]["text"] == response

import pytest

from tools.evaluate_candidate_predictions import evaluate


def gold(case_id, labels, kind="single_reference_label", code="D013-D008", origin="g1"):
    return {"case_id": case_id, "gold_labels": labels, "reference_kind": kind,
            "task_type": "type3" if kind == "admissible_candidates" else "type4",
            "source_code": code, "related_case_group": origin,
            "label_origin": "MentalBench_synthetic_graph_grounded"}


def prediction(case_id, labels):
    return {"case_id": case_id, "candidates": [{"disorder": label, "confidence": .8} for label in labels]}


def test_semantic_strata_are_not_pooled_as_one_diagnostic_ppv():
    rows = [gold("a", ["D013"]), gold("b", ["D013", "D008"], "admissible_candidates")]
    predictions = [prediction("a", ["D013"]), prediction("b", ["D013"])]
    predictions[0]["final_diagnoses"] = ["D013"]
    predictions[1]["final_diagnoses"] = ["D013"]
    report = evaluate(rows, predictions)
    assert report["candidate_ppv_micro"] is None
    single = report["by_reference_kind"]["single_reference_label"]
    ambiguous = report["by_reference_kind"]["admissible_candidates"]
    assert single["candidate_recall_micro"] == 1
    assert ambiguous["candidate_recall_micro"] == .5
    assert single["final"]["exact_match_rate"] == 1
    assert ambiguous["final"]["cases_with_outputs"] == 0


def test_macro_average_and_insomnia_outputs_are_explicit():
    rows = [gold("a", ["D013"]), gold("b", ["D013"]), gold("c", ["D008"])]
    report = evaluate(rows, [prediction("a", ["D013", "Insomnia"]), prediction("b", ["D013"]), prediction("c", [])])
    assert report["candidate_recall_micro"] == 2 / 3
    assert report["candidate_recall_macro"] == .5
    assert report["candidate_ppv_micro"] == 2 / 3
    assert "Insomnia" not in report["reference_label_space"]
    assert report["out_of_reference_space_predictions"] == {"Insomnia": 1}
    assert report["per_label"]["Insomnia"]["recall"] is None


def test_full_l3_reports_support_step2_and_final_metrics():
    rows = [gold("a", ["D013"])]
    p = prediction("a", ["D013", "D008"])
    p["checked"] = [{"disorder": "D013", "conclusion": "ruled_out"}, {"disorder": "D008", "conclusion": "watch"}]
    p["diagnosis"] = {"primary": None, "secondary": []}
    report = evaluate(rows, [{"case_id": "a", "report": p}])
    assert report["step2"]["false_elimination_rate"] == 1
    assert report["final"]["reference_hit_rate"] == 0


def test_reused_step1_records_are_not_counted_as_an_independent_step2():
    rows = [gold("a", ["D013"])]
    p = prediction("a", ["D013"])
    p["checked"] = [{"disorder": "D013", "conclusion": "meets_criteria", "source": "step1_tree_records"}]
    p["diagnosis"] = {"primary": {"disorder": "D013"}, "secondary": []}
    report = evaluate(rows, [{"case_id": "a", "report": p}])
    assert report["step2"]["cases_with_decisions"] == 0
    assert report["step2"]["false_elimination_rate"] is None
    assert report["final"]["exact_match_rate"] == 1


def test_group_bootstrap_counts_source_groups_separately_from_records():
    rows = [gold("a", ["D013"], origin="same"), gold("b", ["D013"], origin="same"),
            gold("c", ["D008"], code="D008-D011", origin="other")]
    outputs = [prediction(row["case_id"], row["gold_labels"]) for row in rows]
    report = evaluate(rows, outputs, bootstrap=30)
    assert report["cluster_summary"]["records"] == 3
    assert report["cluster_summary"]["source_groups"] == 2
    assert report["primary_cluster_intervals"]["clusters"] == 2
    assert report["primary_cluster_intervals"]["intervals_95"]["candidate_recall_micro"] == [1, 1]
    assert evaluate(rows, outputs, bootstrap=30) == report
    outputs[0]["candidates"][0]["confidence"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate(rows, outputs)

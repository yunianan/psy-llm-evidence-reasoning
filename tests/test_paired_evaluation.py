import copy
import json
from pathlib import Path

import pytest

from tools import paired_evaluation as paired

FIXTURES = Path(__file__).parent / "fixtures/paired"


def fixture(number):
    root = FIXTURES / f"toy-{number}"
    return (paired.read_rows(root / "gold.jsonl"), paired.read_rows(root / "arm_x.jsonl"),
            paired.read_rows(root / "arm_y.jsonl"), json.loads((root / "expected.json").read_text(encoding="utf-8")))


def monitor_examples():
    def example(kind, count):
        return paired.monitor_block([{"case_id": str(i), "failure_class": kind if i < count else None,
                                      "transport_attempt_errors": 11} for i in range(20)])
    return {"contract2": example("contract", 2), "contract3": example("contract", 3),
            "transport3": example("transport", 3), "transport4": example("transport", 4),
            "recovered_only": example(None, 0)}


@pytest.mark.parametrize("number", range(1, 10))
def test_full_fixture_json_matches_protocol_oracle(number):
    gold, x, y, expected = fixture(number)
    actual = paired.compare(gold, x, y, kind="single_reference_label")
    if number == 9:
        actual = {"comparison": actual, "monitor_variants": monitor_examples()}
    assert actual == expected


def test_original_v1_three_cases_and_boundary_ci():
    rows = [fixture(i) for i in (1, 2, 3)]
    gold, x, y = ([row for values in rows for row in values[index]] for index in range(3))
    result = paired.compare(gold, x, y, kind="single_reference_label")
    assert result["metrics"]["x"]["candidate_recall_micro"] == "2/3"
    assert result["metrics"]["y"]["candidate_ppv_micro"] == "3/5"
    assert result["deltas"]["candidate_recall_micro"]["ci95"] == ["0", "1/2"]
    assert result["deltas"]["candidate_ppv_micro"]["ci95"] == ["0", "1/6"]
    assert result["tests"]["iut"] == {"recall_superior": False, "ppv_noninferior": True, "verdict": "not_win"}


def test_all_comparisons_share_one_plan_and_identical_arms_have_zero_ci(monkeypatch):
    gold, x, _, _ = fixture(5)
    calls = []
    original = paired.shared_draws
    def draws(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(paired, "shared_draws", draws)
    study = paired.evaluate_study(gold, {arm: x for arm in ("A0", "A1", "A2", "A3")}, kind="single_reference_label")
    assert calls == [(2,)]
    for result in study["comparisons"].values():
        assert result["deltas"]["candidate_recall_micro"]["ci95"] == ["0", "0"]
    assert study["decision"]["verdicts"]["H2"] == "exploratory_gate_closed"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "bool", "nan", "no_final", "bad_coverage", "missing_retained", "integrity"])
def test_strict_prediction_contracts(mutation):
    gold, x, y, _ = fixture(5)
    y = copy.deepcopy(y)
    if mutation == "missing": y.pop()
    elif mutation == "duplicate": y.append(copy.deepcopy(y[0]))
    elif mutation == "unknown": y[0]["candidates"][0]["disorder"] = "D099"
    elif mutation == "bool": y[0]["candidates"][0]["confidence"] = True
    elif mutation == "nan": y[0]["candidates"][0]["confidence"] = float("nan")
    elif mutation == "no_final": y[0].pop("final_diagnoses")
    elif mutation == "bad_coverage": y[0]["candidate_coverage"] = {"D001": "not_evaluated"}
    elif mutation == "missing_retained": y[0]["candidate_coverage"] = {"D001": "candidate", "D002": "candidate"}
    elif mutation == "integrity": y[0]["protocol_integrity_failure"] = True
    with pytest.raises(ValueError):
        paired.compare(gold, x, y, kind="single_reference_label")


@pytest.mark.parametrize("mutation", ["missing_group", "invalid_group", "wrong_kind", "wrong_count", "duplicate_gold", "insomnia_gold"])
def test_strict_gold_and_no_group_fallback(mutation):
    gold, x, y, _ = fixture(5)
    gold = copy.deepcopy(gold)
    if mutation == "missing_group": gold[0].pop("source_code")
    elif mutation == "invalid_group": gold[0]["source_code"] = "D099"
    elif mutation == "wrong_kind": gold[0]["reference_kind"] = "unspecified"
    elif mutation == "wrong_count": gold[0]["reference_kind"] = "admissible_candidates"
    elif mutation == "duplicate_gold": gold[0]["gold_labels"] = ["D001", "D001"]
    elif mutation == "insomnia_gold": gold[0]["gold_labels"] = ["Insomnia"]
    with pytest.raises(ValueError):
        paired.compare(gold, x, y, kind="single_reference_label")


def test_undefined_ppv_does_not_invalidate_top5_and_never_resamples():
    gold, x, y, _ = fixture(5)
    result = paired.compare(gold, x, y, kind="single_reference_label", hypothesis="H2")
    assert result["deltas"]["candidate_ppv_micro"]["ci95"] is None
    assert result["deltas"]["candidate_ppv_micro"]["undefined_replicates"] > 0
    assert result["sampling"]["replicates"] == 1000
    assert result["statistical_verdict"] == "not_win"


def test_double_reference_is_not_any_hit_or_final_diagnosis():
    gold = [{"case_id": "d", "gold_labels": ["D011", "D013"], "reference_kind": "admissible_candidates", "source_code": "D013-D011"}]
    rows = [{"case_id": "d", "candidates": [{"disorder": "D013", "confidence": .2}], "final_diagnoses": ["D013"]}]
    result = paired.compare(gold, rows, rows, kind="admissible_candidates")
    assert result["metrics"]["x"]["candidate_recall_micro"] == "1/2"
    assert result["metrics"]["x"]["final.reference_hit_rate"] is None
    assert result["confirmatory_eligible"] is False


def test_ties_insomnia_and_low_confidence_stay_in_denominators():
    gold = [{"case_id": "a", "gold_labels": ["D006"], "reference_kind": "single_reference_label", "source_code": "D006"}]
    rows = [{"case_id": "a", "candidates": [{"disorder": f"D{i:03d}", "confidence": .1} for i in range(6, 0, -1)]
             + [{"disorder": "Insomnia", "confidence": 0}], "final_diagnoses": []}]
    result = paired.compare(gold, rows, rows, kind="single_reference_label")
    assert result["metrics"]["x"]["candidate_ppv_micro"] == "1/7"
    assert result["metrics"]["x"]["top_k_recall_micro"] == "0"


def test_administrative_failure_is_empty_but_not_model_failure():
    gold, x, y, _ = fixture(5)
    y = copy.deepcopy(y)
    y[0].update(run_failure=True, model_failure=False, failure_class="budget")
    result = paired.compare(gold, x, y, kind="single_reference_label")
    assert result["counts"]["y"]["predicted"] == 1
    assert result["failures"]["y"]["model_failure_cases"] == 0
    assert not result["quality_gate_passed"]


def test_exact_equality_never_passes_the_strict_iut_boundary():
    assert paired.iut_verdict(["1/2", "1/2"], ["-1/100", "-1/100"]) == {
        "recall_superior": True, "ppv_noninferior": False, "verdict": "not_win"}


def test_fixed_sequence_does_not_promote_later_winners():
    win = {"confirmatory_eligible": True, "quality_gate_passed": True, "inference_status": "win"}
    loss = {**win, "inference_status": "not_win"}
    assert paired.decide({"H1": win, "H2": loss, "H3": win})["verdicts"] == {
        "H1": "win", "H2": "not_win", "H3": "exploratory_gate_closed"}


@pytest.mark.parametrize("kwargs", [{"top_k": 4}, {"n_boot": 10}, {"seed": 7}])
def test_protocol_parameters_cannot_silently_change(kwargs):
    gold, x, y, _ = fixture(5)
    with pytest.raises(ValueError):
        paired.compare(gold, x, y, kind="single_reference_label", **kwargs)

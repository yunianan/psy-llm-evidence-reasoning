import pytest

from audit_metrics import active_labels, decision, error_counts, percentile


def test_gate_includes_exact_five_percentage_point_boundary():
    full = {"overcall": 10, "nonmet_denominator": 100, "undercall": 4, "met_denominator": 20}
    candidate = {**full, "overcall": 15}
    assert decision(full, candidate, 20, 20)["gate"] == "pass"
    assert decision(full, {**candidate, "overcall": 16}, 20, 20)["gate"] == "fail"


def test_pending_and_wrong_not_met_both_count_as_undercall():
    reference = [{"node_id": node, "reference_status": status} for node, status in [("a", "met"), ("b", "met"), ("c", "pending")]]
    prediction = [{"node_id": node, "status": status} for node, status in [("a", "pending"), ("b", "not_met"), ("c", "met")]]
    counts = error_counts(reference, prediction)
    assert counts["undercall"] == 2 and counts["undercall_pending"] == counts["undercall_not_met"] == 1
    assert counts["overcall"] == 1 and counts["nonmet_denominator"] == 1


def test_pending_to_met_is_not_an_activity_regression():
    graph = {"paths": {"label": {"screens": ["screen"]}}}
    assert active_labels([{"node_id": "screen", "status": "pending"}], graph) == active_labels([{"node_id": "screen", "status": "met"}], graph)
    assert active_labels([{"node_id": "screen", "status": "not_met"}], graph) == []


def test_missing_node_fails_instead_of_dropping_it():
    with pytest.raises(ValueError):
        error_counts([{"node_id": "a", "reference_status": "met"}], [])


def test_cost_percentiles_follow_linear_interpolation():
    assert percentile([1, 2, 3, 4], .5) == 2.5
    assert percentile([1, 2, 3, 4], .95) == pytest.approx(3.85)

"""Predeclared screen-audit counting, independent of model calls."""
from fractions import Fraction


def active_labels(judgments, graph):
    by_id = {item["node_id"]: item["status"] for item in judgments}
    return sorted(label for label, path in graph["paths"].items()
                  if any(by_id[node] != "not_met" for node in path["screens"]))


def error_counts(reference, judgments):
    expected = {item["node_id"]: item["reference_status"] for item in reference}
    observed = {item["node_id"]: item["status"] for item in judgments}
    if len(observed) != len(judgments) or set(observed) != set(expected):
        raise ValueError("All reference nodes require one prediction")
    return {
        "nonmet_denominator": sum(value != "met" for value in expected.values()),
        "met_denominator": sum(value == "met" for value in expected.values()),
        "overcall": sum(value != "met" and observed[node] == "met" for node, value in expected.items()),
        "undercall": sum(value == "met" and observed[node] != "met" for node, value in expected.items()),
        "undercall_pending": sum(value == "met" and observed[node] == "pending" for node, value in expected.items()),
        "undercall_not_met": sum(value == "met" and observed[node] == "not_met" for node, value in expected.items()),
    }


def decision(full, candidate, identical_cases, case_count):
    denominators = [full["nonmet_denominator"], full["met_denominator"],
                    candidate["nonmet_denominator"], candidate["met_denominator"]]
    if not all(denominators):
        return {"gate": "not_estimable"}
    overcall_ok = Fraction(candidate["overcall"], candidate["nonmet_denominator"]) <= (
        Fraction(full["overcall"], full["nonmet_denominator"]) + Fraction(5, 100)
    )
    undercall_ok = Fraction(candidate["undercall"], candidate["met_denominator"]) <= Fraction(full["undercall"], full["met_denominator"])
    activity_ok = identical_cases == case_count and case_count > 0
    return {"overcall_noninferiority_pass": overcall_ok, "undercall_not_worse_pass": undercall_ok,
            "activity_sets_identical_pass": activity_ok, "gate": "pass" if overcall_ok and undercall_ok and activity_ok else "fail"}


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

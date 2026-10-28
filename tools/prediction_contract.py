"""Shared historical normalization and strict v1.1 scoring contracts; no inference."""
import math

from shared.disorder_catalog import SUPPORTED_DISORDERS
from shared.candidate_ordering import candidate_sort_key
from tools.mentalbench_partition import family_key

KINDS = {"single_reference_label": 1, "admissible_candidates": 2}
GOLD_LABELS = {f"D{i:03d}" for i in range(1, 24)}
KEEP = {"candidate", "provisional_candidate"}
COVERAGE = KEEP | {"excluded", "not_triggered", "entry_pending", "anchor_pending"}


def normalize_prediction(row):
    """Historical evaluate() behavior, moved unchanged out of its CLI module."""
    result = dict(row.get("report", row))
    result["case_id"] = row["case_id"]
    failed = row.get("model_failure", result.get("model_failure", False))
    if type(failed) is not bool:
        raise ValueError("model_failure must be an explicit boolean")
    result["model_failure"] = failed
    if failed:
        result.update(candidates=[], candidate_coverage={}, final_diagnoses=[])
        result.pop("checked", None)
        result.pop("step2_decisions", None)
        return result
    if "checked" in result and "step2_decisions" not in result:
        legacy = [item for item in result["checked"] if item.get("source") != "step1_tree_records"]
        if legacy:
            result["step2_decisions"] = {item["disorder"]: item["conclusion"] for item in legacy}
    if "diagnosis" in result and "final_diagnoses" not in result:
        diagnosis = result["diagnosis"]
        result["final_diagnoses"] = [item["disorder"] for item in [diagnosis.get("primary"), *diagnosis.get("secondary", [])] if item]
    return result


def keyed(rows):
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str) or not row["case_id"].strip():
            raise ValueError("Expected a nonempty case_id")
        if row["case_id"] in result:
            raise ValueError("Duplicate case_id")
        result[row["case_id"]] = row
    return result


def validate_gold(rows):
    result = keyed(rows)
    if not result:
        raise ValueError("Gold must not be empty")
    for row in result.values():
        labels, kind = row.get("gold_labels"), row.get("reference_kind")
        if kind not in KINDS or not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
            raise ValueError("Invalid reference_kind/gold_labels")
        if len(labels) != KINDS[kind] or len(set(labels)) != len(labels) or not set(labels) <= GOLD_LABELS:
            raise ValueError("Invalid gold labels or stratum cardinality")
        code = row.get("source_code")
        if not isinstance(code, str) or not set(code.split("-")) <= GOLD_LABELS:
            raise ValueError("A valid source_code is required; no group fallback")
        group = family_key(code)
        if "supergroup" in row and row["supergroup"] != group:
            raise ValueError("supergroup disagrees with source_code")
    return result


def normalize_for_protocol(row):
    if row.get("protocol_integrity_failure"):
        raise ValueError("Protocol/cache integrity failures cannot be scored as ordinary empty predictions")
    if "report" in row and not isinstance(row["report"], dict):
        raise ValueError("report must be an object")
    result = normalize_prediction(row)
    run_failure = row.get("run_failure", result.get("run_failure", False))
    if type(run_failure) is not bool:
        raise ValueError("run_failure must be boolean")
    failure_class = row.get("failure_class", row.get("failure_kind", result.get("failure_class")))
    if failure_class is not None and not isinstance(failure_class, str):
        raise ValueError("failure_class must be string or null")
    failed = run_failure or result["model_failure"] or failure_class is not None
    if failed:
        result.update(candidates=[], candidate_coverage={}, final_diagnoses=[])
    result["complete"] = not failed
    result["failure_class"] = failure_class or ("model_or_contract" if failed else None)
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Explicit candidates list required")
    labels = []
    for item in candidates:
        if not isinstance(item, dict) or item.get("disorder") not in SUPPORTED_DISORDERS:
            raise ValueError("Invalid candidate label")
        value = item.get("confidence")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Confidence must be finite numeric [0,1], not bool")
        labels.append(item["disorder"])
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate candidate label")
    coverage = result.get("candidate_coverage")
    if coverage is not None and not isinstance(coverage, dict):
        raise ValueError("candidate_coverage must be an object")
    if coverage:
        if not set(coverage) <= set(SUPPORTED_DISORDERS) or any(
            not isinstance(s, str) or not (s in COVERAGE or s.startswith("excluded:")) for s in coverage.values()
        ):
            raise ValueError("Invalid coverage state")
        if not set(labels) <= set(coverage):
            raise ValueError("Candidate missing from coverage")
        keep = {label for label, status in coverage.items() if status in KEEP}
        candidates = [c for c in candidates if c["disorder"] in keep]
        if {c["disorder"] for c in candidates} != keep:
            raise ValueError("Missing retained candidate")
    result["candidates"] = sorted(candidates, key=candidate_sort_key)
    if "final_diagnoses" not in result or not isinstance(result["final_diagnoses"], list):
        raise ValueError("Explicit final_diagnoses or full diagnosis required")
    final = result["final_diagnoses"]
    if any(not isinstance(x, str) for x in final) or len(final) != len(set(final)) or not set(final) <= set(SUPPORTED_DISORDERS):
        raise ValueError("Invalid final_diagnoses")
    if not failed and "diagnosis" in result:
        diagnosis = result["diagnosis"]
        expected = [x["disorder"] for x in [diagnosis.get("primary"), *diagnosis.get("secondary", [])] if x]
        if set(expected) != set(final):
            raise ValueError("Final diagnoses disagree with diagnosis")
    return result

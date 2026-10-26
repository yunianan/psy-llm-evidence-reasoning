"""Read-only accounting of the frozen full/computed comparison, including failures."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment as experiment


def quantile(values, fraction):
    if not values:
        return None
    items = sorted(values)
    index = (len(items) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(items) - 1)
    return items[lower] + (items[upper] - items[lower]) * (index - lower)


def distribution(values):
    return {"n": len(values), "sum": sum(values), "p50": quantile(values, .5), "p95": quantile(values, .95)}


def usage(requests):
    totals = experiment.extract(requests)
    totals["known_usage_totals"] = experiment.extract([row for row in requests if row.get("usage")])
    totals["total_tokens"] = (totals["prompt_tokens"] + totals["completion_tokens"]
                              if totals["prompt_tokens"] is not None and totals["completion_tokens"] is not None else None)
    totals["known_subtotals"] = {field: sum((r.get("usage") or {}).get(field) or 0 for r in requests)
                                 for field in ["prompt_tokens", "completion_tokens", "total_tokens"]}
    totals["missing_usage_requests"] = sum(not r.get("usage") for r in requests)
    totals["http_errors"] = dict(Counter(r.get("error_category") or r["error_type"]
                                        for r in requests if r.get("error_type")))
    totals["http_retries"] = sum(r["attempt"] > 1 for r in requests)
    totals["structured_retry_http_attempts"] = sum(bool(r["is_structured_retry"]) for r in requests)
    totals["finish_reasons"] = dict(Counter(r["finish_reason"] for r in requests if r.get("finish_reason") is not None))
    totals["elapsed_seconds"] = distribution([r["elapsed_seconds"] for r in requests if r.get("elapsed_seconds") is not None])
    return totals


def output_effects(full, computed):
    if not full or not computed or not full["ok"] or not computed["ok"]:
        return None
    left, right = full["result"], computed["result"]
    snapshots = [experiment.final_snapshot(value) for value in [left, right]]
    candidate_maps = [{value["disorder"]: value for value in result["candidates"]} for result in [left, right]]
    confidence_changes = [{"disorder": key, "full": candidate_maps[0][key]["confidence"],
                           "computed": candidate_maps[1][key]["confidence"]}
                          for key in sorted(set(candidate_maps[0]) & set(candidate_maps[1]))
                          if candidate_maps[0][key]["confidence"] != candidate_maps[1][key]["confidence"]]
    match_changes = [{"disorder": key, "full": candidate_maps[0][key]["symptom_match"],
                      "computed": candidate_maps[1][key]["symptom_match"]}
                     for key in sorted(set(candidate_maps[0]) & set(candidate_maps[1]))
                     if candidate_maps[0][key]["symptom_match"] != candidate_maps[1][key]["symptom_match"]]
    ranks = [[value["disorder"] for value in sorted(result["candidates"],
              key=lambda value: (-value["confidence"], value["disorder"]))] for result in [left, right]]
    diagnostic_keys = ["primary", "secondary", "provisional_primary"]
    def question_fields(value):
        if isinstance(value, dict):
            found = [value["field"]] if isinstance(value.get("field"), str) else []
            return found + [field for child in value.values() for field in question_fields(child)]
        if isinstance(value, list):
            return [field for child in value for field in question_fields(child)]
        return []
    missing = [sorted(set(question_fields(result.get("missing_dimensions", []))
                          + question_fields(result.get("candidate_missing_dimensions", {})))) for result in [left, right]]
    return {"case_id": full["case_id"], "diagnostic_labels_identical": all(snapshots[0][key] == snapshots[1][key] for key in diagnostic_keys),
            "both_have_no_definitive_diagnosis": all(not row["primary"] and not row["secondary"] for row in snapshots),
            "assessment_conclusions_identical": snapshots[0]["assessment_conclusions"] == snapshots[1]["assessment_conclusions"],
            "verifier_routing_identical": snapshots[0]["verifier_routing"] == snapshots[1]["verifier_routing"],
            "candidate_confidence_changes": confidence_changes, "candidate_ranking_identical": ranks[0] == ranks[1],
            "candidate_symptom_match_changes": match_changes,
            "other_result_fields_changed": [key for key in ["severity", "severity_level", "verifier_score", "verifier_decision",
                                          "verifier_checks", "verifier_fail_reasons", "flags", "candidate_confidence", "confidence"]
                                            if left.get(key) != right.get(key)],
            "ranks": {"full": ranks[0], "computed": ranks[1]},
            "case_confidence": {"full": left.get("confidence"), "computed": right.get("confidence")},
            "missing_question_fields_identical": missing[0] == missing[1],
            "missing_question_fields_added": sorted(set(missing[1]) - set(missing[0])),
            "missing_question_fields_removed": sorted(set(missing[0]) - set(missing[1])),
            "computed_validated_fact_batches": len(computed.get("computation_log", []))}


def failure_classification(record):
    raw = record.get("failure_kind")
    types = {value["type"] for value in record.get("exception_chain", [])}
    # The frozen runner checked exact class names, missing JSONDecodeError,
    # which is a ValueError subclass. Preserve raw labels and report this audit.
    contract = raw in {"model_or_runner_failure", "model_failure"} and "JSONDecodeError" in types
    return {"raw": raw, "audited": "contract_failure" if contract else raw,
            "basis": "JSONDecodeError after the one structured repair" if contract else "unchanged"}


def partial_checkpoints():
    checkpoints = {}
    for arm in ["full", "computed"]:
        checkpoints[arm] = {}
        for path in (HERE / "checkpoints" / arm).glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            checkpoints[arm][value["context"]["case_id"]] = value["nodes"]
    result = []
    for case_id, nodes in checkpoints["computed"].items():
        baseline = checkpoints["full"].get(case_id, {})
        ids = [key for key, value in nodes.items() if value["definition"]["stage"] == "criterion" and key in baseline]
        differences = [{"node_id": key, "full": baseline[key]["judgment"]["status"], "computed": nodes[key]["judgment"]["status"]}
                       for key in sorted(ids) if baseline[key]["judgment"]["status"] != nodes[key]["judgment"]["status"]]
        result.append({"case_id": case_id, "validated_computed_criteria": len(ids), "differences": differences,
                       "not_a_complete_pair_or_acceptance_result": True})
    return result


def paired_resources(comparisons, resources):
    complete = [row for row in comparisons if row["complete"]]
    indexed = {(row["case_id"], row["arm"]): row for row in resources}
    known = [row for row in complete if all(indexed[(row["case_id"], arm)]["fresh_case_usage_estimate"]["prompt_tokens"] is not None
                                          and indexed[(row["case_id"], arm)]["fresh_case_usage_estimate"]["completion_tokens"] is not None
                                          for arm in ["full_llm", "computed"])]
    result = {"complete_pair_count": len(complete), "complete_usage_pair_count": len(known),
              "complete_usage_case_ids": [row["case_id"] for row in known],
              "excluded_missing_usage_case_ids": [row["case_id"] for row in complete if row not in known]}
    for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        values = {}
        for arm in ["full_llm", "computed"]:
            items = [indexed[(row["case_id"], arm)]["fresh_case_usage_estimate"][field] for row in known]
            values[arm] = distribution(items)
        baseline = values["full_llm"]["sum"]
        values["aggregate_reduction_fraction"] = (baseline - values["computed"]["sum"]) / baseline if baseline else None
        result[field] = values
    result["elapsed_seconds"] = {arm: distribution([indexed[(row["case_id"], arm)]["fresh_case_elapsed_estimate_seconds"] for row in complete])
                                 for arm in ["full_llm", "computed"]}
    return result


def main():
    run = experiment.RUN
    records = experiment.infra.read(run / "results.jsonl")
    journal = experiment.infra.read(run / "requests.jsonl")
    latest = {r["attempt_id"]: r for r in journal}
    requests = list(latest.values())
    arms = defaultdict(list)
    stages = defaultdict(list)
    for r in requests:
        arms[r["arm"]].append(r)
        stages[r["arm"] + ":" + r["stage"]].append(r)
    paired = defaultdict(dict)
    for r in records:
        paired[r["case_id"]][r["arm"]] = r
    comparisons = [experiment.comparison(values.get("full_llm"), values.get("computed")) for values in paired.values()]
    complete = [r for r in comparisons if r["complete"]]
    denominator = sum(r["criteria_nodes"] for r in complete)
    changed = sum(r["criteria_status_changed"] for r in complete)
    per_case_resources = []
    for case_id, values in paired.items():
        for arm, row in values.items():
            prefix = [r for r in values.get("full_llm", {}).get("requests", []) if r["stage"] not in {"criteria", "verifier"}]
            fresh = prefix + row["requests"] if arm == "computed" else row["requests"]
            per_case_resources.append({"case_id": case_id, "arm": arm, "ok": row["ok"],
                "incremental_usage": usage(row["requests"]), "fresh_case_usage_estimate": usage(fresh),
                "incremental_elapsed_seconds": row.get("elapsed_seconds"),
                "fresh_case_elapsed_estimate_seconds": (row.get("elapsed_seconds", 0) + sum(r.get("elapsed_seconds", 0) for r in prefix)) if arm == "computed" else row.get("elapsed_seconds"),
                "fresh_case_estimate_includes_observed_prefix": arm == "computed"})
    result = {
        "analysis_at": experiment.now(), "execution_status": "finished" if (run / "summary.json").exists() else "running",
        "finished_arms": len(records), "successful_arms": sum(r["ok"] for r in records),
        "failures": [{k: r.get(k) for k in ["case_id", "arm", "failure_kind", "exception_chain"]} for r in records if not r["ok"]],
        "failure_classification_audit": [{"case_id": r["case_id"], "arm": r["arm"], **failure_classification(r)} for r in records if not r["ok"]],
        "audited_failure_counts": dict(Counter(failure_classification(r)["audited"] for r in records if not r["ok"])),
        "complete_pairs": len(complete), "passing_pairs": sum(r["passed"] for r in complete),
        "criteria_status_numerator": denominator - changed, "criteria_status_denominator": denominator,
        "criteria_status_agreement_overall": (denominator - changed) / denominator if denominator else None,
        "final_identical_pairs": sum(r["final_identical"] for r in complete),
        "comparisons": comparisons, "all_http": usage(requests),
        "output_effects": [effect for values in paired.values()
                           if (effect := output_effects(values.get("full_llm"), values.get("computed"))) is not None],
        "usage_by_arm": {key: usage(value) for key, value in arms.items()},
        "terminal_arms_by_type": {arm: {"completed": sum(r["arm"] == arm for r in records),
                                       "successful": sum(r["arm"] == arm and r["ok"] for r in records)} for arm in ["full_llm", "computed"]},
        "per_case_resources": per_case_resources,
        "paired_resources": paired_resources(comparisons, per_case_resources),
        "partial_checkpoint_comparisons": partial_checkpoints(),
        "usage_by_arm_and_stage": {key: usage(value) for key, value in stages.items()},
        "in_flight": [{k: r.get(k) for k in ["case_id", "arm", "stage", "attempt", "started_at"]}
                      for r in requests if r["event"] == "attempt_started"],
        "criteria_initial_calls_full": distribution([r["initial_criteria_calls_full"] for r in complete]),
        "criteria_initial_calls_computed": distribution([r["initial_criteria_calls_computed"] for r in complete]),
        "full_case_http_attempts": distribution([r["full_case_http_attempts"] for r in complete]),
        "fresh_computed_http_attempt_estimate": distribution([r["fresh_case_http_attempt_estimate_computed"] for r in complete]),
        "status_difference_nodes": dict(Counter(d["node_id"] for r in complete for d in r["status_differences"])),
        "g1_replay_completed_arms": sum("identical" in r["g1_shared_validation"] for r in records),
        "g1_replay_failed_arms": sum(r["g1_shared_validation"].get("identical") is False for r in records),
        "actual_platform_invoice": None, "formal_N_frozen": False,
        "cost_caveats": ["Authorized official-price reference, not the proxy invoice",
                         "Known subtotals exclude missing usage; complete totals remain null",
                         "Computed arm reuses the paired prefix; fresh-case calls add that prefix back",
                         "Reasoning is a subset of completion, never billed twice"],
    }
    experiment.infra.cost.write(HERE / "analysis.json", result)
    print(json.dumps({key: result[key] for key in ["execution_status", "finished_arms", "successful_arms", "complete_pairs", "passing_pairs", "final_identical_pairs", "terminal_arms_by_type", "criteria_status_agreement_overall", "in_flight"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

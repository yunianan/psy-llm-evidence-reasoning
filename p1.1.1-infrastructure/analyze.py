"""Offline audit accounting; no model calls and no gold-label scoring."""
from collections import Counter
import json

from common import HERE, cost, read
from tools.l3_engineering_runtime import reasoning_observation, recover_ledger


def usage_summary(requests):
    answer = cost.accounting(requests)
    fields = {"cached_input_tokens": [], "uncached_input_tokens": [], "reasoning_tokens": []}
    for request in requests:
        usage = request.get("usage") or {}
        cached = usage.get("prompt_cache_hit_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        prompt = usage.get("prompt_tokens")
        fields["cached_input_tokens"].append(cached)
        fields["uncached_input_tokens"].append(prompt - cached if isinstance(prompt, int) and isinstance(cached, int) else None)
        fields["reasoning_tokens"].append(reasoning_observation(request)["effective_reasoning_tokens"])
    for name, values in fields.items():
        known = sum(value for value in values if isinstance(value, int))
        missing = sum(value is None for value in values)
        answer[name] = {"total": None if missing else known, "known_sum": known, "missing_requests": missing}
    return answer


def analyze(folder):
    rows = read(folder / "results.jsonl")
    latest = {event["attempt_id"]: event for event in read(folder / "requests.jsonl")}
    requests = list(latest.values())
    answer = {"completed_case_arms": len(rows), "http_attempts": len(requests),
              "unfinished_attempts": sum(r["event"] == "attempt_started" for r in requests),
              "budget_ledger": recover_ledger(folder / "requests.jsonl"), "usage": usage_summary(requests),
              "per_stage": {}, "per_arm": {}, "failures": [], "request_errors": []}
    for stage in sorted({r["stage"] for r in requests}):
        selected = [r for r in requests if r["stage"] == stage]
        answer["per_stage"][stage] = {"http_attempts": len(selected), "logical_requests": sum(r["attempt"] == 1 for r in selected),
            "structured_repairs": sum(r["attempt"] == 1 and r["is_structured_retry"] for r in selected),
            "usage": usage_summary(selected), "latency_seconds": cost.distribution([r.get("elapsed_seconds") for r in selected])}
    for arm in sorted({r["arm"] for r in rows}):
        selected_rows = [r for r in rows if r["arm"] == arm]
        selected = [r for r in requests if r["arm"] == arm]
        answer["per_arm"][arm] = {"attempted": len(selected_rows), "complete": sum(r["ok"] for r in selected_rows),
            "infrastructure_successful": sum(r["infrastructure_success"] for r in selected_rows),
            "usage_all_attempts": usage_summary(selected),
            "all_case_latency_seconds": cost.distribution([r["elapsed_seconds"] for r in selected_rows]),
            "successful_case_latency_seconds": cost.distribution([r["elapsed_seconds"] for r in selected_rows if r["ok"]]),
            "successful_case_total_tokens": cost.distribution([r["cost_accounting"]["total_tokens"]["total"] for r in selected_rows if r["ok"]]),
            "reasoning_field_basis": dict(Counter(reasoning_observation(r)["basis"] for r in selected))}
    for row in rows:
        if not row["ok"]:
            detail = {key: row.get(key) for key in ["case_id", "arm", "failure_kind", "exception_chain", "elapsed_seconds", "checkpoint_nodes_after"]}
            last = (row.get("requests") or [{}])[-1]
            detail["last_request"] = {key: last.get(key) for key in ["stage", "node_ids", "batch_number", "structured_attempt", "attempt", "status_code", "error_type"]}
            answer["failures"].append(detail)
    for request in requests:
        if request.get("error_type"):
            detail = {key: request.get(key) for key in ["case_id", "arm", "stage", "batch_number", "node_ids", "attempt", "error_type", "error_category", "exception_types", "status_code", "provider_error_code", "response_request_id", "elapsed_seconds", "retryable", "usage", "finish_reason", "time_to_first_content_seconds"]}
            detail["output_characters"] = len(request.get("raw_output") or "")
            try:
                json.loads(request.get("raw_output") or "")
                detail["complete_json"] = True
            except ValueError:
                detail["complete_json"] = False
            answer["request_errors"].append(detail)
    paired = {}
    for row in rows:
        paired.setdefault(row["case_id"], {})[row["arm"]] = row
    complete_pairs = [arms for arms in paired.values() if all(arm in arms and arms[arm]["ok"] for arm in ["thinking_on", "thinking_off"])]
    if complete_pairs:
        stages, shared, changed = {}, 0, 0
        for pair in complete_pairs:
            on, off = [pair[arm] for arm in ["thinking_on", "thinking_off"]]
            left, right = [{node["node_id"]: node for node in row["result"]["tree_trace"]} for row in [on, off]]
            stage_by_node = {node_id: request["stage"] for row in [on, off] for request in row["requests"] for node_id in request["node_ids"]}
            for node_id in set(left) & set(right):
                stage = stage_by_node[node_id]
                transition = left[node_id]["status"] + " -> " + right[node_id]["status"]
                counts = stages.setdefault(stage, Counter())
                counts[transition] += 1
                shared += 1
                changed += left[node_id]["status"] != right[node_id]["status"]
        answer["paired_state_comparisons"] = {"complete_pairs": len(complete_pairs),
            "incomplete_or_failed_pairs": len(paired) - len(complete_pairs),
            "shared_visited_nodes": shared, "shared_node_status_differences": changed,
            "on_to_off_status_counts_by_stage": {stage: dict(counts) for stage, counts in stages.items()},
            "scope": "on is a comparator, not clinical gold; only jointly visited nodes in complete pairs"}
        answer["paired_successes_only_cost_description"] = {"cases": len(complete_pairs), "scope": "descriptive paired complete runs; failures remain in all-attempt metrics",
                "arms": {arm: {"total_tokens": cost.distribution([pair[arm]["cost_accounting"]["total_tokens"]["total"] for pair in complete_pairs]),
                               "elapsed_seconds": cost.distribution([pair[arm]["elapsed_seconds"] for pair in complete_pairs])}
                         for arm in ["thinking_on", "thinking_off"]}}
    return answer


def main():
    result = {}
    for name in ["thinking-audit-v1", "repair9-v1"]:
        folder = HERE / "runs" / name
        if (folder / "summary.json").exists() or (folder / "stopped-summary.json").exists():
            result[name] = analyze(folder)
            result[name]["execution_status"] = "completed" if (folder / "summary.json").exists() else "stopped_by_user"
    cost.write(HERE / "resource-analysis.json", result)
    print(json.dumps({name: {key: value[key] for key in ["completed_case_arms", "http_attempts", "unfinished_attempts"]} for name, value in result.items()}))


if __name__ == "__main__":
    main()

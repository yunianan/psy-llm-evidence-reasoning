"""Offline usage extraction and user-authorized official-price scenarios."""
import json
from collections import Counter
from math import floor

from common import HERE, cost, read
from tools.l3_engineering_runtime import reasoning_observation


RATES = {"off_peak": {"miss": 1, "hit": .02, "output": 4},
         "peak": {"miss": 2, "hit": .04, "output": 8}}


def extract(requests):
    values = {name: [] for name in ["prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens", "reasoning_tokens"]}
    for request in requests:
        usage = request.get("usage") or {}
        prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        hit = usage.get("prompt_cache_hit_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        miss = usage.get("prompt_cache_miss_tokens")
        if miss is None and isinstance(prompt, int) and isinstance(hit, int):
            miss = prompt - hit
        for name, value in [("prompt_tokens", prompt), ("prompt_cache_hit_tokens", hit),
                            ("prompt_cache_miss_tokens", miss), ("completion_tokens", output),
                            ("reasoning_tokens", reasoning_observation(request)["effective_reasoning_tokens"])]:
            values[name].append(value)
    result = {name: sum(items) if all(isinstance(v, int) for v in items) else None for name, items in values.items()}
    result["requests"] = len(requests)
    result["reasoning_is_subset_of_completion_not_extra_fee"] = True
    result["reasoning_accounting_basis"] = dict(Counter(reasoning_observation(r)["basis"] for r in requests))
    result["reference_cny"] = {}
    for period, rates in RATES.items():
        required = [result[key] for key in ["prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens"]]
        result["reference_cny"][period] = None if any(v is None for v in required) else (
            result["prompt_cache_hit_tokens"] * rates["hit"] + result["prompt_cache_miss_tokens"] * rates["miss"]
            + result["completion_tokens"] * rates["output"]) / 1e6
    return result


def mean_fee(rows, period):
    return sum(row["reference_cny"][period] for row in rows) / len(rows)


def main():
    audit = read(HERE / "runs/thinking-audit-v1/results.jsonl")
    successes = [{"case_id": r["case_id"], "arm": r["arm"], **extract(r["requests"])} for r in audit if r["ok"]]
    assert len(successes) == 14
    old_dir = HERE.parent / "p1.1-engineering/runs"
    old = [r for name in ["cached-full-smoke-v1", "cached-full-smoke25-continuation-v1", "cached-full-smoke25-bounded-v1"]
           for r in read(old_dir / name / "results.jsonl") if r["attempted"] and r["ok"]]
    assert len(old) == 16
    old_g1 = [{"case_id": r["case_id"], **extract([q for q in r["requests"] if q["stage"] != "verifier"])} for r in old]
    old_verifier = [extract([q for q in r["requests"] if q["stage"] == "verifier"]) for r in old]
    on = [r for r in successes if r["arm"] == "thinking_on"]
    result = {"pricing_basis": "User-authorized official DeepSeek CNY reference snapshot dated 2026-10-24; actual proxy price unverified",
        "rates_per_million": RATES, "actual_platform_invoice": None,
        "audit_successful_arms": successes, "audit_success_counts": dict(Counter(r["arm"] for r in successes)),
        "old_16_g1": old_g1, "old_16_conditional_verifier": old_verifier,
        "cost_formula": "(cache_miss*miss_rate + cache_hit*hit_rate + completion*output_rate)/1e6; reasoning already in completion",
        "budget_cny": 300, "reserve_fraction_assumed": .2, "available_for_planned_work_cny": 240,
        "scenarios": {}, "formal_N_frozen": False,
        "limits": ["10 off successes excluded from thinking-on budget baseline", "4 on G1 and old 16 successful G1 are descriptive cohorts",
                   "repair/failure overhead must be updated from the repair ledger", "G2/C/adjudication extra costs unknown, not zero", "reference prices do not establish actual proxy charges"]}
    for period in RATES:
        audit_fee, old_fee = mean_fee(on, period), mean_fee(old_g1, period)
        baseline = max(audit_fee, old_fee)
        verifier = mean_fee(old_verifier, period)
        known_shared_three_arm_cost = baseline + 3 * verifier
        result["scenarios"][period] = {"audit_4_on_g1_mean_cny": audit_fee, "old_16_g1_mean_cny": old_fee,
            "planning_g1_mean_cny": baseline, "basis": "larger observed mean, not a frozen upper confidence bound",
            "old_16_conditional_verifier_mean_cny": verifier,
            "known_shared_g1_plus_3_conditional_verifiers_cny": known_shared_three_arm_cost,
            "budget_ceiling_profiles_g1_only": floor(240 / baseline),
            "budget_ceiling_N_each_of_DEV_and_EVAL_known_components_only": floor(240 / (2 * known_shared_three_arm_cost)),
            "additional_g2_case_adjudication_per_profile_cny": None}
    failures = [q for r in audit for q in r["requests"] if q.get("error_type")]
    result["historical_error_inventory"] = [{key: q.get(key) for key in ["case_id", "arm", "stage", "batch_number", "node_ids", "error_type", "status_code", "provider_error_code", "response_request_id", "started_at", "elapsed_seconds"]} for q in failures]
    result["historical_error_counts"] = {"timeout": sum(q.get("error_type") == "ReadTimeout" for q in failures),
        "unknown_stream_APIError": sum(q.get("error_type") == "APIError" for q in failures),
        "confirmed_connection_reset": 0, "confirmed_429": 0, "confirmed_5xx": 0}
    cost.write(HERE / "successful-usage-and-budget.json", result)
    print(json.dumps({key: result[key] for key in ["audit_success_counts", "historical_error_counts", "scenarios"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

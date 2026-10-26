"""Official-price scenarios, never a claim about the configured proxy's invoice."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFICIAL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"


def tokens(requests):
    result = {"input": 0, "cache_hit": 0, "cache_miss": 0, "output": 0}
    for request in requests:
        usage = request["usage"]
        hit = usage.get("prompt_cache_hit_tokens")
        if hit is None:
            hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if hit is None:
            raise ValueError("Cache usage not exposed; do not silently assign zero")
        result["input"] += usage["prompt_tokens"]
        result["cache_hit"] += hit
        result["cache_miss"] += usage["prompt_tokens"] - hit
        result["output"] += usage["completion_tokens"]
    return result


def price(values, peak=False):
    offpeak = (values["cache_miss"] + .02 * values["cache_hit"] + 4 * values["output"]) / 1e6
    return offpeak * (2 if peak else 1)


def partial_reference(requests):
    """Expose known usage without inventing tokens or prices for failed transports."""
    complete = [r for r in requests if (r.get("usage") or {}).get("prompt_tokens") is not None
                and (r.get("usage") or {}).get("completion_tokens") is not None]
    values = tokens(complete)
    unmetered = len(requests) - len(complete)
    extra_attempts = sum(max(0, (r.get("transport_attempts") or 1) - 1) for r in requests)
    return {"logical_requests": len(requests), "metered_requests": len(complete),
            "unmetered_requests": unmetered, "extra_sdk_attempts_without_separate_usage": extra_attempts,
            "known_tokens": values, "total_tokens": None if unmetered else values["input"] + values["output"],
            "known_offpeak_reference_yuan": price(values), "known_peak_reference_yuan": price(values, True),
            "full_request_billing_known": unmetered == 0 and extra_attempts == 0,
            "actual_platform_price_verified": False}


def main():
    requests = [json.loads(x) for x in (HERE.parent / "p1.0.8b/runs/full-a1-v1/requests.jsonl").read_text(encoding="utf-8").splitlines()]
    all_tokens = tokens(requests)
    g1 = tokens([r for r in requests if r["stage"] != "verifier"])
    verifier = tokens([r for r in requests if r["stage"] == "verifier"])
    output = {"price_source": OFFICIAL, "price_checked_date": "2026-10-23", "currency": "CNY",
              "configured_platform_prices_verified": False, "official_reference_authorized": True,
              "latest_authorized_smoke_total_attempts": 25,
              "formal_total_budget": 300, "reserve_fraction": .2, "formal_operating_reference_budget": 240,
              "old20_tokens": all_tokens, "old20_offpeak_reference_yuan": price(all_tokens),
              "old20_peak_reference_yuan": price(all_tokens, True),
              "old20_mean_output": all_tokens["output"] / 20,
              "g1_once_per_profile_offpeak_reference": price(g1) / 20,
              "a1_verifier_per_profile_offpeak_reference": price(verifier) / 20,
              "unmeasured": ["G2 retrieval/LLM overhead", "C retrieval/case briefs", "adjudication/extra verifier", "index/card building and embeddings"],
              "formal_N_frozen": False, "target_N_400_not_yet_affirmed": True}
    # N means DEV=N and EVAL=N, not an assertion that both splits will be run now.
    upper = {}
    for peak in [False, True]:
        g1_per = price(g1, peak) / 20
        three_verifiers = 3 * price(verifier, peak) / 20
        upper["peak" if peak else "offpeak"] = {
            "G1_only_upper_N_per_split_before_G2_C_and_all_verifiers": math.floor(240 / (2 * g1_per)),
            "G1_plus_3_old_verifier_allowances_upper_N_before_G2_C": math.floor(240 / (2 * (g1_per + three_verifiers))),
            "N400_G1_only_yuan": 800 * g1_per,
            "N400_G1_and_3_old_verifier_allowances_yuan": 800 * (g1_per + three_verifiers)}
    output["old_baseline_optimistic_upper_bounds_not_sample_size_commitments"] = upper
    live_requests = []
    for name in ("cached-full-smoke-v1", "cached-full-smoke25-continuation-v1", "cached-full-smoke25-bounded-v1"):
        live = HERE / "runs" / name / "requests.jsonl"
        if live.exists():
            rows = [json.loads(x) for x in live.read_text(encoding="utf-8").splitlines()]
            output[name] = partial_reference(rows)
            output[name]["run_finished"] = (live.parent / "summary.json").exists()
            live_requests += rows
    if live_requests:
        output["combined_smoke_reference"] = partial_reference(live_requests)
        output["new_successful_case_budget_projection"] = "not frozen; use complete cases only, plus all failed consumption and G2/C overhead"
    combined = HERE / "smoke25-summary.json"
    if combined.exists():
        summary = json.loads(combined.read_text(encoding="utf-8"))
        basis = summary.get("successful_only_reference_scenario")
        if basis:
            g1_per, verifier_per = basis["G1_once_offpeak_per_case"], basis["A1_verifier_offpeak_per_case"]
            scenarios = {}
            for peak in [False, True]:
                factor = 2 if peak else 1
                shared = factor * g1_per
                allowance = factor * (g1_per + 3 * verifier_per)
                scenarios["peak" if peak else "offpeak"] = {
                    "G1_only_optimistic_upper_N_per_split": math.floor(240 / (2 * shared)),
                    "G1_plus_3_verifier_allowances_upper_N_per_split": math.floor(240 / (2 * allowance)),
                    "DEV400_plus_EVAL400_G1_only_yuan": 800 * shared,
                    "DEV400_plus_EVAL400_G1_plus_3_verifier_allowances_yuan": 800 * allowance}
            output["new16_complete_case_reference_scenarios"] = {
                **basis, "scenarios": scenarios,
                "omits": ["failed requests including unknown usage", "engineering smoke if charged to same 300 budget", *output["unmeasured"]],
                "formal_sample_size_frozen": False, "current_proxy_billing_verified": False}
        output["smoke_carried_and_new_peak_reference_upper_yuan"] = summary["carried_and_new_peak_reference_upper_yuan"]
    (HERE / "budget-reference.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

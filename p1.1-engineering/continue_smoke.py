"""Prospective timeout amendment; never-attempted profiles only, no case retries."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from time import perf_counter
from unittest.mock import patch

import run_cached_smoke as smoke

HERE, cost = smoke.HERE, smoke.cost
PARENT = HERE / "runs/cached-full-smoke-v1"
RUN = HERE / "runs/cached-full-smoke-continuation-v1"
OPTIONS = {**smoke.OPTIONS, "timeout_seconds": 180}


def request_upper(request):
    # Four UTF-8 bytes per Unicode character is a conservative input bound.
    return ((request["prompt_characters"] * 4 + 128) * 2 + OPTIONS["max_tokens"] * 8) / 1e6


def carry_budget(extra_runs=()):
    paths = {HERE / "runs/cached-full-smoke-v1/requests.jsonl"}
    paths.update(run / "requests.jsonl" for run in extra_runs)
    requests = [r for path in sorted(paths) for r in cost.base.read_rows(path)]
    for name in ("provider-cache-probe.json", "provider-long-context-probe.json"):
        requests += json.loads((HERE / name).read_text(encoding="utf-8"))["requests"]
    known, uncertain = 0.0, 0.0
    for request in requests:
        usage = request.get("usage") or {}
        if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
            uncertain += 2 * request_upper(request)
        else:
            known += (usage["prompt_tokens"] * 2 + usage["completion_tokens"] * 8) / 1e6
            uncertain += max(0, (request.get("transport_attempts") or 1) - 1) * request_upper(request)
    return {"known_peak_all_miss_reference_yuan": known,
            "unmetered_requests_and_prior_sdk_attempts_upper_yuan": uncertain,
            "carried_reference_upper_yuan": known + uncertain,
            "actual_usage_imputed": False}


class ContinuationBudget(smoke.ReferenceBudget):
    def __init__(self, carry, ceiling=60, stop_on_unknown_usage=True):
        super().__init__(ceiling=ceiling)
        self.spent = carry
        self.unmetered_upper = 0.0
        self.stop_on_unknown_usage = stop_on_unknown_usage
        self.had_unknown_usage = False

    def finish(self, amount, usage, attempts=1):
        with self.lock:
            self.reserved -= amount
            if not usage or usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
                self.had_unknown_usage = True
                self.unknown = self.stop_on_unknown_usage
                self.unmetered_upper += amount
                self.spent += amount
            else:
                extra_attempt_upper = amount / 2 * max(0, (attempts or 1) - 1)
                self.unmetered_upper += extra_attempt_upper
                self.spent += (usage["prompt_tokens"] * 2 + usage["completion_tokens"] * 8) / 1e6 + extra_attempt_upper


def bounded_observer(original, journal, budget):
    observed = cost.observer(original, journal)
    def bounded(*, timeout):
        api = observed(timeout=OPTIONS["timeout_seconds"])
        create = api.chat.completions.create
        def tracked(**request):
            amount = budget.reserve(request["messages"][0]["content"])
            request.update(max_tokens=OPTIONS["max_tokens"], stream_options=OPTIONS["stream_options"])
            owner = cost.base.LOCAL.record
            before = len(owner["requests"])
            try:
                stream = create(**request)
            except Exception:
                entry = owner["requests"][-1] if len(owner["requests"]) > before else {}
                budget.finish(amount, entry.get("usage"), entry.get("transport_attempts"))
                raise
            entry = owner["requests"][-1]
            entry.update(max_tokens=OPTIONS["max_tokens"], effective_read_timeout_seconds=180)
            def chunks():
                try:
                    yield from stream
                finally:
                    budget.finish(amount, entry.get("usage"), entry.get("transport_attempts"))
            return chunks()
        api.chat.completions.create = tracked
        return api
    return bounded


def main():
    global RUN, PARENT
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--total-cases", type=int, choices=[25, 50], default=50)
    parser.add_argument("--finish-unattempted-25", action="store_true")
    args = parser.parse_args()
    if args.finish_unattempted_25:
        assert args.total_cases == 25
        PARENT = HERE / "runs/cached-full-smoke25-continuation-v1"
    parent_summary = json.loads((PARENT / "summary.json").read_text(encoding="utf-8"))
    original = json.loads((PARENT / "manifest.json").read_text(encoding="utf-8"))
    assert original["official_price_reference_approved"]
    probe = json.loads((HERE / "provider-long-context-probe.json").read_text(encoding="utf-8"))
    assert probe["ok"] and all(r.get("usage") for r in probe["requests"])
    sources = [PARENT]
    if args.finish_unattempted_25:
        sources.append(HERE / "runs/cached-full-smoke-v1")
    attempted = {r["case_id"] for source in sources for r in cost.base.read_rows(source / "results.jsonl") if r["attempted"]}
    remaining = set(parent_summary["not_attempted_case_ids"])
    inputs = [r for r in cost.base.read_rows(HERE / "smoke.inputs.jsonl") if r["case_id"] in remaining]
    expected_plan = 25 if args.finish_unattempted_25 else 50
    assert len(inputs) == len(remaining) == expected_plan - len(attempted) and not attempted.intersection(remaining)
    ceiling = 60
    if args.total_cases == 25:
        selected = json.loads((HERE / "smoke25-cohort.json").read_text(encoding="utf-8"))
        assert set(selected["prior_attempted_case_ids"]).issubset(attempted)
        inputs = [r for r in inputs if r["case_id"] in set(selected["remaining_case_ids"])]
        assert len(inputs) == 25 - len(attempted)
        RUN = HERE / ("runs/cached-full-smoke25-bounded-v1" if args.finish_unattempted_25 else "runs/cached-full-smoke25-continuation-v1")
        ceiling = 30
    provider = cost.base.JsonDiagnosticKnowledgeProvider()
    model = cost.base.APICLINICRReasoner(knowledge=provider).model
    verifier_model = cost.APIRewardModelVerifier(knowledge=provider).model
    url, digest = cost.base.transport.get_base_url(), smoke.knowledge_hash(provider)
    assert (model, verifier_model, url, digest) == (original["reasoner_model"], original["verifier_model"], original["service_url"], original["knowledge_hash"])
    carry = carry_budget(extra_runs=[PARENT] if args.finish_unattempted_25 else [])
    manifest = {**original, "started_at": cost.base.now(), "parent_run": PARENT.name,
                "amendment": "180-second read timeout; no whole-case retry; scheduling/budget amendment documented before each remaining cohort",
                "case_ids": [r["case_id"] for r in inputs], "prior_attempted_case_ids_never_retried": sorted(attempted),
                "generation_options": OPTIONS, "max_workers": 4, "budget_carry": carry,
                "authorized_total_attempts": args.total_cases, "smoke_official_peak_reference_ceiling_yuan": ceiling,
                "unknown_usage_policy": "retain full reserved cost upper; continue other first attempts within reference ceiling" if args.finish_unattempted_25 else "stop new requests",
                "prior_started_at": original["started_at"], "formal_scoring": False}
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=True, indent=2))
        return 0
    assert not RUN.exists(), "Never overwrite first attempts"
    RUN.mkdir(parents=True)
    cost.write(RUN / "manifest.json", manifest)
    smoke.RUN, smoke.OPTIONS = RUN, OPTIONS
    budget, records = ContinuationBudget(carry["carried_reference_upper_yuan"], ceiling=ceiling, stop_on_unknown_usage=not args.finish_unattempted_25), []
    start = perf_counter()
    with (RUN / "requests.jsonl").open("w", encoding="utf-8") as journal, patch.object(cost.base.transport, "_bounded_client", bounded_observer(cost.base.transport._bounded_client, journal, budget)), ThreadPoolExecutor(max_workers=4) as pool, (RUN / "results.jsonl").open("w", encoding="utf-8") as output, (RUN / "A1.predictions.jsonl").open("w", encoding="utf-8") as predictions:
        futures = [pool.submit(smoke.run_case, row, digest, model, verifier_model, url) for row in inputs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if record["attempted"]:
                prediction = {"case_id": record["case_id"], "arm": "A1", "model_failure": record["model_failure"],
                              "run_failure": not record["ok"], "failure_kind": record.get("failure_kind"),
                              "report": record["result"] if record["ok"] else {"candidates": [], "final_diagnoses": []}}
                predictions.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                predictions.flush()
            print(json.dumps({"completed_remaining": len(records), "case_id": record["case_id"],
                              "attempted": record["attempted"], "ok": record["ok"],
                              "total_tokens": record["total_tokens"], "reference_upper_yuan": round(budget.spent, 4)}), flush=True)
    summary = cost.summarize([r for r in records if r["attempted"]], expected=len(inputs))
    summary.update(wall_elapsed_seconds=round(perf_counter() - start, 3),
                   not_attempted_case_ids=[r["case_id"] for r in records if not r["attempted"]],
                   carried_and_new_peak_reference_upper_yuan=budget.spent,
                   continuation_unmetered_reference_upper_yuan=budget.unmetered_upper,
                   reference_unknown_usage=budget.had_unknown_usage, current_platform_billing_verified=False)
    cost.write(RUN / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("attempted_cases", "successful_cases", "logical_api_calls", "cost_accounting")}), flush=True)
    return 0 if summary["measurement_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

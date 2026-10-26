"""Preregistered new-cohort smoke; full medical content, explicit billing assumptions."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter
from unittest.mock import patch
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1.0.8b"))
import run_full_cost as cost
from layers.layer3.diagnosis.g1_cache import G1NodeCache, CachedG1Reasoner, cache_context, knowledge_hash
from layers.layer3.diagnosis.schemas import CandidateGeneration, DiagnosticCandidate
from layers.layer3.llm.client import NODE_PROMPT_VERSION, ANCHOR_PROMPT_VERSION

RUN = HERE / "runs/cached-full-smoke-v1"
OPTIONS = {"max_tokens": 16000, "stream_options": {"include_usage": True},
           "timeout_seconds": 60, "sdk_max_retries": 1, "structured_retries_per_batch": 1}


class ReferenceBudgetStop(RuntimeError):
    pass


class ReferenceBudget:
    """Conservative official-peak reference ceiling, never represented as proxy billing."""
    def __init__(self, ceiling=60):
        self.ceiling, self.spent, self.reserved, self.unknown = ceiling, 0.0, 0.0, False
        self.lock = threading.Lock()

    def reserve(self, prompt):
        # UTF-8 bytes bound ordinary prompt tokens; add chat framing and both SDK attempts.
        amount = 2 * ((len(prompt.encode("utf-8")) + 128) * 2 + OPTIONS["max_tokens"] * 8) / 1e6
        with self.lock:
            if self.unknown or self.spent + self.reserved + amount > self.ceiling:
                raise ReferenceBudgetStop("Reference ceiling or unknown usage; no new request")
            self.reserved += amount
        return amount

    def finish(self, amount, usage):
        with self.lock:
            self.reserved -= amount
            if not usage or usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
                self.unknown = True
            else:
                # Charge all input at peak cache-miss price for the reference ceiling.
                self.spent += (usage["prompt_tokens"] * 2 + usage["completion_tokens"] * 8) / 1e6


def observed_bounded(original, journal, budget):
    observed = cost.observer(original, journal)
    def bounded(*, timeout):
        api = observed(timeout=timeout)
        create = api.chat.completions.create
        def tracked(**request):
            reservation = budget.reserve(request["messages"][0]["content"])
            request.update(max_tokens=OPTIONS["max_tokens"], stream_options=OPTIONS["stream_options"])
            owner = cost.base.LOCAL.record
            before = len(owner["requests"])
            try:
                stream = create(**request)
            except Exception:
                budget.finish(reservation, owner["requests"][-1].get("usage") if len(owner["requests"]) > before else None)
                raise
            entry = owner["requests"][-1]
            entry.update(max_tokens=OPTIONS["max_tokens"], stream_options=OPTIONS["stream_options"])
            def chunks():
                try:
                    yield from stream
                finally:
                    budget.finish(reservation, entry.get("usage"))
            return chunks()
        api.chat.completions.create = tracked
        return api
    return bounded


def run_case(row, digest, model, verifier_model, service_url):
    profile = row["profile"]
    model_profile = {**profile, cost.base.ALLOWED_EVIDENCE_REFS_KEY: sorted(cost.base.CLINICR._known_evidence_refs(profile))}
    record = {"case_id": row["case_id"], "started_at": cost.base.now(), "requests": [],
              "screen_denial_scope_events": [], "executed_in_this_run": True,
              "reasoner_model": model, "verifier_model": verifier_model}
    cost.base.LOCAL.record = record
    provider = cost.base.JsonDiagnosticKnowledgeProvider()
    delegate = cost.base.APICLINICRReasoner(model=model, knowledge=provider, criterion_batching="per_label", static_knowledge_prefix=True)
    prompt_version = ":".join([NODE_PROMPT_VERSION, cost.base.SCREENING_PROMPT_VERSION, ANCHOR_PROMPT_VERSION])
    context = cache_context(row["case_id"], model_profile, model=model, service_url=service_url,
                            knowledge_digest=digest, prompt_version=prompt_version, criterion_batching="per_label",
                            static_knowledge_prefix=True, generation_options=OPTIONS)
    cache = G1NodeCache(RUN / "g1-nodes")
    def provenance():
        return {"source_run": RUN.name, "fingerprints": sorted({q.get("system_fingerprint", "provider_not_exposed") for q in record["requests"]}),
                "response_models": sorted({q.get("response_model", "provider_not_exposed") for q in record["requests"]})}
    reasoner = CachedG1Reasoner(cache, context, knowledge=provider, delegate=delegate, mode="record", provenance=provenance)
    reward = cost.APIRewardModelVerifier(model=verifier_model, knowledge=provider)
    engine = cost.base.CLINICR(reasoner=reasoner, verifier=cost.IndependentVerifier(model=reward, knowledge=provider), knowledge=provider)
    start = perf_counter()
    try:
        record.update(ok=True, result=engine.run(profile), model_failure=False)
    except Exception as error:
        chain = cost.exception_chain(error)
        budget_stop = any(e["type"] == "ReferenceBudgetStop" for e in chain)
        infra_failure = any(e["type"] in {"G1CacheMiss", "G1CacheIntegrityError", "PermissionError", "FileNotFoundError"} for e in chain)
        record.update(ok=False, error_type=type(error).__name__, exception_chain=chain,
                      model_failure=not (budget_stop or infra_failure),
                      failure_kind="reference_budget_stop" if budget_stop else "cache_or_storage" if infra_failure else "model_or_contract")
    record.update(completed_at=cost.base.now(), elapsed_seconds=round(perf_counter() - start, 3),
                  reasoner_screen_denial_scope_events=reasoner.screen_denial_scope_events,
                  g1_cache_key=cache.path_for(context).name, g1_initial_cache_hit=reasoner.cache_hit)
    record["attempted"] = bool(record["requests"])
    record["cost_accounting"] = cost.accounting(record["requests"])
    for key, value in record["cost_accounting"].items():
        record[key] = value["total"]
    if cache.load(context) is None:
        record["g1_shared_validation"] = {"source_g1_unavailable": True, "new_model_calls": 0}
    else:
        try:
            generated = []
            for arm in ["A2", "A3"]:
                shared = CachedG1Reasoner(cache, context, knowledge=provider, mode="require")
                generation = shared.generate_candidates(model_profile)
                generation.candidates = cost.base.CLINICR._validate_generation(generation, cost.base.CLINICR._known_evidence_refs(profile))
                generated.append(generation.model_dump(mode="json"))
            assert generated[0] == generated[1]
            if record["ok"]:
                result = record["result"]
                expected = CandidateGeneration(considered_disorders=result["considered_disorders"],
                    candidates=[DiagnosticCandidate.model_validate({k: c[k] for k in DiagnosticCandidate.model_fields}) for c in result["candidates"]],
                    tree_trace=result["tree_trace"], missing_dimensions=result["candidate_missing_dimensions"], candidate_coverage=result["candidate_coverage"])
                assert generated[0] == expected.model_dump(mode="json")
            record["g1_shared_validation"] = {"A2_A3_identical": True, "A1_identical": record["ok"], "new_model_calls": 0}
        except Exception as error:
            record["g1_shared_validation"] = {"ok": False, "error_type": type(error).__name__, "new_model_calls": 0}
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-official-price-reference", action="store_true")
    args = parser.parse_args()
    inputs = cost.base.read_rows(HERE / "smoke.inputs.jsonl")
    assert len(inputs) == 50
    provider = cost.base.JsonDiagnosticKnowledgeProvider()
    model = cost.base.APICLINICRReasoner(knowledge=provider).model
    verifier_model = cost.APIRewardModelVerifier(knowledge=provider).model
    url = cost.base.transport.get_base_url()
    parsed = urlsplit(url)
    assert not (parsed.username or parsed.password or parsed.query or parsed.fragment)
    digest = knowledge_hash(provider)
    manifest = {"phase": "P1.1 engineering smoke", "prepared_at": cost.base.now(),
                "case_ids": [r["case_id"] for r in inputs], "seed": 20261015,
                "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cost.base.REPO, text=True).strip(),
                "reasoner_model": model, "verifier_model": verifier_model, "service_url": url, "knowledge_hash": digest,
                "node_prompt_version": NODE_PROMPT_VERSION, "screening_prompt_version": cost.base.SCREENING_PROMPT_VERSION,
                "anchor_prompt_version": ANCHOR_PROMPT_VERSION, "criterion_batching": "per_label", "static_knowledge_prefix": True,
                "screening": "zero", "anchor_guard_criteria": "full", "generation_options": OPTIONS,
                "max_workers": 4, "model_failure_policy": "explicit empty predictions, no whole-case rerun",
                "formal_budget_yuan": 300, "smoke_official_peak_reference_ceiling_yuan": 60,
                "current_platform_billing_verified": parsed.hostname == "api.deepseek.com",
                "official_price_reference_approved": args.allow_official_price_reference,
                "price_reference_url": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
                "actual_price_not_inferred_from_token_reference": True, "formal_scoring": False,
                "development_gold_metadata_used_for_sampling_only": True, "eval_opened": False}
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=True, indent=2))
        return 0
    if not manifest["current_platform_billing_verified"] and not args.allow_official_price_reference:
        raise ValueError("Platform unit prices unverified; explicit reference-price authorization required before paid smoke")
    assert not RUN.exists(), "Preserve all first attempts"
    RUN.mkdir(parents=True)
    manifest["started_at"] = cost.base.now()
    cost.write(RUN / "manifest.json", manifest)
    budget, records = ReferenceBudget(), []
    start = perf_counter()
    with (RUN / "requests.jsonl").open("w", encoding="utf-8") as journal, patch.object(cost.base.transport, "_bounded_client", observed_bounded(cost.base.transport._bounded_client, journal, budget)), ThreadPoolExecutor(max_workers=4) as pool, (RUN / "results.jsonl").open("w", encoding="utf-8") as output, (RUN / "A1.predictions.jsonl").open("w", encoding="utf-8") as predictions:
        futures = [pool.submit(run_case, row, digest, model, verifier_model, url) for row in inputs]
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
            print(json.dumps({"completed": len(records), "case_id": record["case_id"], "attempted": record["attempted"],
                              "ok": record["ok"], "total_tokens": record["total_tokens"], "reference_peak_spend": round(budget.spent, 4)}), flush=True)
    summary = cost.summarize([r for r in records if r["attempted"]], expected=50)
    summary.update(wall_elapsed_seconds=round(perf_counter() - start, 3),
                   not_attempted_case_ids=[r["case_id"] for r in records if not r["attempted"]],
                   reference_peak_all_input_miss_yuan=budget.spent, reference_unknown_usage=budget.unknown,
                   current_platform_billing_verified=manifest["current_platform_billing_verified"],
                   g1_shared_validations=[{"case_id": r["case_id"], **r["g1_shared_validation"]} for r in records])
    cost.write(RUN / "summary.json", summary)
    print(json.dumps({"attempted": summary["attempted_cases"], "successful": summary["successful_cases"],
                      "logical_api_calls": summary["logical_api_calls"], "cost_accounting": summary["cost_accounting"]}), flush=True)
    return 0 if summary["measurement_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

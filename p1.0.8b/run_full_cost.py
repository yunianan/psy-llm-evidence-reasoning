"""One full-engine DEV cost run; no scoring, prompt edits, cache or compact tuning."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import threading
from time import perf_counter
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1.0.7"))
from run_denial_audit import base, EventCapture
from layers.layer3.llm.client import ANCHOR_PROMPT_VERSION, APIRewardModelVerifier
from layers.layer3.diagnosis.verifier import IndependentVerifier
import openai
import pydantic

RUN = HERE / "runs/full-a1-v1"
STAGES = ("screening", "anchor", "guard", "criteria", "verifier")
WRITE_LOCK = threading.Lock()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def distribution(values):
    observed = [x for x in values if x is not None]
    return {"observed": len(observed), "sum": sum(observed),
            "p50": base.percentile(observed, .5), "p95": base.percentile(observed, .95),
            "min": min(observed) if observed else None, "max": max(observed) if observed else None}


def accounting(requests):
    answer = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(r.get("usage") or {}).get(key) for r in requests]
        missing = sum(x is None for x in values)
        known = sum(x for x in values if x is not None)
        answer[key] = {"total": None if missing else known, "known_sum": known,
                       "missing_request_usage": missing}
    return answer


def classify(prompt):
    if "当前节点 JSON：" in prompt:
        nodes, _ = json.JSONDecoder().raw_decode(prompt.split("当前节点 JSON：", 1)[1])
        stages = {n["stage"] for n in nodes}
        assert len(stages) == 1, "Do not silently assign mixed stages"
        stage = {"exclusion": "guard", "criterion": "criteria"}.get(next(iter(stages)), next(iter(stages)))
        return stage, [n["node_id"] for n in nodes]
    if prompt.startswith("你是 L3 的独立 Reward Model Verifier"):
        return "verifier", []
    raise ValueError("Unrecognized model operation; preserve cost attribution")


def observer(original_bounded, journal=None):
    def observed_client(*, timeout):
        api = original_bounded(timeout=timeout)
        original_create = api.chat.completions.create

        def create(**request):
            owner = base.LOCAL.record
            prompt = request["messages"][0]["content"]
            stage, ids = classify(prompt)
            item = {"case_id": owner["case_id"], "stage": stage, "node_ids": ids,
                    "is_structured_retry": any(r["stage"] == stage and r["node_ids"] == ids for r in owner["requests"]),
                    "prompt_characters": len(prompt), "usage": None,
                    "requested_model": request["model"], "temperature": request.get("temperature"),
                    "response_format": request.get("response_format"), "transport_attempts": None}
            owner["requests"].append(item)
            hooks = getattr(getattr(api, "_client", None), "event_hooks", None)
            if isinstance(hooks, dict) and "request" in hooks:
                item["transport_attempts"] = 0
                def count_attempt(_request):
                    item["transport_attempts"] += 1
                hooks["request"].append(count_attempt)
            start = perf_counter()

            def finish(parts):
                item.update(raw_output="".join(parts), elapsed_seconds=round(perf_counter() - start, 3))
                if journal is not None:
                    with WRITE_LOCK:
                        journal.write(json.dumps(item, ensure_ascii=False) + "\n")
                        journal.flush()

            try:
                stream = original_create(**request)
            except Exception as error:
                item.update(error_type=type(error).__name__, status_code=getattr(error, "status_code", None))
                finish([])
                raise

            def chunks():
                parts = []
                try:
                    for chunk in stream:
                        item["response_model"] = chunk.model
                        if getattr(chunk, "system_fingerprint", None):
                            item["system_fingerprint"] = chunk.system_fingerprint
                        if getattr(chunk, "usage", None):
                            item["usage"] = chunk.usage.model_dump(mode="json")
                        for choice in chunk.choices or []:
                            if choice.delta.content:
                                parts.append(choice.delta.content)
                            if choice.finish_reason:
                                item["finish_reason"] = choice.finish_reason
                        yield chunk
                except Exception as error:
                    item.update(error_type=type(error).__name__, status_code=getattr(error, "status_code", None))
                    raise
                finally:
                    finish(parts)
            return chunks()
        api.chat.completions.create = create
        return api
    return observed_client


def exception_chain(error):
    seen, chain = set(), []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append({"type": type(error).__name__, "status_code": getattr(error, "status_code", None)})
        error = error.__cause__ or error.__context__
    return chain


def run_case(row):
    record = {"case_id": row["case_id"], "started_at": base.now(), "requests": [],
              "screen_denial_scope_events": [], "executed_in_this_run": True}
    base.LOCAL.record = record
    provider = base.JsonDiagnosticKnowledgeProvider()
    reasoner = base.APICLINICRReasoner(knowledge=provider, screening_zero_knowledge=True,
                                     screening_compact_view=False, anchor_compact_view=False)
    reward = APIRewardModelVerifier(knowledge=provider)
    verifier = IndependentVerifier(model=reward, knowledge=provider)
    engine = base.CLINICR(reasoner=reasoner, verifier=verifier, knowledge=provider)
    record.update(reasoner_model=reasoner.model, verifier_model=reward.model)
    start = perf_counter()
    try:
        result = engine.run(row["profile"])
        record.update(ok=True, result=result)
    except Exception as error:
        record.update(ok=False, error_type=type(error).__name__, exception_chain=exception_chain(error))
    record.update(completed_at=base.now(), elapsed_seconds=round(perf_counter() - start, 3),
                  reasoner_screen_denial_scope_events=reasoner.screen_denial_scope_events)
    record["cost_accounting"] = accounting(record["requests"])
    for key, value in record["cost_accounting"].items():
        record[key] = value["total"]
    return record


def summarize(records, expected=20):
    requests = [r for row in records for r in row["requests"]]
    successful = [row for row in records if row["ok"]]
    initial = [r for r in requests if not r["is_structured_retry"]]
    repairs = [r for r in requests if r["is_structured_retry"]]
    summary = {"completed_at": base.now(), "expected_cases": expected, "attempted_cases": len(records),
               "successful_cases": len(successful), "failed_cases": [{"case_id": r["case_id"], "exception_chain": r.get("exception_chain")} for r in records if not r["ok"]],
               "measurement_status": "complete" if len(records) == expected and len(successful) == expected else "incomplete",
               "logical_api_calls": len(requests), "initial_batches": len(initial), "structured_retries": len(repairs),
               "structured_retry_rate": len(repairs) / len(initial) if initial else None,
               "cost_accounting": accounting(requests), "stages": {},
               "response_models": sorted({r.get("response_model", "not_returned") for r in requests}),
               "fingerprints": sorted({r.get("system_fingerprint", "not_returned") for r in requests}),
               "verifier_called_cases": sum(any(r["stage"] == "verifier" for r in row["requests"]) for row in records),
               "verifier_routing_on_successful": dict(Counter(row["result"]["verifier_routing"] for row in successful))}
    attempts = [r["transport_attempts"] for r in requests]
    summary["transport_attempts"] = sum(attempts) if all(x is not None for x in attempts) else None
    summary["request_usage_complete"] = all(v["missing_request_usage"] == 0 for v in summary["cost_accounting"].values())
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "elapsed_seconds"):
        summary[key + "_per_attempted_case"] = distribution([row[key] for row in records])
        summary[key + "_per_successful_case"] = distribution([row[key] for row in successful])
    summary["logical_calls_per_case"] = distribution([len(row["requests"]) for row in records])
    summary["candidate_count_on_successful"] = distribution([len(row["result"]["candidates"]) for row in successful])
    for stage in STAGES:
        reqs = [r for r in requests if r["stage"] == stage]
        stage_initial = sum(not r["is_structured_retry"] for r in reqs)
        stage_repair = sum(r["is_structured_retry"] for r in reqs)
        info = {"reached_cases": sum(any(r["stage"] == stage for r in row["requests"]) for row in records),
                "logical_calls": len(reqs), "initial_batches": stage_initial, "structured_retries": stage_repair,
                "structured_retry_rate": stage_repair / stage_initial if stage_initial else None,
                "cost_accounting": accounting(reqs)}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            info[key + "_per_case"] = distribution([accounting([r for r in row["requests"] if r["stage"] == stage])[key]["total"] for row in records])
            info[key + "_per_request"] = distribution([(r.get("usage") or {}).get(key) for r in reqs])
        info["request_elapsed_seconds_per_case"] = distribution([sum(r["elapsed_seconds"] for r in row["requests"] if r["stage"] == stage) for row in records])
        summary["stages"][stage] = info
    screen_nodes = {(row["case_id"], node) for row in records for r in row["requests"] if r["stage"] == "screening" for node in r["node_ids"]}
    events = [dict(event, case_id=row["case_id"]) for row in records for event in row["reasoner_screen_denial_scope_events"]]
    downgraded = {(e["case_id"], e["node_id"]) for e in events}
    summary["screen_denial_scope"] = {"screen_unique_nodes_requested": len(screen_nodes), "downgrade_log_events": len(events),
                                     "unique_downgraded_nodes": len(downgraded), "unique_node_downgrade_ratio": len(downgraded) / len(screen_nodes) if screen_nodes else None,
                                     "events": events}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    inputs = base.read_rows(HERE.parent / "p1.0.6/inputs.jsonl")
    assert len(inputs) == len({r["case_id"] for r in inputs}) == 20
    assert all(r["profile"]["key_narratives"] and base.CLINICR._known_evidence_refs(r["profile"]) for r in inputs)
    graph = base.load_candidate_graph()
    manifest = {"phase": "P1.0.8b", "prepared_at": base.now(), "case_ids": [r["case_id"] for r in inputs],
                "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.REPO, text=True).strip(),
                "python": platform.python_version(), "openai": openai.__version__, "pydantic": pydantic.__version__,
                "tree_version": graph["version"], "screening_prompt_version": base.SCREENING_PROMPT_VERSION,
                "anchor_prompt_version": ANCHOR_PROMPT_VERSION, "screening_mode": "zero", "anchor_guard_criteria_mode": "full",
                "model_alias": base.transport.get_model(), "model_official_mapping_at_last_check": "DeepSeek-V4.1-Flash-1017",
                "models_returned_at_last_check": "DeepSeek-V4.1-Flash", "mapping_check_date": "2026-10-22",
                "expected_fingerprint": "e72d3c7f663d3e1383b00b3884cb4501", "max_workers": 4, "selection_seed": 20261015,
                "temperature": 0, "response_format": {"type": "json_object"}, "timeout_seconds": 60,
                "structured_retry_budget_per_batch": 1, "sdk_max_retries": 1, "cached_model_outputs": False,
                "default_provider": type(base.get_default_knowledge_provider()).__name__, "gold_read": False,
                "engine_entry": "CLINICR.run; offline JSON provider; production API input contract bypassed",
                "formal_scoring": False, "compact_tuning": False}
    assert type(base.get_default_knowledge_provider()).__name__ == "UnconnectedDiagnosticKnowledgeProvider"
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    assert not RUN.exists(), "Never overwrite an existing attempt"
    RUN.mkdir(parents=True)
    manifest["started_at"] = base.now()
    write(RUN / "manifest.json", manifest)
    handler = EventCapture()
    logger = logging.getLogger("layers.layer3.diagnosis.screening")
    logger.addHandler(handler)
    records = []
    started = perf_counter()
    print(json.dumps({"starting": "P1.0.8b full A1 cost", "cases": 20}), flush=True)
    try:
        with (RUN / "requests.jsonl").open("w", encoding="utf-8") as journal, patch.object(base.transport, "_bounded_client", observer(base.transport._bounded_client, journal)), ThreadPoolExecutor(max_workers=4) as pool, (RUN / "results.jsonl").open("w", encoding="utf-8") as output:
            futures = [pool.submit(run_case, row) for row in inputs]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(json.dumps({"case_id": record["case_id"], "completed": len(records), "ok": record["ok"],
                                  "logical_calls": len(record["requests"]), "total_tokens": record["total_tokens"],
                                  "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
    finally:
        logger.removeHandler(handler)
    summary = summarize(records)
    summary["wall_elapsed_seconds"] = round(perf_counter() - started, 3)
    write(RUN / "summary.json", summary)
    print(json.dumps({"measurement_status": summary["measurement_status"], "successful_cases": summary["successful_cases"],
                      "logical_api_calls": summary["logical_api_calls"], "cost_accounting": summary["cost_accounting"]}), flush=True)
    return 0 if summary["measurement_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

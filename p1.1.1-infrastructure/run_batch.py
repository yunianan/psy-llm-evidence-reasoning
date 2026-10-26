"""Separate engineering batches with durable per-node resume; no formal scoring."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from collections import Counter
import json
import random
import subprocess
import threading
from unittest.mock import patch

from common import HERE, REPO, POLICY, cost, node_client, read, run_case, context
from tools.l3_engineering_runtime import ReferenceBudget, ReliableStreaming, recover_ledger, now, reasoning_observation, NodeCheckpoint


def reference_carry(mode):
    probe = json.loads((HERE / "runs/thinking-probe-v1/summary.json").read_text(encoding="utf-8"))["budget"]["reference_spent_upper_yuan"]
    if mode == "repair":
        audit = HERE / "runs/thinking-audit-v1"
        if audit.exists():
            assert (audit / "summary.json").exists() or (audit / "stopped-summary.json").exists(), "Finish or explicitly stop the audit before repairing nine; no simultaneous budgets"
            probe += recover_ledger(audit / "requests.jsonl")["spent"]
    return probe


def summarize(records, mode, budget, carry, *, ledger_path=None):
    if ledger_path is not None:
        events = {r["attempt_id"]: r for r in read(ledger_path)}
        requests = list(events.values())
    else:
        requests = [r for record in records for r in record["requests"]]
    successful = [r for r in records if r["ok"]]
    repairs = [r for r in requests if r["attempt"] == 1 and r["is_structured_retry"]]
    logical = [r for r in requests if r["attempt"] == 1]
    snapshot = budget.snapshot()
    result = {"completed_at": now(), "mode": mode, "completed_case_arms": len(records),
        "successful_case_arms": len(successful), "infrastructure_successful_case_arms": sum(r["infrastructure_success"] for r in records),
        "failure_kinds": dict(Counter(r["failure_kind"] for r in records if not r["ok"])),
        "http_attempts": len(requests), "logical_requests": len(logical), "structured_repairs": len(repairs),
        "transport_attempt_errors": sum(bool(r.get("error_type")) for r in requests),
        "transport_error_categories": dict(Counter(r.get("error_category", "historical_unknown") for r in requests if r.get("error_type"))),
        "requests_with_usage": sum(r.get("usage") is not None for r in requests), "cost_accounting": cost.accounting(requests),
        "latency_seconds_successful": cost.distribution([r["elapsed_seconds"] for r in successful]),
        "per_request_latency_seconds": cost.distribution([r.get("elapsed_seconds") for r in requests]),
        "unfinished_http_attempts": sum(r.get("event") == "attempt_started" for r in requests),
        "reused_nodes": sum(r.get("checkpoint_stats", {}).get("reused_nodes", 0) for r in records),
        "new_nodes": sum(r.get("checkpoint_stats", {}).get("new_nodes", 0) for r in records),
        "g1_shared_replay_cases": sum(r["g1_shared_validation"].get("identical", False) for r in records),
        "g1_shared_replay_failures": [r["case_id"] for r in records if r["g1_shared_validation"].get("identical") is False],
        "g1_replay_new_model_calls": 0, "budget": snapshot,
        "phase_reference_upper_including_prior_batches": carry + snapshot["reference_spent_upper_yuan"],
        "actual_platform_invoice": None, "formal_scoring": False, "eval_opened": False}
    result["reasoning_observations"] = dict(Counter(reasoning_observation(r)["basis"] for r in requests))
    if mode == "audit":
        paired = {}
        for row in records:
            paired.setdefault(row["case_id"], {})[row["arm"]] = row
        comparisons = []
        for case_id, arms in sorted(paired.items()):
            on, off = arms.get("thinking_on"), arms.get("thinking_off")
            if not on or not off or not on["ok"] or not off["ok"]:
                comparisons.append({"case_id": case_id, "complete": False, "passed": False})
                continue
            left, right = on["result"], off["result"]
            lm, rm = ({r["node_id"]: r for r in x["tree_trace"]} for x in [left, right])
            common = set(lm) & set(rm)
            diffs = [{"node_id": key, "thinking_on": lm[key]["status"], "thinking_off": rm[key]["status"]}
                     for key in sorted(common) if lm[key]["status"] != rm[key]["status"]]
            coverage = left["candidate_coverage"] == right["candidate_coverage"]
            labels = {c["disorder"] for c in left["candidates"]} == {c["disorder"] for c in right["candidates"]}
            passed = set(lm) == set(rm) and not diffs and coverage and labels
            comparisons.append({"case_id": case_id, "complete": True, "passed": passed,
                "status_differences": diffs, "only_on_node_ids": sorted(set(lm) - set(rm)), "only_off_node_ids": sorted(set(rm) - set(lm)),
                "coverage_identical": coverage, "candidate_labels_identical": labels,
                "symptom_id_differences": [key for key in sorted(common) if set(lm[key]["symptom_ids"]) != set(rm[key]["symptom_ids"])]})
        result["comparisons"] = comparisons
        result["passed_cases"] = sum(c["passed"] for c in comparisons)
        result["default_non_thinking_allowed"] = len(comparisons) == 20 and result["passed_cases"] == 20
        result["arms"] = {arm: {"successful": sum(r["ok"] for r in records if r["arm"] == arm),
            "cost_accounting": cost.accounting([q for r in records if r["arm"] == arm for q in r["requests"]]),
            "elapsed_seconds": cost.distribution([r["elapsed_seconds"] for r in successful if r["arm"] == arm])}
            for arm in ["thinking_on", "thinking_off"]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "repair"], required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    mode = args.mode
    verification = json.loads((HERE / "runs/thinking-probe-v1/verification.json").read_text(encoding="utf-8"))
    if mode == "audit":
        assert verification["thinking_control_observed"], "No pretend non-thinking arm if the source cannot disable thinking"
    rows = read(HERE / ("audit20.inputs.jsonl" if mode == "audit" else "repair9.inputs.jsonl"))
    assert len(rows) == (20 if mode == "audit" else 9)
    if mode == "repair":
        imported = json.loads((HERE / "legacy-import.json").read_text(encoding="utf-8"))
        expected = {r["case_id"]: r["checkpoint_nodes"] for r in imported["cases"]}
        provider = cost.base.JsonDiagnosticKnowledgeProvider()
        for row in rows:
            checkpoint = NodeCheckpoint(HERE / "repair-checkpoints", context(row, provider, None))
            assert sum(n["origin"].startswith("validated_legacy:") for n in checkpoint.nodes.values()) == expected[row["case_id"]], "Imported model/profile/knowledge context must remain identical; never silently rerun completed nodes"
    run = HERE / "runs" / ("thinking-audit-v1" if mode == "audit" else "repair9-v1")
    carry = reference_carry(mode)
    model = cost.base.APICLINICRReasoner(knowledge=cost.base.JsonDiagnosticKnowledgeProvider()).model
    manifest = {"started_at": now(), "mode": mode, "case_ids": [r["case_id"] for r in rows], "policy": POLICY,
                "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "model": model, "base_url": cost.base.transport.get_base_url(), "prior_phase_reference_upper_yuan": carry,
                "thinking_policy": "explicit true/false, separate caches" if mode == "audit" else "provider default, same semantics as imported nodes",
                "audit_scope": "complete G1, no Verifier" if mode == "audit" else "complete offline CLINICR A1",
                "whole_case_rerun_of_completed_nodes": False, "phase_reference_limit": 100,
                "actual_platform_billing_verified": False, "formal_scoring": False, "eval_opened": False}
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=True, indent=2))
        return
    if args.resume:
        assert run.exists()
        original = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        assert all(original[key] == manifest[key] for key in ["mode", "case_ids", "policy", "model", "base_url", "baseline_commit"])
    else:
        assert not run.exists(), "Use --resume for an interrupted new engineering batch; never overwrite first attempts"
        run.mkdir(parents=True)
        cost.write(run / "manifest.json", manifest)
    records = read(run / "results.jsonl") if (run / "results.jsonl").exists() else []
    completed = {(r["case_id"], r["arm"]) for r in records}
    ledger = recover_ledger(run / "requests.jsonl")
    budget = ReferenceBudget(batch_limit=100 - carry, case_limit=10)
    budget.spent, budget.unknown_upper, budget.by_case = ledger["spent"], ledger["unknown_upper"], ledger["by_case"]
    lock, rng = threading.Lock(), random.Random(20261015)
    orders = {}
    for row in rows:
        arms = [("thinking_on", True), ("thinking_off", False)] if mode == "audit" else [("repair_provider_default", None)]
        rng.shuffle(arms)
        orders[row["case_id"]] = arms
    cost.write(run / "arm-orders.json", orders)
    with (run / "requests.jsonl").open("a", encoding="utf-8") as journal, (run / "results.jsonl").open("a", encoding="utf-8") as output:
        transport = ReliableStreaming(api_key=cost.base.transport.get_api_key(), base_url=cost.base.transport.get_base_url(), journal=journal, budget=budget)
        def task(row):
            for arm, thinking in orders[row["case_id"]]:
                if (row["case_id"], arm) in completed:
                    continue
                try:
                    record = run_case(row, arm=arm, thinking=thinking, run=run, transport=transport,
                        checkpoint_root=HERE / ("audit-checkpoints" if mode == "audit" else "repair-checkpoints"), g1_only=mode == "audit")
                except Exception as error:
                    record = {"case_id": row["case_id"], "arm": arm, "ok": False, "requests": [],
                              "failure_kind": "runner_initialization_or_validation", "infrastructure_success": False,
                              "exception_chain": cost.exception_chain(error), "elapsed_seconds": 0,
                              "g1_shared_validation": {"source_g1_unavailable": True}}
                with lock:
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(json.dumps({"completed_arms": len(records), "case_id": row["case_id"], "arm": arm,
                                      "ok": record["ok"], "failure_kind": record["failure_kind"],
                                      "reference_upper_yuan": round(carry + budget.spent, 4)}), flush=True)
        with patch.object(node_client, "ask_llm_stream", transport.ask), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(task, row) for row in rows]
            for future in as_completed(futures):
                future.result()
    summary = summarize(records, mode, budget, carry, ledger_path=run / "requests.jsonl")
    cost.write(run / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["completed_case_arms", "successful_case_arms", "failure_kinds", "http_attempts", "phase_reference_upper_including_prior_batches"]}), flush=True)


if __name__ == "__main__":
    main()

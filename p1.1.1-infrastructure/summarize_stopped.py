"""Summarize existing files after the user's stop request. No model calls/tests."""
import json

from common import HERE, cost, read
from run_batch import summarize, reference_carry
from analyze import analyze
from tools.l3_engineering_runtime import ReferenceBudget, recover_ledger, now


run = HERE / "runs/thinking-audit-v1"
records = read(run / "results.jsonl")
latest = {r["attempt_id"]: r for r in read(run / "requests.jsonl")}
ledger = recover_ledger(run / "requests.jsonl")
carry = reference_carry("audit")
budget = ReferenceBudget(batch_limit=100 - carry, case_limit=10)
budget.spent, budget.unknown_upper, budget.by_case = ledger["spent"], ledger["unknown_upper"], ledger["by_case"]
result = summarize(records, "audit", budget, carry, ledger_path=run / "requests.jsonl")
orders = json.loads((run / "arm-orders.json").read_text(encoding="utf-8"))
planned = {(case_id, arm) for case_id, arms in orders.items() for arm, _ in arms}
completed = {(r["case_id"], r["arm"]) for r in records}
interrupted = {(r["case_id"], r["arm"]) for r in latest.values() if r["event"] == "attempt_started"}
unstarted = planned - completed - interrupted
result.update(status="stopped_by_user", stopped_at=now(), experiment_completed=False,
    planned_cases=20, planned_case_arms=40, finalized_case_arms=len(completed),
    interrupted_case_arms=[{"case_id": c, "arm": a} for c, a in sorted(interrupted)],
    unstarted_case_arms=[{"case_id": c, "arm": a} for c, a in sorted(unstarted)],
    complete_successful_pairs=sum(c["complete"] for c in result["comparisons"]),
    no_additional_tests_or_model_calls_after_stop=True, repair9_started=False,
    interrupted_attempt_cost_policy="unknown usage retained at reservation upper bound; not zero and not invoice",
    original_results_overwritten=False)
cost.write(run / "stopped-summary.json", result)
analysis = {"execution_status": "stopped_by_user", "thinking-audit-v1": analyze(run)}
cost.write(HERE / "resource-analysis.json", analysis)
print(json.dumps({key: result[key] for key in ["finalized_case_arms", "successful_case_arms", "failure_kinds", "complete_successful_pairs", "passed_cases", "g1_shared_replay_cases", "http_attempts", "unfinished_http_attempts", "phase_reference_upper_including_prior_batches"]}, ensure_ascii=False, indent=2))
print(json.dumps({"interrupted_arms": len(interrupted), "unstarted_arms": len(unstarted), "paired_state": analysis["thinking-audit-v1"].get("paired_state_comparisons")}, ensure_ascii=False, indent=2))

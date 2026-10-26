"""Engineering-only comparison; never replace or formally rescore first attempts."""
from collections import Counter
import json

from common import HERE, cost, read, context
from tools.l3_engineering_runtime import NodeCheckpoint, recover_ledger, reasoning_observation


def main():
    previous = HERE.parent / "p1.1-engineering"
    audit_dir, repair_dir = HERE / "runs/thinking-audit-v1", HERE / "runs/repair9-v1"
    audit_path = audit_dir / "summary.json"
    if not audit_path.exists():
        audit_path = audit_dir / "stopped-summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    repaired = read(repair_dir / "results.jsonl")
    original = [r for name in ["cached-full-smoke-v1", "cached-full-smoke25-continuation-v1", "cached-full-smoke25-bounded-v1"]
                for r in read(previous / "runs" / name / "results.jsonl") if r["attempted"]]
    failed_ids = {r["case_id"] for r in original if not r["ok"]}
    assert len(repaired) == len({r["case_id"] for r in repaired}) == 9 and {r["case_id"] for r in repaired} == failed_ids
    combined = [{**r, "engineering_origin": "first_attempt_success"} for r in original if r["ok"]]
    combined += [{**r, "engineering_origin": "repair9-v1"} for r in repaired]
    assert len(combined) == len({r["case_id"] for r in combined}) == 25
    provider = cost.base.JsonDiagnosticKnowledgeProvider()
    profiles = {r["case_id"]: r for r in read(HERE / "repair9.inputs.jsonl")}
    duplicates = []
    for row in repaired:
        checkpoint = NodeCheckpoint(HERE / "repair-checkpoints", context(profiles[row["case_id"]], provider, None))
        legacy_nodes = {key for key, value in checkpoint.nodes.items() if value["origin"].startswith("validated_legacy:")}
        for request in row["requests"]:
            duplicated = set(request["node_ids"]) & legacy_nodes
            if duplicated:
                duplicates.append({"case_id": row["case_id"], "nodes": sorted(duplicated)})
    assert not duplicates, "Completed legacy nodes must never be regenerated"
    available = sum(r["g1_shared_validation"].get("A2_A3_identical", r["g1_shared_validation"].get("identical", False)) for r in combined)
    replay_failures = [r["case_id"] for r in combined if r["g1_shared_validation"].get("identical") is False or r["g1_shared_validation"].get("A2_A3_identical") is False]
    infra = 16 + sum(r["infrastructure_success"] for r in repaired)
    complete = sum(r["ok"] for r in combined)
    result = {"first_attempts": {"cases": 25, "complete": 16, "ReadTimeout": 4, "budget_abort": 5},
        "repair_batch": {"cases": 9, "complete": sum(r["ok"] for r in repaired),
                         "failure_kinds": dict(Counter(r["failure_kind"] for r in repaired if not r["ok"]))},
        "combined_engineering_25": {"infrastructure_successful": infra, "complete_outputs": complete,
            "infrastructure_success_rate": infra / 25, "complete_output_rate": complete / 25,
            "g1_available_cases": available, "identical_readonly_replays": available * 2, "replay_new_model_calls": 0,
            "replay_failures": replay_failures,
            "passed": infra == 25 and complete >= 24 and available >= complete and not replay_failures},
        "legacy_completed_node_duplicate_model_calls": duplicates,
        "thinking_audit": {"cases_planned": 20, "complete_successful_pairs": sum(c["complete"] for c in audit["comparisons"]),
                           "passed_cases": audit["passed_cases"], "status": audit.get("status", "completed"),
                           "negative_result": True, "default_non_thinking_allowed": audit["default_non_thinking_allowed"]},
        "formal_scoring": False, "original_results_overwritten": False, "actual_platform_invoice": None, "eval_opened": False}
    requests = []
    phase_reference = 0
    for folder in [HERE / "runs/thinking-probe-v1", audit_dir, repair_dir]:
        ledger_path = folder / "requests.jsonl"
        raw = read(ledger_path)
        if raw and "event" in raw[0]:
            # Preserve interrupted attempts with null usage in the denominator.
            requests += list({r["attempt_id"]: r for r in raw}.values())
            phase_reference += recover_ledger(ledger_path)["spent"]
        else:  # The earlier synthetic probe predates the start/finish ledger extension.
            requests += raw
            phase_reference += json.loads((folder / "summary.json").read_text(encoding="utf-8"))["budget"]["reference_spent_upper_yuan"]
    result["new_phase_cost_accounting"] = cost.accounting(requests)
    result["new_phase_reference_upper_yuan"] = phase_reference
    result["observed_reasoning_basis"] = dict(Counter(reasoning_observation(r)["basis"] for r in requests))
    cost.write(HERE / "final-summary.json", result)
    predictions = [{"case_id": r["case_id"], "batch_origin": r["engineering_origin"], "engineering_only": True,
                    "run_failure": not r["ok"], "model_failure": not r["ok"], "failure_kind": r.get("failure_kind"),
                    "contract_failure": r.get("failure_kind") == "contract_failure",
                    "report": r["result"] if r["ok"] else {"candidates": [], "final_diagnoses": []}}
                   for r in sorted(combined, key=lambda r: r["case_id"])]
    (HERE / "engineering-repaired25.predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

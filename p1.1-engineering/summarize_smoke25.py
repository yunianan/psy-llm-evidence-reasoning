"""Combine distinct first attempts; retain transport failures and unknown usage."""
from collections import Counter
from datetime import datetime
import json

from continue_smoke import HERE, cost
from budget_reference import partial_reference, price, tokens
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from layers.layer3.diagnosis.screening import validate_screen_denial_scope


def screen_declaration_audit(records):
    requested, responded, downgrade_nodes, declarations = set(), set(), set(), Counter()
    invalid_batches = 0
    for row in records:
        for request in row["requests"]:
            if request["stage"] != "screening":
                continue
            expected = set(request["node_ids"])
            requested.update((row["case_id"], key) for key in expected)
            try:
                answers = [TreeNodeJudgment.model_validate(n) for n in json.loads(request["raw_output"])["nodes"]]
                assert len(answers) == len(expected) and {a.node_id for a in answers} == expected
            except (ValueError, KeyError, TypeError, AssertionError):
                invalid_batches += 1
                continue
            for answer in answers:
                key = (row["case_id"], answer.node_id)
                responded.add(key)
                declarations[f"{answer.status}/{answer.denial_scope}"] += 1
                _, event = validate_screen_denial_scope(answer, stage="screening")
                if event:
                    downgrade_nodes.add(key)
    return {"method": "offline schema-valid raw declaration audit, not a clinical evidence audit or replacement of original logs",
            "requested_unique_screen_nodes": len(requested), "schema_valid_responded_unique_nodes": len(responded),
            "invalid_or_missing_raw_batches": invalid_batches, "declaration_counts_including_retries": dict(declarations),
            "raw_downgrade_candidate_unique_nodes": len(downgrade_nodes),
            "raw_downgrade_candidate_ratio": len(downgrade_nodes) / len(responded) if responded else None,
            "failed_run_original_wrapper_logs_may_be_incomplete": True}


def main():
    plan = json.loads((HERE / "smoke25-cohort.json").read_text(encoding="utf-8"))
    runs = [HERE / "runs/cached-full-smoke-v1", HERE / "runs/cached-full-smoke25-continuation-v1"]
    final_run = HERE / "runs/cached-full-smoke25-bounded-v1"
    if final_run.exists():
        runs.append(final_run)
    assert all((run / "summary.json").exists() for run in runs), "Do not summarize in-flight cases as final"
    records, predictions = [], []
    for run in runs:
        records += [r for r in cost.base.read_rows(run / "results.jsonl") if r["attempted"]]
        predictions += cost.base.read_rows(run / "A1.predictions.jsonl")
    ids = [r["case_id"] for r in records]
    assert len(ids) == len(set(ids)) and set(ids).issubset(plan["case_ids"])
    assert {p["case_id"] for p in predictions} == set(ids)
    summary = cost.summarize(records, expected=25)
    successful = [r for r in records if r["ok"]]
    requests = [r for r in records for r in r["requests"]]
    reusable = [r for r in records if r["g1_shared_validation"].get("A2_A3_identical")]
    failed = [r for r in records if not r["ok"]]
    summary.update(execution_status="all_25_first_attempts_recorded" if len(records) == 25 else "stopped_before_25",
                   authorized_case_ids=plan["case_ids"], not_attempted_case_ids=sorted(set(plan["case_ids"]) - set(ids)),
                   whole_case_retries=0, formal_scoring=False, eval_opened=False,
                   failure_kinds=dict(Counter(r.get("failure_kind") for r in failed)),
                   model_failure_cases=sum(r["model_failure"] for r in records),
                   model_failure_rate_over_attempted=sum(r["model_failure"] for r in records) / len(records) if records else None,
                   successful_G1_cached_cases=len(reusable), readonly_G1_replays=len(reusable) * 2,
                   replay_new_model_calls=sum(r["g1_shared_validation"]["new_model_calls"] for r in records),
                   A1_G1_exact_shared_cases=sum(r["g1_shared_validation"].get("A1_identical", False) for r in records),
                   pricing=partial_reference(requests),
                   run_parameters_differ="60-second first four vs preregistered 180-second remaining 21",
                   complete_cost_not_imputed=True)
    summary["screen_raw_declaration_audit"] = screen_declaration_audit(records)
    clock_gaps = []
    for row in records:
        timestamp_span = (datetime.fromisoformat(row["completed_at"]) - datetime.fromisoformat(row["started_at"])).total_seconds()
        if abs(timestamp_span - row["elapsed_seconds"]) > 60:
            clock_gaps.append({"case_id": row["case_id"], "timestamp_span_seconds": timestamp_span,
                               "perf_counter_elapsed_seconds": row["elapsed_seconds"]})
    summary["timestamp_and_monotonic_clock_discrepancies"] = clock_gaps
    summary["latency_clock"] = "recorded perf_counter elapsed; timestamp gaps are not attributed to model latency"
    if successful:
        complete_requests = [r for row in successful for r in row["requests"]]
        g1 = tokens([r for r in complete_requests if r["stage"] != "verifier"])
        verifier = tokens([r for r in complete_requests if r["stage"] == "verifier"])
        summary["successful_only_reference_scenario"] = {
            "case_count": len(successful), "G1_once_offpeak_per_case": price(g1) / len(successful),
            "A1_verifier_offpeak_per_case": price(verifier) / len(successful),
            "selection": "complete cases only; excludes all failed consumption; not a formal sample-size budget",
            "cache_hits_observed_not_assumed": True}
    continuation = json.loads((runs[-1] / "summary.json").read_text(encoding="utf-8"))
    summary["carried_and_new_peak_reference_upper_yuan"] = continuation["carried_and_new_peak_reference_upper_yuan"]
    cost.write(HERE / "smoke25-summary.json", summary)
    (HERE / "smoke25.A1.predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(predictions, key=lambda x: x["case_id"])), encoding="utf-8")
    keys = ("execution_status", "attempted_cases", "successful_cases", "failure_kinds", "model_failure_cases",
            "logical_api_calls", "structured_retry_rate", "cost_accounting", "readonly_G1_replays",
            "replay_new_model_calls", "pricing", "carried_and_new_peak_reference_upper_yuan")
    print(json.dumps({k: summary[k] for k in keys}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

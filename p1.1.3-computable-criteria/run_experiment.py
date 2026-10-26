"""New-holdout engineering comparison; never installs a production default."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1.1.1-infrastructure"))
import common as infra
from extract_success_costs import extract
from tools.l3_computable_criteria import VERSION
from tools.l3_hybrid_criteria import ComputableCriteriaReasoner, BATCHING
from tools.l3_engineering_runtime import NodeCheckpoint, attach_checkpoints, ReferenceBudget, ReliableStreaming, recover_ledger, now
from layers.layer3.diagnosis.g1_cache import CachedG1Reasoner, G1NodeCache
from layers.layer3.diagnosis.schemas import TreeNodeJudgment, DiagnosticCandidate

RUN = HERE / "runs/full-vs-computed-v1"


def hybrid_context(row, provider, *, full=False):
    value = infra.context(row, provider, True, transport_policy=full)
    value["prompt_version"] += ":" + VERSION
    value["criterion_batching"] = BATCHING
    value["criteria_execution"] = VERSION
    return value


def hybrid_case(row, transport):
    record = {"case_id": row["case_id"], "arm": "computed", "started_at": now(), "requests": []}
    transport.bind(row["case_id"], record["requests"], thinking=True, arm="computed")
    provider = infra.cost.base.JsonDiagnosticKnowledgeProvider()
    baseline = NodeCheckpoint(HERE / "checkpoints/full", infra.context(row, provider, True))
    checkpoint = NodeCheckpoint(HERE / "checkpoints/computed", hybrid_context(row, provider))
    prefix = [value for value in baseline.nodes.values() if value["definition"]["stage"] != "criterion"]
    if prefix:
        checkpoint.save([value["definition"] for value in prefix],
            [TreeNodeJudgment.model_validate(value["judgment"]) for value in prefix], origin="paired_full_validated_prefix")
    delegate = ComputableCriteriaReasoner(knowledge=provider, criterion_batching="per_label", static_knowledge_prefix=True,
                                         shared_prefix={}, batch_observer=transport.tags)
    stats = attach_checkpoints(delegate, checkpoint, transport)
    context = hybrid_context(row, provider, full=True)
    cache = G1NodeCache(RUN / "g1-nodes")
    reasoner = CachedG1Reasoner(cache, context, knowledge=provider, delegate=delegate, mode="record",
        provenance=lambda: {"strategy": VERSION, "shared_prefix_nodes": len(prefix), "source_run": RUN.name})
    reward = infra.cost.APIRewardModelVerifier(knowledge=provider)
    engine = infra.cost.base.CLINICR(reasoner=reasoner,
        verifier=infra.cost.IndependentVerifier(model=reward, knowledge=provider), knowledge=provider)
    started = perf_counter()
    try:
        record.update(result=engine.run(row["profile"]), ok=True, failure_kind=None, infrastructure_success=True)
    except Exception as error:
        chain = infra.cost.exception_chain(error)
        types = {value["type"] for value in chain}
        last = (record["requests"] or [{}])[-1]
        kind = ("shared_prefix_unavailable" if "SharedPrefixUnavailable" in types else
                "budget" if types & {"CaseBudgetExceeded", "BatchBudgetExceeded"} else
                "transport_or_http" if last.get("error_type") else
                "contract_failure" if types & {"ValueError", "ValidationError"} else "model_or_runner_failure")
        record.update(ok=False, failure_kind=kind, exception_chain=chain, infrastructure_success=kind == "contract_failure")
    record.update(completed_at=now(), elapsed_seconds=round(perf_counter() - started, 3), checkpoint_stats=stats,
        shared_prefix_nodes=len(prefix), computation_log=delegate.computation_log,
        g1_shared_validation={"source_g1_unavailable": True, "new_model_calls": 0},
        cost_accounting=infra.cost.accounting(record["requests"]))
    if cache.load(context):
        generations = []
        for _ in ["A2", "A3"]:
            generation = CachedG1Reasoner(cache, context, knowledge=provider, mode="require").generate_candidates(infra.model_profile(row))
            generation.candidates = infra.cost.base.CLINICR._validate_generation(generation, infra.cost.base.CLINICR._known_evidence_refs(row["profile"]))
            generations.append(generation.model_dump(mode="json"))
        identical = generations[0] == generations[1]
        if record["ok"]:
            result = record["result"]
            expected = [DiagnosticCandidate.model_validate({key: value[key] for key in DiagnosticCandidate.model_fields}).model_dump(mode="json") for value in result["candidates"]]
            identical = identical and generations[0]["tree_trace"] == result["tree_trace"] and generations[0]["candidate_coverage"] == result["candidate_coverage"] and generations[0]["candidates"] == expected
        record["g1_shared_validation"] = {"identical": identical, "new_model_calls": 0}
    return record


def final_snapshot(result):
    diagnosis = result["diagnosis"]
    return {"primary": (diagnosis["primary"] or {}).get("disorder"),
        "secondary": sorted(value["disorder"] for value in diagnosis["secondary"]),
        "provisional_primary": (result["provisional_diagnosis"] or {}).get("disorder"),
        "assessment_conclusions": {value["disorder"]: value["conclusion"] for value in result["checked"]},
        "verifier_routing": result["verifier_routing"]}


def comparison(full, computed):
    case_id = (full or computed)["case_id"]
    if not full or not computed or not full["ok"] or not computed["ok"]:
        rescued = bool(full and computed and not full["ok"] and computed["ok"] and full["failure_kind"] == "transport_or_http"
                       and full["requests"][-1]["stage"] == "criteria")
        return {"case_id": case_id, "complete": False, "passed": False, "criteria_timeout_rescued": rescued}
    a, b = full["result"], computed["result"]
    left, right = [{value["node_id"]: value for value in result["tree_trace"]} for result in [a, b]]
    criteria_ids = {key for key, value in left.items() if value["stage"] == "criterion"}
    differences = [{"node_id": key, "full": left[key]["status"], "computed": right[key]["status"]}
                   for key in sorted(set(left) & set(right)) if left[key]["status"] != right[key]["status"]]
    changed = sum(value["node_id"] in criteria_ids for value in differences)
    rate = 1 - changed / len(criteria_ids) if criteria_ids else 1.0
    coverage_loss = [label for label, status in a["candidate_coverage"].items()
                     if (status == "candidate" and b["candidate_coverage"][label] != "candidate") or
                     (status == "provisional_candidate" and b["candidate_coverage"][label] not in {"candidate", "provisional_candidate"})]
    labels_equal = {value["disorder"] for value in a["candidates"]} == {value["disorder"] for value in b["candidates"]}
    final_a, final_b = final_snapshot(a), final_snapshot(b)
    prefix_calls = [r for r in full["requests"] if r["stage"] not in {"criteria", "verifier"}]
    full_initial = [r for r in full["requests"] if r["stage"] == "criteria" and r["attempt"] == 1 and not r["is_structured_retry"]]
    computed_initial = [r for r in computed["requests"] if r["stage"] == "criteria" and r["attempt"] == 1 and not r["is_structured_retry"]]
    return {"case_id": case_id, "complete": True,
        "passed": set(left) == set(right) and not coverage_loss and labels_equal and rate >= .95,
        "same_visited_node_set": set(left) == set(right), "criteria_nodes": len(criteria_ids), "criteria_status_changed": changed,
        "criteria_status_agreement": rate, "all_node_status_agreement": 1 - len(differences) / max(len(left), len(right), 1),
        "coverage_loss": coverage_loss, "candidate_labels_identical": labels_equal, "status_differences": differences,
        "final_identical": final_a == final_b, "full_final": final_a, "computed_final": final_b,
        "initial_criteria_calls_full": len(full_initial), "initial_criteria_calls_computed": len(computed_initial),
        "fresh_case_http_attempt_estimate_computed": len(prefix_calls) + len(computed["requests"]),
        "full_case_http_attempts": len(full["requests"]), "shared_prefix_usage": extract(prefix_calls),
        "computed_incremental_usage": extract(computed["requests"])}


def summarize(records, budget, carry):
    paired = {}
    for row in records:
        paired.setdefault(row["case_id"], {})[row["arm"]] = row
    comparisons = [comparison(arms.get("full_llm"), arms.get("computed")) for arms in paired.values()]
    good = [row for row in comparisons if row["complete"]]
    denominator = sum(row["criteria_nodes"] for row in good)
    numerator = sum(row["criteria_status_changed"] for row in good)
    overall = 1 - numerator / denominator if denominator else None
    return {"completed_at": now(), "planned_cases": 20, "completed_case_arms": len(records),
        "successful_case_arms": sum(row["ok"] for row in records), "complete_pairs": len(good),
        "failure_kinds": dict(Counter(row["failure_kind"] for row in records if not row["ok"])),
        "criteria_status_agreement_overall": overall, "criteria_status_denominator": denominator,
        "comparisons": comparisons, "final_output_different_cases": [row["case_id"] for row in good if not row["final_identical"]],
        "g1_replay_failures": [row["case_id"] + ":" + row["arm"] for row in records if row["g1_shared_validation"].get("identical") is False],
        "criteria_timeout_rescued_cases": [row["case_id"] for row in comparisons if row.get("criteria_timeout_rescued")],
        "equivalence_passed": len(good) == 20 and all(row["passed"] for row in comparisons) and overall is not None and overall >= .95,
        "budget": budget.snapshot(), "phase_reference_upper_yuan": carry + budget.spent,
        "production_default_changed": False, "formal_scoring": False, "eval_opened": False, "actual_platform_invoice": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repair = infra.HERE / "runs/repair9-v1"
    assert (repair / "summary.json").exists(), "Finish the nine-case infrastructure batch first"
    carry = json.loads((repair / "summary.json").read_text(encoding="utf-8"))["phase_reference_upper_including_prior_batches"]
    rows = infra.read(HERE / "inputs.jsonl")
    assert len(rows) == 20
    manifest = {"started_at": now(), "case_ids": [row["case_id"] for row in rows], "version": VERSION,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=infra.REPO, text=True).strip(),
        "model": infra.cost.base.transport.get_model(), "base_url": infra.cost.base.transport.get_base_url(),
        "policy": infra.POLICY, "thinking": True, "arms": ["full_llm", "computed"],
        "prior_reference_upper_yuan": carry, "phase_reference_limit": 100,
        "shared_prefix": "same validated screen/anchor/guard; never regenerate missing prefix", "eval_opened": False}
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=True, indent=2))
        return
    if args.resume:
        original = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
        assert all(original[key] == manifest[key] for key in ["case_ids", "version", "commit", "model", "base_url", "policy", "thinking", "prior_reference_upper_yuan"])
    else:
        assert not RUN.exists(), "Never overwrite; use --resume after interruption"
        RUN.mkdir(parents=True)
        infra.cost.write(RUN / "manifest.json", manifest)
    records = infra.read(RUN / "results.jsonl") if (RUN / "results.jsonl").exists() else []
    done = {(row["case_id"], row["arm"]) for row in records}
    ledger = recover_ledger(RUN / "requests.jsonl")
    budget = ReferenceBudget(batch_limit=100 - carry, case_limit=10)
    budget.spent, budget.unknown_upper, budget.by_case = ledger["spent"], ledger["unknown_upper"], ledger["by_case"]
    lock = threading.Lock()
    with (RUN / "requests.jsonl").open("a", encoding="utf-8") as journal, (RUN / "results.jsonl").open("a", encoding="utf-8") as output:
        transport = ReliableStreaming(api_key=infra.cost.base.transport.get_api_key(), base_url=infra.cost.base.transport.get_base_url(), journal=journal, budget=budget)
        def task(row):
            for arm in ["full_llm", "computed"]:
                if (row["case_id"], arm) in done:
                    continue
                try:
                    record = (infra.run_case(row, arm=arm, thinking=True, run=RUN, transport=transport,
                              checkpoint_root=HERE / "checkpoints/full") if arm == "full_llm" else hybrid_case(row, transport))
                except Exception as error:
                    record = {"case_id": row["case_id"], "arm": arm, "ok": False, "failure_kind": "runner_or_replay_failure",
                        "requests": [], "exception_chain": infra.cost.exception_chain(error), "g1_shared_validation": {"source_g1_unavailable": True}}
                with lock:
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(json.dumps({"finished_arms": len(records), "case_id": row["case_id"], "arm": arm,
                        "ok": record["ok"], "failure_kind": record["failure_kind"], "phase_reference_upper": round(carry + budget.spent, 4)}), flush=True)
        with patch.object(infra.node_client, "ask_llm_stream", transport.ask), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(task, row) for row in rows]
            for future in as_completed(futures):
                future.result()
    result = summarize(records, budget, carry)
    infra.cost.write(RUN / "summary.json", result)
    print(json.dumps({key: result[key] for key in ["completed_case_arms", "complete_pairs", "equivalence_passed", "phase_reference_upper_yuan"]}), flush=True)


if __name__ == "__main__":
    main()

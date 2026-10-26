"""One fixed 20-case full/anchor-compact audit, no later-frontier model calls."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1.0.7"))
from run_denial_audit import base, observer, EventCapture, annotate_raw
from layers.layer3.diagnosis.decision_tree import anchor_frontier_state
from layers.layer3.llm.client import ANCHOR_PROMPT_VERSION
from layers.layer3.llm.node_knowledge import ANCHOR_VIEW_VERSION

RUN = HERE / "runs/full-vs-anchor-compact-v1"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")


def run_case(index, row, screens, graph):
    screen_map = {item["node_id"]: base.TreeNodeJudgment.model_validate(item) for item in screens["judgments"]}
    active = {label: path for label, path in graph["paths"].items()
              if any(screen_map[node].status != "not_met" for node in path["screens"])}
    ids = list(dict.fromkeys(node for path in active.values() for node in path["anchors"]))
    nodes = [{"node_id": node, **graph["nodes"][node]} for node in ids]
    records = []
    for arm in (["full", "compact"] if index % 2 == 0 else ["compact", "full"]):
        record = {"case_id": row["case_id"], "arm": arm, "started_at": base.now(), "requests": [],
                  "expected_node_ids": ids, "cached_screen_source": "p1.0.7/runs/dev-v1/results.jsonl#zero",
                  "screening_new_calls": 0, "screen_denial_scope_events": []}
        base.LOCAL.record = record
        provider = base.JsonDiagnosticKnowledgeProvider()
        counts = {"get_candidate": 0, "get_symptoms": 0}
        class Counted:
            def get_candidate(self, label):
                counts["get_candidate"] += 1
                return provider.get_candidate(label)
            def get_symptoms(self):
                counts["get_symptoms"] += 1
                return provider.get_symptoms()
        reasoner = base.APICLINICRReasoner(knowledge=Counted(), anchor_compact_view=arm == "compact")
        record["model"] = reasoner.model
        profile = {**row["profile"], base.ALLOWED_EVIDENCE_REFS_KEY: sorted(base.CLINICR._known_evidence_refs(row["profile"]))}
        start = perf_counter()
        try:
            answers = reasoner._evaluate_tree_nodes(profile, nodes)
            assert len(answers) == len(ids) and {item.node_id for item in answers} == set(ids)
            for offset in range(0, len(ids), 8):
                batch_ids = ids[offset:offset + 8]
                request = [item for item in record["requests"] if item["node_ids"] == batch_ids][-1]
                payload = json.loads(request["raw_output"])
                assert request.get("finish_reason") == "stop" and set(payload) == {"nodes"}
                assert len(payload["nodes"]) == len(batch_ids)
                for item in payload["nodes"]:
                    assert set(item) == set(base.TreeNodeJudgment.model_fields)
                    base.TreeNodeJudgment.model_validate(item, strict=True)
            judgments = [item.model_dump(mode="json") for item in answers]
            kept, coverage = anchor_frontier_state(graph, {**screen_map, **{item.node_id: item for item in answers}})
            record.update(ok=True, judgments=judgments, anchor_met_nodes=sorted(item.node_id for item in answers if item.status == "met"),
                          anchor_kept_labels=sorted(kept), candidate_coverage_after_anchor=coverage)
        except Exception as error:
            record.update(ok=False, error_type=type(error).__name__, status_code=getattr(error, "status_code", None))
        record.update(elapsed_seconds=round(perf_counter() - start, 3), knowledge_provider_calls=counts,
                      logical_api_calls=len(record["requests"]), structured_retry_count=sum(item["is_structured_retry"] for item in record["requests"]))
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            record[key] = base.request_total(record["requests"], key)
        records.append(annotate_raw(record))
    return records


def summarize(records, references, inputs):
    expected = {(row["case_id"], row["node_id"]): row for row in references}
    complete = len(records) == 40 and all(row["ok"] for row in records)
    summary = {"completed_at": base.now(), "attempted_case_arm_pairs": len(records), "expected_pairs": 40,
               "reference_type": "agent_pre_audit_not_expert_gold", "arms": {}}
    audit = []
    for arm in ["full", "compact"]:
        rows = [row for row in records if row["arm"] == arm]
        counts = Counter()
        for row in rows:
            if not row["ok"]:
                continue
            for item in row["judgments"]:
                ref = expected[(row["case_id"], item["node_id"])]
                positive = ref["reference_status"] == "met"
                counts["reference_met" if positive else "reference_not_met"] += 1
                if positive and item["status"] != "met":
                    counts["false_elimination_risk"] += 1
                    counts["false_not_met" if item["status"] == "not_met" else "false_pending"] += 1
                if not positive and item["status"] == "met":
                    counts["false_met"] += 1
                if item["status"] == "pending":
                    counts["pending"] += 1
                if item["status"] == "not_met":
                    counts["not_met_total"] += 1
                    counts["not_met_with_denial_present" if ref["explicit_symptom_denials"] else "not_met_without_denial"] += 1
                    if not item["evidence_refs"]:
                        counts["not_met_without_citation"] += 1
                    if set(item["evidence_refs"]) & {part["ref"] for part in ref["explicit_symptom_denials"]}:
                        counts["not_met_denial_ref_overlap"] += 1
                audit.append({"case_id": row["case_id"], "arm": arm, **item, "reference_status": ref["reference_status"],
                              "reference_support": ref["support"], "explicit_symptom_denials": ref["explicit_symptom_denials"],
                              "negative_basis": ref["negative_basis"]})
        requests = [request for row in rows for request in row["requests"]]
        info = {"attempted_cases": len(rows), "successful_cases": sum(row["ok"] for row in rows),
                "observed_nodes": sum(len(row.get("judgments", [])) for row in rows if row["ok"]),
                "counts_on_observed": dict(counts), "complete_20_case_counts": dict(counts) if len(rows) == 20 and all(row["ok"] for row in rows) else None,
                "logical_api_calls": len(requests), "structured_retries": sum(row["structured_retry_count"] for row in rows),
                "initial_batches": sum(not request["is_structured_retry"] for request in requests),
                "fingerprints": sorted({request.get("system_fingerprint") or "not_returned" for request in requests}),
                "failures": [{"case_id": row["case_id"], "error_type": row["error_type"], "status_code": row.get("status_code")} for row in rows if not row["ok"]]}
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "elapsed_seconds"]:
            values = [row[key] for row in rows if row[key] is not None]
            info[key] = {"observed_cases": len(values), "sum": sum(values), "p50": base.percentile(values, .5), "p95": base.percentile(values, .95),
                         "min": min(values) if values else None, "max": max(values) if values else None}
        batch_inputs = [request["usage"]["prompt_tokens"] for request in requests if request.get("usage")]
        info["request_prompt_tokens"] = {"observed_requests": len(batch_inputs), "p50": base.percentile(batch_inputs, .5),
                                        "p95": base.percentile(batch_inputs, .95), "max": max(batch_inputs) if batch_inputs else None}
        info["cost_accounting"] = {}
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            missing = sum((request.get("usage") or {}).get(key) is None for request in requests)
            known = sum((request.get("usage") or {}).get(key, 0) for request in requests)
            info["cost_accounting"][key] = {"known_sum": known, "missing_request_usage": missing, "total": None if missing else known}
        summary["arms"][arm] = info
    if complete:
        by_pair = {(row["case_id"], row["arm"]): row for row in records}
        for name in ["anchor_met_nodes", "anchor_kept_labels", "candidate_coverage_after_anchor"]:
            same = [row["case_id"] for row in inputs if by_pair[(row["case_id"], "full")][name] == by_pair[(row["case_id"], "compact")][name]]
            summary[name + "_identical_cases"] = len(same)
            summary[name + "_differences"] = [{"case_id": row["case_id"], "full": by_pair[(row["case_id"], "full")][name],
                                                "compact": by_pair[(row["case_id"], "compact")][name]} for row in inputs if row["case_id"] not in same]
        full, compact = summary["arms"]["full"], summary["arms"]["compact"]
        gates = {"coverage_20_of_20": summary["candidate_coverage_after_anchor_identical_cases"] == 20,
                 "met_nodes_20_of_20": summary["anchor_met_nodes_identical_cases"] == 20,
                 "kept_labels_20_of_20": summary["anchor_kept_labels_identical_cases"] == 20,
                 "zero_false_not_met": compact["counts_on_observed"].get("false_not_met", 0) == 0,
                 "zero_false_elimination_risk": compact["counts_on_observed"].get("false_elimination_risk", 0) == 0,
                 "false_met_not_above_full": compact["counts_on_observed"].get("false_met", 0) <= full["counts_on_observed"].get("false_met", 0),
                 "batch_input_p95_le_8000": compact["request_prompt_tokens"]["p95"] is not None and compact["request_prompt_tokens"]["p95"] <= 8000,
                 "usage_complete": all(info["cost_accounting"]["prompt_tokens"]["missing_request_usage"] == 0 for info in summary["arms"].values()),
                 "fingerprints_compatible": full["fingerprints"] == compact["fingerprints"] == ["e72d3c7f663d3e1383b00b3884cb4501"],
                 "compact_source_policy": all(row["knowledge_provider_calls"] == {"get_candidate": 0, "get_symptoms": 1} for row in records if row["arm"] == "compact")}
        summary.update(gates=gates, gate="pass" if all(gates.values()) else "fail")
    else:
        summary["gate"] = "incomplete"
    summary["anchor_default"] = "full"  # No automatic default edit by an experiment runner.
    (RUN / "audit-rows.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in audit), encoding="utf-8")
    write(RUN / "summary.json", summary)
    return summary


def main():
    assert not RUN.exists(), "Keep completed and partial runs, never overwrite"
    inputs = base.read_rows(HERE.parent / "p1.0.6/inputs.jsonl")
    refs = base.read_rows(HERE / "reference.jsonl")
    source = HERE.parent / "p1.0.7/runs/dev-v1"
    old_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    assert old_manifest["screening_prompt_version"] == base.SCREENING_PROMPT_VERSION
    screens = {row["case_id"]: row for row in base.read_rows(source / "results.jsonl") if row["arm"] == "zero"}
    assert len(inputs) == len(screens) == 20 and len(refs) == 480 and all(row["ok"] for row in screens.values())
    graph = base.load_candidate_graph()
    assert graph["version"] == old_manifest["tree_version"]
    RUN.mkdir(parents=True)
    write(RUN / "manifest.json", {"started_at": base.now(), "phase": "P1.0.8", "case_ids": [row["case_id"] for row in inputs],
          "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.REPO, text=True).strip(),
          "anchor_prompt_version": ANCHOR_PROMPT_VERSION, "anchor_view_version": ANCHOR_VIEW_VERSION,
          "tree_version": graph["version"], "model_alias": base.transport.get_model(), "screening_prompt_version": base.SCREENING_PROMPT_VERSION,
          "model_official_mapping_at_last_check": "DeepSeek-V4.1-Flash-1017", "mapping_check_date": "2026-10-22",
          "expected_fingerprint": "e72d3c7f663d3e1383b00b3884cb4501", "seed_for_selection": 20261015,
          "max_workers": 4, "temperature": 0, "response_format": {"type": "json_object"}, "base_new_requests": 120,
          "structured_retry_budget_per_batch": 1, "sdk_max_retries": 1, "batch_input_p95_target": 8000,
          "default_anchor_compact": False, "default_provider": type(base.get_default_knowledge_provider()).__name__,
          "reference_type": "agent_pre_audit_symptom_witnesses_not_expert_gold", "gold_read": False,
          "criteria_or_guard_model_calls": 0, "screening_new_calls": 0})
    handler = EventCapture()
    logger = logging.getLogger("layers.layer3.diagnosis.screening")
    logger.addHandler(handler)
    records = []
    print(json.dumps({"starting": "P1.0.8 full vs anchor compact", "cases": 20, "base_new_requests": 120}), flush=True)
    try:
        with patch.object(base.transport, "_bounded_client", observer(base.transport._bounded_client)), ThreadPoolExecutor(max_workers=4) as pool, (RUN / "results.jsonl").open("w", encoding="utf-8") as output:
            futures = [pool.submit(run_case, index, row, screens[row["case_id"]], graph) for index, row in enumerate(inputs)]
            for future in as_completed(futures):
                batch = future.result()
                records.extend(batch)
                for record in batch:
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')
                output.flush()
                print(json.dumps({"case_id": batch[0]["case_id"], "completed_pairs": len(records),
                                  "failed_pairs": sum(not row["ok"] for row in records)}), flush=True)
    finally:
        logger.removeHandler(handler)
    summary = summarize(records, refs, inputs)
    print(json.dumps({"gate": summary["gate"], "completed_pairs": len(records), "anchor_default": "full"}), flush=True)
    return 0 if summary["gate"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

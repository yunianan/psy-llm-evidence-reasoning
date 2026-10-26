"""Paired 20-case DEV screen audit; no gold, EVAL, or full-engine calls."""
from collections import Counter
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import threading
from time import perf_counter
from unittest.mock import patch

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "L1-L5系统"
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env", override=False, encoding="utf-8-sig")

from shared import llm_client as transport
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider, get_default_knowledge_provider
from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.diagnosis.decision_tree import load_candidate_graph
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from layers.layer3.llm.client import ALLOWED_EVIDENCE_REFS_KEY, APICLINICRReasoner, SCREENING_PROMPT_VERSION
from layers.layer3.llm.node_knowledge import NODE_KNOWLEDGE_VIEW_VERSION, COMPACT_SCREEN_VIEW_VERSION
from audit_metrics import active_labels, decision, error_counts, percentile

RUN = HERE / "runs/zero-vs-full-v1"
LOCAL = threading.local()
CANDIDATE_ARM = "zero"
REUSE_FULL = False


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def request_total(requests, key):
    values = [(item.get("usage") or {}).get(key) for item in requests]
    return sum(values) if values and all(isinstance(value, int) for value in values) else None


def run_case(index, row, graph, nodes):
    records = []
    profile = row["profile"]
    model_profile = {**profile, ALLOWED_EVIDENCE_REFS_KEY: sorted(CLINICR._known_evidence_refs(profile))}
    arms = [CANDIDATE_ARM] if REUSE_FULL else (["full", CANDIDATE_ARM] if index % 2 == 0 else [CANDIDATE_ARM, "full"])
    for arm in arms:
        record = {"case_id": row["case_id"], "arm": arm, "started_at": now(), "requests": [], "executed_in_this_run": True}
        LOCAL.record = record
        base = JsonDiagnosticKnowledgeProvider()
        counts = {"get_candidate": 0, "get_symptoms": 0}

        class CountingKnowledge:
            def get_candidate(self, label):
                counts["get_candidate"] += 1
                return base.get_candidate(label)

            def get_symptoms(self):
                counts["get_symptoms"] += 1
                return base.get_symptoms()

        reasoner = APICLINICRReasoner(knowledge=CountingKnowledge(), screening_zero_knowledge=arm == "zero", screening_compact_view=arm == "compact")
        record["model"] = reasoner.model
        started = perf_counter()
        try:
            answers = reasoner._evaluate_tree_nodes(model_profile, nodes)
            record["judgments"] = [item.model_dump(mode="json") for item in answers]
            assert len(answers) == 9 and {item.node_id for item in answers} == {node["node_id"] for node in nodes}
            for offset in range(0, len(nodes), 8):
                expected = [node["node_id"] for node in nodes[offset:offset+8]]
                last = [item for item in record["requests"] if item["node_ids"] == expected][-1]
                assert last.get("finish_reason") == "stop"
                payload = json.loads(last["raw_output"])
                assert set(payload) == {"nodes"} and isinstance(payload["nodes"], list)
                assert len(payload["nodes"]) == len(expected)
                for item in payload["nodes"]:
                    assert set(item) == set(TreeNodeJudgment.model_fields)
                    TreeNodeJudgment.model_validate(item, strict=True)
            record["active_labels"] = active_labels(record["judgments"], graph)
            record["ok"] = True
        except Exception as error:
            record.update(ok=False, error_type=type(error).__name__, status_code=getattr(error, "status_code", None))
        record["elapsed_seconds"] = round(perf_counter() - started, 3)
        record["knowledge_provider_calls"] = counts
        record["structured_retry_count"] = sum(item["is_structured_retry"] for item in record["requests"])
        record["logical_api_calls"] = len(record["requests"])
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            record[key] = request_total(record["requests"], key)
        records.append(record)
    return records


def summarize(results, inputs, references, graph):
    reference_by_case = {row["case_id"]: [item for item in references if item["case_id"] == row["case_id"]] for row in inputs}
    by_pair = {(item["case_id"], item["arm"]): item for item in results}
    complete = len(results) == 40 and all(item["ok"] for item in results)
    summary = {"completed_at": now(), "case_count": len(inputs), "expected_case_arm_pairs": 40,
               "completed_case_arm_pairs": len(results), "failed_pairs": [{"case_id": item["case_id"], "arm": item["arm"], "error_type": item.get("error_type")} for item in results if not item["ok"]],
               "reference_type": "agent_pre_audit", "arms": {}}
    audit_rows = []
    for arm in ["full", CANDIDATE_ARM]:
        records = [item for item in results if item["arm"] == arm]
        counts = Counter()
        for record in records:
            if not record["ok"]:
                continue
            expected = reference_by_case[record["case_id"]]
            counts.update(error_counts(expected, record["judgments"]))
            ref_by_node = {item["node_id"]: item for item in expected}
            quotes = {ref: item["value"] for row in inputs if row["case_id"] == record["case_id"] for item in row["profile"]["key_narratives"] for ref in item["sources"]}
            for item in record["judgments"]:
                ref = ref_by_node[item["node_id"]]
                category = ("overcall" if item["status"] == "met" and ref["reference_status"] != "met" else
                            "undercall" if ref["reference_status"] == "met" and item["status"] != "met" else "compatible")
                audit_rows.append({"case_id": record["case_id"], "arm": arm, "node_id": item["node_id"],
                                   "reference_status": ref["reference_status"], "predicted_status": item["status"],
                                   "category": category, "model_rationale": item["rationale"], "reference_support": ref["support"],
                                   "cited_patient_quotes": [{"ref": evidence, "quote": quotes.get(evidence)} for evidence in item["evidence_refs"]],
                                   "reference_support_overlap": bool(set(item["evidence_refs"]) & {evidence["ref"] for evidence in ref["support"]})})
        info = {"cases": len(records), "successful_cases": sum(item["ok"] for item in records),
                "errors": dict(counts) if complete else None,
                "overcall_rate": counts["overcall"] / counts["nonmet_denominator"] if complete else None,
                "undercall_rate": counts["undercall"] / counts["met_denominator"] if complete else None,
                "logical_api_calls": sum(item["logical_api_calls"] for item in records),
                "structured_retries": sum(item["structured_retry_count"] for item in records),
                "reused_cases": sum(not item.get("executed_in_this_run", True) for item in records),
                "new_logical_api_calls": sum(item["logical_api_calls"] for item in records if item.get("executed_in_this_run", True)),
                "fingerprints": sorted({request.get("system_fingerprint") or "not_returned" for item in records for request in item["requests"]})}
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "elapsed_seconds"]:
            values = [item[key] for item in records if item[key] is not None]
            info[key] = {"observed_cases": len(values), "sum": round(sum(values), 4),
                         "p50": percentile(values, .5), "p95": percentile(values, .95),
                         "min": min(values) if values else None, "max": max(values) if values else None}
        summary["arms"][arm] = info
    if complete:
        same = [row["case_id"] for row in inputs if by_pair[(row["case_id"], "full")]["active_labels"] == by_pair[(row["case_id"], CANDIDATE_ARM)]["active_labels"]]
        differences = [{"case_id": row["case_id"], "full_active": by_pair[(row["case_id"], "full")]["active_labels"],
                        "candidate_active": by_pair[(row["case_id"], CANDIDATE_ARM)]["active_labels"]} for row in inputs if row["case_id"] not in same]
        summary.update(activity_identical_cases=len(same), activity_identical_rate=len(same)/20, activity_differences=differences)
        summary.update(decision(summary["arms"]["full"]["errors"], summary["arms"][CANDIDATE_ARM]["errors"], len(same), 20))
        same_fingerprint = summary["arms"]["full"]["fingerprints"] == summary["arms"][CANDIDATE_ARM]["fingerprints"] and "not_returned" not in summary["arms"][CANDIDATE_ARM]["fingerprints"]
        knowledge_policy = all(item["knowledge_provider_calls"]["get_candidate"] == 0 and
                               (CANDIDATE_ARM == "compact" or item["knowledge_provider_calls"]["get_symptoms"] == 0)
                               for item in results if item["arm"] == CANDIDATE_ARM)
        summary.update(fingerprint_compatible=same_fingerprint, candidate_knowledge_policy_verified=knowledge_policy)
        if not same_fingerprint or not knowledge_policy:
            summary["gate"] = "incomplete"
    else:
        summary["gate"] = "incomplete"
    (RUN / "audit-rows.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in audit_rows), encoding="utf-8")
    (RUN / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    return summary


def main():
    global RUN, CANDIDATE_ARM, REUSE_FULL
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-mode", choices=["zero", "compact"], default="zero")
    args = parser.parse_args()
    CANDIDATE_ARM = args.candidate_mode
    REUSE_FULL = CANDIDATE_ARM == "compact"
    RUN = HERE / f"runs/{CANDIDATE_ARM}-vs-full-v1"
    assert not (RUN / "results.jsonl").exists(), "Keep completed runs; do not overwrite"
    inputs = read_rows(HERE / "inputs.jsonl")
    references = read_rows(HERE / "reference.jsonl")
    assert len(inputs) == 20 and len({row["case_id"] for row in inputs}) == 20
    assert len(references) == 180
    graph = load_candidate_graph()
    ids = list(dict.fromkeys(node for path in graph["paths"].values() for node in path["screens"]))
    nodes = [{"node_id": node, **graph["nodes"][node]} for node in ids]
    RUN.mkdir(parents=True, exist_ok=True)
    manifest = {"started_at": now(), "model": transport.get_model(), "screening_prompt_version": SCREENING_PROMPT_VERSION,
                "node_knowledge_view_version": NODE_KNOWLEDGE_VIEW_VERSION, "tree_version": graph["version"],
                "candidate_mode": CANDIDATE_ARM, "compact_view_version": COMPACT_SCREEN_VIEW_VERSION if REUSE_FULL else None,
                "default_provider": type(get_default_knowledge_provider()).__name__, "case_ids": [row["case_id"] for row in inputs],
                "seed": 20261015, "max_case_workers": 4, "arm_order": "alternating by fixed case index",
                "sdk_max_retries": 1, "structured_retry_budget_per_batch": 1, "reference_version": "screen-reference-v1",
                "gold_read": False, "scope": "screening only"}
    (RUN / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    original_bounded = transport._bounded_client

    def observed_client(*, timeout):
        api = original_bounded(timeout=timeout)
        original_create = api.chat.completions.create

        def create(**request):
            owner = LOCAL.record
            prompt = request["messages"][0]["content"]
            batch, _ = json.JSONDecoder().raw_decode(prompt.split("当前节点 JSON：", 1)[1])
            node_ids = [node["node_id"] for node in batch]
            record = {"node_ids": node_ids, "is_structured_retry": any(item["node_ids"] == node_ids for item in owner["requests"]),
                      "prompt_characters": len(prompt), "usage": None, "response_format": request.get("response_format")}
            owner["requests"].append(record)
            started = perf_counter()
            try:
                response_stream = original_create(**request)
            except Exception as error:
                record.update(error_type=type(error).__name__, elapsed_seconds=round(perf_counter()-started, 3))
                raise

            def chunks():
                parts = []
                try:
                    for chunk in response_stream:
                        record["response_model"] = chunk.model
                        if getattr(chunk, "system_fingerprint", None):
                            record["system_fingerprint"] = chunk.system_fingerprint
                        if getattr(chunk, "usage", None):
                            record["usage"] = chunk.usage.model_dump(mode="json")
                        for choice in chunk.choices or []:
                            if choice.delta.content:
                                parts.append(choice.delta.content)
                            if choice.finish_reason:
                                record["finish_reason"] = choice.finish_reason
                        yield chunk
                except Exception as error:
                    record["error_type"] = type(error).__name__
                    raise
                finally:
                    record["raw_output"] = ''.join(parts)
                    record["elapsed_seconds"] = round(perf_counter()-started, 3)
            return chunks()
        api.chat.completions.create = create
        return api

    results = []
    if REUSE_FULL:
        previous = HERE / "runs/zero-vs-full-v1"
        old_manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
        assert old_manifest["screening_prompt_version"] == SCREENING_PROMPT_VERSION
        assert old_manifest["case_ids"] == manifest["case_ids"]
        results = [item for item in read_rows(previous / "results.jsonl") if item["arm"] == "full"]
        assert len(results) == 20 and all(item["ok"] for item in results)
        for item in results:
            item.update(executed_in_this_run=False, reused_from="zero-vs-full-v1/results.jsonl")
    print(json.dumps({"starting": "20-case paired screen audit", "cases": 20, "candidate_mode": CANDIDATE_ARM,
                      "base_new_requests": 40 if REUSE_FULL else 80, "reused_full_cases": len(results)}), flush=True)
    with patch.object(transport, "_bounded_client", observed_client), ThreadPoolExecutor(max_workers=4) as executor, (RUN / "results.jsonl").open("w", encoding="utf-8") as output:
        for record in results:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
        output.flush()
        futures = [executor.submit(run_case, index, row, graph, nodes) for index, row in enumerate(inputs)]
        for future in as_completed(futures):
            records = future.result()
            results.extend(records)
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps({"completed_pairs": len(results), "total_pairs": 40, "case_id": records[0]["case_id"],
                              "failed_pairs": sum(not item["ok"] for item in results)}), flush=True)
    summary = summarize(results, inputs, references, graph)
    print(json.dumps({key: summary[key] for key in ["gate", "completed_case_arm_pairs", "activity_identical_cases"] if key in summary}), flush=True)
    return 0 if summary["gate"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

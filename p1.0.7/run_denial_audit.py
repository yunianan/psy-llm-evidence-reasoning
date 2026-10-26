"""Bounded P1.0.7 audit; preserve P1.0.6 artifacts and capture raw/effective decisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import logging
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "p1.0.6"
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location("p106_audit", PREVIOUS / "run_screen_audit.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from layers.layer3.diagnosis.screening import SCREEN_DENIAL_POLICY_VERSION
from layers.layer3.llm import client


class EventCapture(logging.Handler):
    def emit(self, record):
        owner = getattr(base.LOCAL, "record", None)
        if owner is not None and getattr(record, "event", None) == "screen_denial_scope_downgrade":
            owner.setdefault("screen_denial_scope_events", []).append({
                key: getattr(record, key) for key in ["event", "node_id", "original_status", "effective_status",
                                                     "denial_scope", "evidence_refs", "policy_version"]
            })


def observer(original_bounded):
    def observed_client(*, timeout):
        api = original_bounded(timeout=timeout)
        original_create = api.chat.completions.create

        def create(**request):
            owner = base.LOCAL.record
            prompt = request["messages"][0]["content"]
            batch, _ = json.JSONDecoder().raw_decode(prompt.split("当前节点 JSON：", 1)[1])
            ids = [item["node_id"] for item in batch]
            record = {"node_ids": ids, "is_structured_retry": any(item["node_ids"] == ids for item in owner["requests"]),
                      "prompt_characters": len(prompt), "usage": None, "response_format": request.get("response_format")}
            owner["requests"].append(record)
            start = perf_counter()
            try:
                response = original_create(**request)
            except Exception as error:
                record.update(error_type=type(error).__name__, elapsed_seconds=round(perf_counter() - start, 3))
                raise

            def chunks():
                parts = []
                try:
                    for chunk in response:
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
                    record.update(raw_output=''.join(parts), elapsed_seconds=round(perf_counter() - start, 3))
            return chunks()
        api.chat.completions.create = create
        return api
    return observed_client


def annotate_raw(record):
    latest = {}
    for request in record["requests"]:
        latest[tuple(request["node_ids"])] = request
    raw = []
    for request in latest.values():
        try:
            raw.extend(json.loads(request["raw_output"])["nodes"])
        except (KeyError, ValueError, TypeError):
            pass
    record["raw_judgments"] = raw
    record.setdefault("screen_denial_scope_events", [])
    return record


def targeted_case(index, row, graph):
    records = []
    for arm in (["zero", "compact"] if index % 2 == 0 else ["compact", "zero"]):
        record = {"case_id": row["case_id"], "arm": arm, "started_at": base.now(), "requests": [], "executed_in_this_run": True}
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
        reasoner = client.APICLINICRReasoner(knowledge=Counted(), screening_zero_knowledge=arm == "zero", screening_compact_view=arm == "compact")
        profile = {**row["profile"], base.ALLOWED_EVIDENCE_REFS_KEY: [row["reference_support"]["ref"]]}
        record["model"] = reasoner.model
        start = perf_counter()
        try:
            answers = reasoner._evaluate_tree_nodes(profile, [{"node_id": row["node_id"], **graph["nodes"][row["node_id"]]}])
            assert len(answers) == 1 and answers[0].node_id == row["node_id"]
            payload = json.loads(record["requests"][-1]["raw_output"])
            assert set(payload) == {"nodes"} and len(payload["nodes"]) == 1
            assert set(payload["nodes"][0]) == set(base.TreeNodeJudgment.model_fields)
            base.TreeNodeJudgment.model_validate(payload["nodes"][0], strict=True)
            assert record["requests"][-1]["finish_reason"] == "stop"
            record.update(ok=True, judgments=[item.model_dump(mode="json") for item in answers])
        except Exception as error:
            record.update(ok=False, error_type=type(error).__name__, status_code=getattr(error, "status_code", None))
        record.update(elapsed_seconds=round(perf_counter() - start, 3), knowledge_provider_calls=counts,
                      logical_api_calls=len(record["requests"]), structured_retry_count=sum(item["is_structured_retry"] for item in record["requests"]))
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            record[key] = base.request_total(record["requests"], key)
        records.append(annotate_raw(record))
    return records


def parallel_run(rows, callback, output, results):
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(callback, index, row) for index, row in enumerate(rows)]
        for future in as_completed(futures):
            records = future.result()
            results.extend(records)
            for record in records:
                annotate_raw(record)
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps({"case_id": records[0]["case_id"], "completed_pairs": len(results), "failures": sum(not row["ok"] for row in results)}), flush=True)


def summarize_targeted(results, inputs):
    ref = {row["case_id"]: row for row in inputs}
    summary = {"suite": "targeted", "completed_at": base.now(), "arms": {}, "completed_pairs": len(results)}
    complete = len(results) == 32 and all(row["ok"] for row in results)
    for arm in ["zero", "compact"]:
        rows = [row for row in results if row["arm"] == arm and row["ok"]]
        info = {"successful_cases": len(rows), "false_not_met_nonfull_denominator": 12, "full_denial_denominator": 4}
        for label, key in [("raw", "raw_judgments"), ("effective", "judgments")]:
            info[label + "_false_not_met"] = sum(ref[row["case_id"]]["expected_status"] != "not_met" and row[key][0]["status"] == "not_met" for row in rows)
        info["full_denial_correct"] = sum(ref[row["case_id"]]["expected_status"] == "not_met" and row["judgments"][0]["status"] == "not_met" and row["judgments"][0]["denial_scope"] == "full" for row in rows)
        info["scope_correct"] = sum(ref[row["case_id"]]["expected_denial_scope"] == row["judgments"][0]["denial_scope"] for row in rows)
        info["status_correct"] = sum(ref[row["case_id"]]["expected_status"] == row["judgments"][0]["status"] for row in rows)
        info["downgrades"] = sum(len(row["screen_denial_scope_events"]) for row in rows)
        info["logical_calls"] = sum(row["logical_api_calls"] for row in rows)
        info["structured_retries"] = sum(row["structured_retry_count"] for row in rows)
        info["gate"] = "pass" if complete and info["effective_false_not_met"] == 0 and info["full_denial_correct"] == 4 and info["status_correct"] == 16 else ("fail" if complete else "incomplete")
        summary["arms"][arm] = info
    summary["gate"] = "pass" if complete and all(item["gate"] == "pass" for item in summary["arms"].values()) else ("fail" if complete else "incomplete")
    return summary


def summarize_dev(results, inputs, graph, run):
    references = base.read_rows(PREVIOUS / "reference.jsonl")
    expected = {(row["case_id"], row["node_id"]): row["reference_status"] for row in references}
    old_full = {row["case_id"]: row["active_labels"] for row in base.read_rows(PREVIOUS / "runs/zero-vs-full-v1/results.jsonl") if row["arm"] == "full"}
    summary = {"suite": "dev", "completed_at": base.now(), "completed_pairs": len(results), "comparisons": {}, "arms": {}}
    for arm in ["zero", "compact"]:
        base.CANDIDATE_ARM = arm
        base.RUN = run / (arm + "-vs-full")
        base.RUN.mkdir()
        summary["comparisons"][arm] = base.summarize([row for row in results if row["arm"] in ["full", arm]], inputs, references, graph)
    for arm in ["full", "zero", "compact"]:
        rows = [row for row in results if row["arm"] == arm]
        info = {"successful_cases": sum(row["ok"] for row in rows), "expected_node_denominator": 180,
                "observed_node_denominator": sum(len(row.get("judgments", [])) for row in rows if row["ok"]),
                "expected_case_denominator": 20}
        for label, key in [("raw", "raw_judgments"), ("effective", "judgments")]:
            false_nodes = [{"case_id": row["case_id"], **item} for row in rows if row["ok"] for item in row[key]
                           if item["status"] == "not_met" and expected[(row["case_id"], item["node_id"])] != "not_met"]
            info[label + "_false_elimination_nodes"] = len(false_nodes)
            info[label + "_false_elimination_cases"] = len({item["case_id"] for item in false_nodes})
            info[label + "_false_eliminations"] = false_nodes
        info["downgrades"] = sum(len(row["screen_denial_scope_events"]) for row in rows)
        info["old_full_activity_identical_cases"] = sum(row["ok"] and row["active_labels"] == old_full[row["case_id"]] for row in rows)
        info["mb017541"] = [{"status": item["status"], "scope": item["denial_scope"], "D017_retained": "D017" in row["active_labels"]}
                            for row in rows if row["ok"] and row["case_id"] == "mb-017541" for item in row["judgments"] if item["node_id"] == "screen.appearance"]
        summary["arms"][arm] = info
    complete = len(results) == 60 and all(row["ok"] for row in results)
    for arm, comparison in summary["comparisons"].items():
        pair_complete = all(summary["arms"][name]["successful_cases"] == 20 for name in ["full", arm])
        comparison["no_false_elimination_pass"] = (all(summary["arms"][name]["effective_false_elimination_nodes"] == 0 for name in ["full", arm])
                                                   if pair_complete else None)
        if complete and not comparison["no_false_elimination_pass"]:
            comparison["gate"] = "fail"
    summary["gate"] = "pass" if complete and all(item["gate"] == "pass" for item in summary["comparisons"].values()) else ("fail" if complete else "incomplete")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["targeted", "dev"], required=True)
    args = parser.parse_args()
    run = HERE / "runs" / (args.suite + "-v1")
    assert not run.exists(), "Keep completed and partial runs; do not overwrite"
    run.mkdir(parents=True)
    inputs = base.read_rows(HERE / "targeted-denial-set.jsonl" if args.suite == "targeted" else PREVIOUS / "inputs.jsonl")
    assert len(inputs) == (16 if args.suite == "targeted" else 20)
    graph = base.load_candidate_graph()
    ids = list(dict.fromkeys(node for path in graph["paths"].values() for node in path["screens"]))
    nodes = [{"node_id": node, **graph["nodes"][node]} for node in ids]
    manifest = {"started_at": base.now(), "suite": args.suite, "case_ids": [row["case_id"] for row in inputs],
                "screening_prompt_version": base.SCREENING_PROMPT_VERSION, "denial_policy_version": SCREEN_DENIAL_POLICY_VERSION,
                "tree_version": graph["version"], "model_alias": base.transport.get_model(),
                "model_official_mapping_at_last_check": "DeepSeek-V4.1-Flash-1017", "mapping_check_date": "2026-10-22",
                "previous_system_fingerprint": "e72d3c7f663d3e1383b00b3884cb4501", "temperature": 0,
                "seed_for_dev_selection": 20261015, "max_workers": 4, "response_format": {"type": "json_object"},
                "structured_retry_budget_per_batch": 1, "sdk_max_retries": 1, "gold_read": False,
                "default_provider": type(base.get_default_knowledge_provider()).__name__,
                "reuse_p106_full": False, "non_screen_stages_run": False}
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    handler = EventCapture()
    logger = logging.getLogger("layers.layer3.diagnosis.screening")
    logger.addHandler(handler)
    results = []
    print(json.dumps({"starting": args.suite, "cases": len(inputs), "base_requests": 32 if args.suite == "targeted" else 120}), flush=True)
    try:
        with patch.object(base.transport, "_bounded_client", observer(base.transport._bounded_client)), (run / "results.jsonl").open("w", encoding="utf-8") as output:
            if args.suite == "targeted":
                parallel_run(inputs, lambda index, row: targeted_case(index, row, graph), output, results)
            else:
                base.CANDIDATE_ARM, base.REUSE_FULL = "zero", False
                parallel_run(inputs, lambda index, row: base.run_case(index, row, graph, nodes), output, results)
                base.CANDIDATE_ARM, base.REUSE_FULL = "compact", True
                parallel_run(inputs, lambda index, row: base.run_case(index, row, graph, nodes), output, results)
    finally:
        logger.removeHandler(handler)
    summary = summarize_targeted(results, inputs) if args.suite == "targeted" else summarize_dev(results, inputs, graph, run)
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    print(json.dumps({"gate": summary["gate"], "completed_pairs": len(results)}), flush=True)
    return 0 if summary["gate"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

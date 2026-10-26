"""One complete compact retest after balance recovery; keep the first run immutable."""
import json
import logging
from pathlib import Path
import subprocess
from unittest.mock import patch

from run_denial_audit import base, EventCapture, observer, parallel_run, summarize_dev
from layers.layer3.diagnosis.screening import SCREEN_DENIAL_POLICY_VERSION
from layers.layer3.llm.node_knowledge import COMPACT_SCREEN_VIEW_VERSION

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/dev-v1"
RUN = HERE / "runs/compact-retest-v1"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")


def main():
    assert not RUN.exists(), "Do not overwrite a completed or partial retest"
    old_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    assert old_manifest["screening_prompt_version"] == base.SCREENING_PROMPT_VERSION
    assert old_manifest["denial_policy_version"] == SCREEN_DENIAL_POLICY_VERSION
    inputs = base.read_rows(HERE.parent / "p1.0.6/inputs.jsonl")
    assert len(inputs) == 20 and [row["case_id"] for row in inputs] == old_manifest["case_ids"]
    original = base.read_rows(SOURCE / "results.jsonl")
    reused = [row for row in original if row["arm"] in ["full", "zero"]]
    assert len(reused) == 40 and all(row["ok"] for row in reused)
    assert len({(row["case_id"], row["arm"]) for row in reused}) == 40
    for row in reused:
        row.update(executed_in_this_run=False, reused_from="../dev-v1/results.jsonl")
    graph = base.load_candidate_graph()
    assert graph["version"] == old_manifest["tree_version"]
    node_ids = list(dict.fromkeys(node for path in graph["paths"].values() for node in path["screens"]))
    nodes = [{"node_id": node, **graph["nodes"][node]} for node in node_ids]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.REPO, text=True).strip()
    manifest = {**old_manifest, "started_at": base.now(), "suite": "compact-retest", "implementation_commit": commit,
                "compact_view_version": COMPACT_SCREEN_VIEW_VERSION, "candidate_arm": "compact",
                "base_new_requests": 40, "new_cases": 20, "reused_case_arm_pairs": 40,
                "reused_from": "../dev-v1/results.jsonl", "full_and_zero_reused": True,
                "retest_policy": "one complete 20-case compact run; no success-subset merging; retain original failures"}
    RUN.mkdir(parents=True)
    write(RUN / "manifest.json", manifest)
    base.CANDIDATE_ARM, base.REUSE_FULL = "compact", True
    handler = EventCapture()
    logger = logging.getLogger("layers.layer3.diagnosis.screening")
    logger.addHandler(handler)
    records = list(reused)
    print(json.dumps({"starting": "compact full-20 retest", "base_new_requests": 40, "reused_pairs": len(reused)}), flush=True)
    try:
        with patch.object(base.transport, "_bounded_client", observer(base.transport._bounded_client)), (RUN / "results.jsonl").open("w", encoding="utf-8") as output:
            for row in reused:
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            parallel_run(inputs, lambda index, row: base.run_case(index, row, graph, nodes), output, records)
    finally:
        logger.removeHandler(handler)
    summary = summarize_dev(records, inputs, graph, RUN)
    checks = {"targeted_pass": json.loads((HERE / "runs/targeted-v1/summary.json").read_text(encoding="utf-8"))["gate"] == "pass",
              "successful_case_schema_and_refs": True,
              "zero_source_reads": True, "compact_no_full_packages": True}
    by_id = {row["case_id"]: row for row in inputs}
    for record in records:
        if not record["ok"]:
            continue
        refs = base.CLINICR._known_evidence_refs(by_id[record["case_id"]]["profile"])
        for judgment in record["judgments"]:
            base.TreeNodeJudgment.model_validate(judgment, strict=True)
            checks["successful_case_schema_and_refs"] &= set(judgment["evidence_refs"]) <= refs
        if record["arm"] == "zero":
            checks["zero_source_reads"] &= record["knowledge_provider_calls"] == {"get_candidate": 0, "get_symptoms": 0}
        if record["arm"] == "compact":
            checks["compact_no_full_packages"] &= record["knowledge_provider_calls"]["get_candidate"] == 0
    current = [row for row in records if row["arm"] == "compact"]
    old_compact = [row for row in original if row["arm"] == "compact"]
    cumulative_requests = [request for row in old_compact + current for request in row["requests"]]
    cumulative = {"case_attempts": len(old_compact) + len(current),
                  "successful_case_attempts": sum(row["ok"] for row in old_compact + current),
                  "first_run_failed_case_ids": [row["case_id"] for row in old_compact if not row["ok"]],
                  "retest_failed_case_ids": [row["case_id"] for row in current if not row["ok"]],
                  "logical_requests": len(cumulative_requests)}
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        missing = sum((request.get("usage") or {}).get(key) is None for request in cumulative_requests)
        known = sum((request.get("usage") or {}).get(key, 0) for request in cumulative_requests)
        cumulative[key] = {"known_sum": known, "missing_request_usage": missing, "total": None if missing else known}
    summary.update(checks=checks, compact_cumulative_accounting=cumulative,
                   historical_first_run_gate="incomplete",
                   decision="select_zero_default" if summary["gate"] == "pass" and all(checks.values()) else "hold")
    write(RUN / "summary.json", summary)
    print(json.dumps({"gate": summary["gate"], "decision": summary["decision"],
                      "new_successful_cases": sum(row["ok"] for row in current),
                      "new_logical_requests": sum(row["logical_api_calls"] for row in current)}), flush=True)
    return 0 if summary["decision"] == "select_zero_default" else 2


if __name__ == "__main__":
    raise SystemExit(main())

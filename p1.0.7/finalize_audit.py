"""Offline coverage/cost accounting, semantic references, and final gate; no model requests."""
from collections import Counter
import json
from pathlib import Path
import platform
from importlib.metadata import version

from run_denial_audit import base

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "p1.0.6"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")


def main():
    target = base.read_rows(HERE / "runs/targeted-v1/results.jsonl")
    dev = base.read_rows(HERE / "runs/dev-v1/results.jsonl")
    inputs = base.read_rows(OLD / "inputs.jsonl")
    references = base.read_rows(OLD / "reference-v1.1-sensitivity.jsonl")
    original_reference = base.read_rows(OLD / "reference.jsonl")
    by_case = {row["case_id"]: row for row in inputs}
    reference_by_node = {(row["case_id"], row["node_id"]): row for row in original_reference}
    fingerprints = set()
    checks = {"json_response_format": True, "successful_case_schema_and_refs": True,
              "zero_source_reads": True, "compact_no_full_packages": True}
    rows_to_review = []
    for row in target + dev:
        for request in row["requests"]:
            checks["json_response_format"] &= request["response_format"] == {"type": "json_object"}
            if request.get("system_fingerprint"):
                fingerprints.add(request["system_fingerprint"])
        if row["arm"] == "zero":
            checks["zero_source_reads"] &= row["knowledge_provider_calls"] == {"get_candidate": 0, "get_symptoms": 0}
        elif row["arm"] == "compact":
            checks["compact_no_full_packages"] &= row["knowledge_provider_calls"]["get_candidate"] == 0
        if not row["ok"]:
            continue
        for item in row["judgments"]:
            base.TreeNodeJudgment.model_validate(item, strict=True)
            profile = by_case[row["case_id"]]["profile"] if row in dev else next(x["profile"] for x in base.read_rows(HERE / "targeted-denial-set.jsonl") if x["case_id"] == row["case_id"])
            allowed = base.CLINICR._known_evidence_refs(profile)
            checks["successful_case_schema_and_refs"] &= set(item["evidence_refs"]) <= allowed
            if row in dev and item["status"] != "pending":
                narratives = {ref: item["value"] for item in profile["key_narratives"] for ref in item["sources"]}
                reference = reference_by_node[(row["case_id"], item["node_id"])]
                rows_to_review.append({"case_id": row["case_id"], "arm": row["arm"], **item,
                                       "reference_status": reference["reference_status"], "reference_support": reference["support"],
                                       "cited_quotes": [{"ref": ref, "quote": narratives.get(ref)} for ref in item["evidence_refs"]]})
    checks["known_fingerprints_compatible"] = fingerprints == {"e72d3c7f663d3e1383b00b3884cb4501"}
    accounting = {}
    for suite, records in [("targeted", target), ("dev", dev)]:
        accounting[suite] = {}
        for arm in sorted({row["arm"] for row in records}):
            arm_rows = [row for row in records if row["arm"] == arm]
            requests = [request for row in arm_rows for request in row["requests"]]
            missing = [request for request in requests if request.get("usage") is None]
            costs = {key: {"known_sum": sum((request.get("usage") or {}).get(key, 0) for request in requests),
                           "missing_request_usage": len(missing),
                           "total": None if missing else sum(request["usage"][key] for request in requests)}
                     for key in ["prompt_tokens", "completion_tokens", "total_tokens"]}
            accounting[suite][arm] = {
                "attempted_cases": len(arm_rows), "successful_cases": sum(row["ok"] for row in arm_rows),
                "complete_nodes": sum(len(row["judgments"]) for row in arm_rows if row["ok"]),
                "logical_requests": len(requests), "structured_retries": sum(row["structured_retry_count"] for row in arm_rows),
                "initial_logical_requests": sum(not request["is_structured_retry"] for request in requests),
                "costs": costs,
                "response_models": sorted({request["response_model"] for request in requests if request.get("response_model")}),
                "fingerprints": sorted({request["system_fingerprint"] for request in requests if request.get("system_fingerprint")}),
                "failures": [{"case_id": row["case_id"], "error_type": row["error_type"], "status_code": row.get("status_code")} for row in arm_rows if not row["ok"]],
            }
    dev_summary = read(HERE / "runs/dev-v1/summary.json")
    for arm, info in dev_summary["arms"].items():
        info.pop("node_denominator", None)
        info.pop("case_denominator", None)
        info.update(expected_node_denominator=180, expected_case_denominator=20,
                    observed_node_denominator=accounting["dev"][arm]["complete_nodes"])
    for arm, comparison in dev_summary["comparisons"].items():
        if accounting["dev"][arm]["successful_cases"] != 20:
            comparison["no_false_elimination_pass"] = None
    # Keep the original successful-run reference counts; sensitivity is separate.
    sensitivity = {}
    for arm in ["full", "zero", "compact"]:
        counts = Counter()
        successful = [row for row in dev if row["arm"] == arm and row["ok"]]
        for row in successful:
            counts.update(base.error_counts([item for item in references if item["case_id"] == row["case_id"]], row["judgments"]))
        sensitivity[arm] = {"successful_cases": len(successful), "counts_on_observed_cases": dict(counts),
                            "complete_20_case_comparison": len(successful) == 20}
    result = {"completed_at": base.now(), "checks": checks, "accounting": accounting,
              "targeted_gate": read(HERE / "runs/targeted-v1/summary.json")["gate"],
              "zero_dev_gate": dev_summary["comparisons"]["zero"]["gate"],
              "compact_dev_gate": dev_summary["comparisons"]["compact"]["gate"],
              "final_gate": "pass" if all(checks.values()) and dev_summary["gate"] == "pass" and read(HERE / "runs/targeted-v1/summary.json")["gate"] == "pass" else "hold",
              "default_change_authorized_by_gate": False, "sensitivity_v1_1": sensitivity,
              "python": platform.python_version(), "openai": version("openai"), "pydantic": version("pydantic"), "pytest": version("pytest")}
    result["default_change_authorized_by_gate"] = result["final_gate"] == "pass"
    write(HERE / "final-audit.json", result)
    write(HERE / "runs/dev-v1/summary.json", dev_summary)
    (HERE / "nonpending-citation-review.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows_to_review), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["targeted_gate", "zero_dev_gate", "compact_dev_gate", "final_gate"]}))


if __name__ == "__main__":
    main()

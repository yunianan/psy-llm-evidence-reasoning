"""Offline paired disagreement and sensitivity analysis for the one fixed retest."""
from collections import Counter
import json
from pathlib import Path

from run_denial_audit import base

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/compact-retest-v1"


def main():
    records = base.read_rows(RUN / "results.jsonl")
    references = base.read_rows(HERE.parent / "p1.0.6/reference.jsonl")
    sensitivity = base.read_rows(HERE.parent / "p1.0.6/reference-v1.1-sensitivity.jsonl")
    by_pair = {(row["case_id"], row["arm"]): row for row in records}
    rows = []
    missed_by_node = {arm: Counter() for arm in ["full", "zero", "compact"]}
    for reference in references:
        case_id, node_id = reference["case_id"], reference["node_id"]
        predictions = {arm: next(item for item in by_pair[(case_id, arm)]["judgments"] if item["node_id"] == node_id)
                       for arm in ["full", "zero", "compact"] if by_pair[(case_id, arm)]["ok"]}
        for arm, item in predictions.items():
            if reference["reference_status"] == "met" and item["status"] != "met":
                missed_by_node[arm][node_id] += 1
        if len(predictions) == 3 and len({item["status"] for item in predictions.values()}) > 1:
            rows.append({"case_id": case_id, "node_id": node_id,
                         "reference_status": reference["reference_status"], "reference_support": reference["support"],
                         "predictions": predictions})
    counts = {}
    for arm in ["full", "zero", "compact"]:
        values = Counter()
        arm_rows = [row for row in records if row["arm"] == arm and row["ok"]]
        for row in arm_rows:
            values.update(base.error_counts([item for item in sensitivity if item["case_id"] == row["case_id"]], row["judgments"]))
        counts[arm] = {"successful_cases": len(arm_rows), "errors": dict(values),
                       "undercall_by_node_on_original_reference": dict(missed_by_node[arm])}
    analysis = {"completed_at": base.now(), "reference_sensitivity": "v1.1, post-hoc errata; original counts unchanged",
                "sensitivity": counts,
                "reference_met_full_met_compact_pending": sum(row["reference_status"] == "met" and row["predictions"]["full"]["status"] == "met" and row["predictions"]["compact"]["status"] == "pending" for row in rows),
                "reference_met_full_pending_compact_met": sum(row["reference_status"] == "met" and row["predictions"]["full"]["status"] == "pending" and row["predictions"]["compact"]["status"] == "met" for row in rows),
                "repair_case_ids": [row["case_id"] for row in records if row["arm"] == "compact" and row["structured_retry_count"]]}
    (RUN / "paired-disagreements.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding="utf-8")
    (RUN / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

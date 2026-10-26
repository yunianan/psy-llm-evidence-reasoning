"""Post-run source/quote review exports; retain original references and gates."""
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1.0.7"))
from run_denial_audit import base

RUN = HERE / "runs/full-vs-anchor-compact-v1"


def main():
    records = base.read_rows(RUN / "results.jsonl")
    references = base.read_rows(HERE / "reference.jsonl")
    inputs = {row["case_id"]: row for row in base.read_rows(HERE.parent / "p1.0.6/inputs.jsonl")}
    by_pair = {(row["case_id"], row["arm"]): row for row in records}
    graph = base.load_candidate_graph()
    symptoms = base.JsonDiagnosticKnowledgeProvider().get_symptoms()
    differences, errors, inconsistency = [], [], []
    for ref in references:
        case_id, node_id = ref["case_id"], ref["node_id"]
        quotes = {key: item["value"] for item in inputs[case_id]["profile"]["key_narratives"] for key in item["sources"]}
        predictions = {}
        for arm in ["full", "compact"]:
            record = by_pair[(case_id, arm)]
            if not record["ok"]:
                continue
            item = next(row for row in record["judgments"] if row["node_id"] == node_id)
            predictions[arm] = {**item, "cited_quotes": [{"ref": key, "quote": quotes.get(key)} for key in item["evidence_refs"]]}
            category = ("false_not_met" if ref["reference_status"] == "met" and item["status"] == "not_met" else
                        "false_pending" if ref["reference_status"] == "met" and item["status"] == "pending" else
                        "false_met" if ref["reference_status"] == "not_met" and item["status"] == "met" else None)
            if category:
                errors.append({"case_id": case_id, "arm": arm, "node_id": node_id, "category": category,
                               "reference_status": ref["reference_status"], "reference_support": ref["support"],
                               "prediction": predictions[arm],
                               "source_symptoms": {sid: symptoms[sid] for sid in sorted(set(item["symptom_ids"]) | {support["symptom_id"] for support in ref["support"]})}})
            if item["status"] != "met":
                pool = set(graph["nodes"][node_id]["symptom_pool"])
                other_witnesses = [{"other_node_id": other["node_id"], "symptom_ids": sorted(set(other["symptom_ids"]) & pool),
                                    "evidence_refs": other["evidence_refs"]}
                                   for other in record["judgments"] if other["status"] == "met" and pool.intersection(other["symptom_ids"])]
                if other_witnesses:
                    inconsistency.append({"case_id": case_id, "arm": arm, "node_id": node_id,
                                          "negative_status": item["status"], "rationale": item["rationale"],
                                          "positive_witnesses_elsewhere": other_witnesses,
                                          "interpretation": "internal inconsistency to review; other LLM mappings are not clinical gold"})
        if len(predictions) == 2 and predictions["full"]["status"] != predictions["compact"]["status"]:
            differences.append({"case_id": case_id, "node_id": node_id, "reference_status": ref["reference_status"],
                                "reference_support": ref["support"], "explicit_symptom_denials": ref["explicit_symptom_denials"],
                                "symptom_pool": graph["nodes"][node_id]["symptom_pool"], "predictions": predictions})
    for name, rows in [("path-disagreements", differences), ("reference-error-review", errors), ("cross-node-consistency-review", inconsistency)]:
        (RUN / (name + ".jsonl")).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding="utf-8")
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    result = {"reviewed_at": base.now(), "status_disagreements": len(differences),
              "full_met_compact_nonmet": sum(row["predictions"]["full"]["status"] == "met" and row["predictions"]["compact"]["status"] != "met" for row in differences),
              "full_nonmet_compact_met": sum(row["predictions"]["full"]["status"] != "met" and row["predictions"]["compact"]["status"] == "met" for row in differences),
              "errors_by_arm_category": {arm: dict(Counter(row["category"] for row in errors if row["arm"] == arm)) for arm in ["full", "compact"]},
              "false_not_met_case_ids": {arm: sorted({row["case_id"] for row in errors if row["arm"] == arm and row["category"] == "false_not_met"}) for arm in ["full", "compact"]},
              "cross_node_inconsistency_nodes": {arm: sum(row["arm"] == arm for row in inconsistency) for arm in ["full", "compact"]},
              "repair_case_ids": {arm: [row["case_id"] for row in records if row["arm"] == arm and row["structured_retry_count"]] for arm in ["full", "compact"]},
              "input_token_reduction_percent": (1 - summary["arms"]["compact"]["prompt_tokens"]["sum"] / summary["arms"]["full"]["prompt_tokens"]["sum"]) * 100,
              "reference_and_gate_unchanged": True}
    (RUN / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

"""Offline staged L3 metrics against dataset reference labels; no model calls."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.disorder_catalog import DISORDERS, SUPPORTED_DISORDERS
from tools.mentalbench_partition import family_key
from tools.prediction_contract import normalize_prediction as _normalize_prediction


def _basic_metrics(gold, predictions, top_k=5):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    def keyed(rows):
        result = {row["case_id"]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate case IDs")
        return result
    targets, outputs = keyed(gold), keyed(predictions)
    if not targets or set(targets) != set(outputs):
        raise ValueError("Prediction IDs must exactly match non-empty gold set; missing cases cannot be dropped")
    true_positive = predicted = relevant = top_true = hits = empty = failures = 0
    by_label = {label: {"gold": 0, "retrieved": 0, "predicted": 0} for label in SUPPORTED_DISORDERS}
    for case_id, row in targets.items():
        truth = set(row["gold_labels"])
        candidates = outputs[case_id]["candidates"]
        labels = [item["disorder"] for item in candidates]
        if not truth or not truth <= set(SUPPORTED_DISORDERS) or not set(labels) <= set(SUPPORTED_DISORDERS):
            raise ValueError("Gold/predictions must use supported labels")
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate candidate labels")
        if any(not isinstance(item["confidence"], (int, float)) or not math.isfinite(item["confidence"]) or not 0 <= item["confidence"] <= 1 for item in candidates):
            raise ValueError("Confidence must be finite and between zero and one")
        ranked = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
        chosen = set(labels)
        top = {item["disorder"] for item in ranked[:top_k]}
        true_positive += len(truth & chosen)
        predicted += len(chosen)
        relevant += len(truth)
        top_true += len(truth & top)
        hits += bool(truth & top)
        empty += not chosen
        failures += outputs[case_id].get("model_failure", False)
        for label in truth: by_label[label]["gold"] += 1
        for label in chosen: by_label[label]["predicted"] += 1
        for label in truth & chosen: by_label[label]["retrieved"] += 1
    for counts in by_label.values():
        counts["recall"] = counts["retrieved"] / counts["gold"] if counts["gold"] else None
    return {"cases": len(targets), "top_k": top_k,
            "candidate_ppv_micro": true_positive / predicted if predicted else None,
            "candidate_recall_micro": true_positive / relevant,
            "top_k_recall_micro": top_true / relevant, "top_k_case_hit_rate": hits / len(targets),
            "empty_candidate_rate": empty / len(targets), "model_failure_cases": failures,
            "model_failure_rate": failures / len(targets), "per_label": by_label,
            "test_label_coverage": sum(c["gold"] > 0 for c in by_label.values()) / len(SUPPORTED_DISORDERS)}




def _cluster_intervals(gold, outputs, top_k, group_field, bootstrap, seed):
    groups = defaultdict(lambda: [0] * 5)
    for row in gold:
        truth = set(row["gold_labels"])
        candidates = sorted(outputs[row["case_id"]]["candidates"], key=lambda item: item["confidence"], reverse=True)
        chosen = {item["disorder"] for item in candidates}
        top = {item["disorder"] for item in candidates[:top_k]}
        source = row.get("related_case_group", row["case_id"])
        key = family_key(row["source_code"]) if group_field == "supergroup" and row.get("source_code") else source
        values = [len(truth & chosen), len(chosen), len(truth), len(truth & top), int(bool(truth & top))]
        groups[key] = [a + b for a, b in zip(groups[key], values)]
    result = {"clusters": len(groups), "bootstrap_replicates": bootstrap, "intervals_95": None}
    if bootstrap < 1 or len(groups) < 2:
        return result
    rng, values, metrics = random.Random(seed), list(groups.values()), defaultdict(list)
    for _ in range(bootstrap):
        total = [0] * 5
        for _ in values:
            total = [a + b for a, b in zip(total, rng.choice(values))]
        tp, predicted, relevant, top_tp, _hits = total
        for key, value in {"candidate_ppv_micro": tp / predicted if predicted else None,
                           "candidate_recall_micro": tp / relevant if relevant else None,
                           "top_k_recall_micro": top_tp / relevant if relevant else None}.items():
            if value is not None: metrics[key].append(value)
    result["intervals_95"] = {}
    for key, data in metrics.items():
        data.sort()
        result["intervals_95"][key] = [data[int((len(data) - 1) * .025)], data[int((len(data) - 1) * .975)]]
    return result


def _stage_metrics(gold, predictions, top_k, bootstrap, seed, allow_final):
    result = _basic_metrics(gold, predictions, top_k)
    outputs = {row["case_id"]: row for row in predictions}
    expected = list(DISORDERS) if all(row.get("label_origin") == "MentalBench_synthetic_graph_grounded" for row in gold) else list(SUPPORTED_DISORDERS)
    counts = result["per_label"]
    supported = [value for value in counts.values() if value["gold"]]
    for value in counts.values():
        value["precision"] = value["retrieved"] / value["predicted"] if value["predicted"] else None
        denominator = value["gold"] + value["predicted"]
        value["f1"] = 2 * value["retrieved"] / denominator if denominator else None
    result.update({"candidate_precision_macro": sum(v["precision"] or 0 for v in supported) / len(supported),
                   "candidate_recall_macro": sum(v["recall"] for v in supported) / len(supported),
                   "candidate_f1_macro": sum(v["f1"] for v in supported) / len(supported),
                   "macro_scope": "gold-supported labels; no predictions gives zero macro precision",
                   "reference_label_space": expected,
                   "test_label_coverage": sum(counts[label]["gold"] > 0 for label in expected) / len(expected),
                   "system_label_coverage": len(supported) / len(SUPPORTED_DISORDERS),
                   "uncovered_system_labels": [label for label, value in counts.items() if not value["gold"]],
                   "out_of_reference_space_predictions": {label: value["predicted"] for label, value in counts.items() if label not in expected and value["predicted"]},
                   "outside_prediction_policy": "retained in strict annotation PPV; not proof of clinical absence"})
    assessed = removed = step2_cases = final_cases = final_hit = final_exact = 0
    for row in gold:
        output, truth = outputs[row["case_id"]], set(row["gold_labels"])
        if "step2_decisions" in output:
            decisions = output["step2_decisions"]
            if not set(decisions) <= {c["disorder"] for c in output["candidates"]} or any(v not in {"meets_criteria", "watch", "ruled_out"} for v in decisions.values()):
                raise ValueError("Invalid Step 2 decisions")
            relevant = truth & set(decisions)
            step2_cases += 1
            assessed += len(relevant)
            removed += sum(decisions[label] == "ruled_out" for label in relevant)
        if allow_final and "final_diagnoses" in output:
            final = output["final_diagnoses"]
            if len(final) != len(set(final)) or not set(final) <= set(SUPPORTED_DISORDERS):
                raise ValueError("Invalid final diagnoses")
            final_cases += 1
            final_hit += bool(truth & set(final))
            final_exact += truth == set(final)
    result["step2"] = {"cases_with_decisions": step2_cases, "gold_candidates_assessed": assessed,
                       "gold_candidates_ruled_out": removed, "false_elimination_rate": removed / assessed if assessed else None}
    result["final"] = {"eligible_reference_kind": allow_final, "cases_with_outputs": final_cases,
                       "reference_hit_rate": final_hit / final_cases if final_cases else None,
                       "exact_match_rate": final_exact / final_cases if final_cases else None}
    result["primary_cluster_intervals"] = _cluster_intervals(gold, outputs, top_k, "supergroup", bootstrap, seed)
    result["source_group_sensitivity_intervals"] = _cluster_intervals(gold, outputs, top_k, "source_group", bootstrap, seed)
    result["cluster_summary"] = {"records": len(gold), "source_groups": len({r.get("related_case_group", r["case_id"]) for r in gold}),
        "supergroups": result["primary_cluster_intervals"]["clusters"], "effective_sample_size": "not inferred from group counts"}
    return result


def evaluate(gold, predictions, top_k=5, *, bootstrap=0, seed=20261015):
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    predictions = [_normalize_prediction(row) for row in predictions]
    kinds = defaultdict(list)
    for row in gold: kinds[row.get("reference_kind", "unspecified")].append(row)
    result = _stage_metrics(gold, predictions, top_k, bootstrap, seed, set(kinds) <= {"single_reference_label", "unspecified"})
    result["by_reference_kind"] = {}
    result["by_task_type"] = {}
    for kind, rows in kinds.items():
        ids = {row["case_id"] for row in rows}
        result["by_reference_kind"][kind] = _stage_metrics(rows, [r for r in predictions if r["case_id"] in ids], top_k, bootstrap, seed, kind == "single_reference_label")
    for task_type in sorted({row["task_type"] for row in gold if row.get("task_type")}):
        rows = [row for row in gold if row.get("task_type") == task_type]
        ids = {row["case_id"] for row in rows}
        result["by_task_type"][task_type] = _stage_metrics(rows, [r for r in predictions if r["case_id"] in ids], top_k, bootstrap, seed, task_type != "type3")
    result["pooled_metrics_comparable"] = len(kinds) == 1
    if len(kinds) > 1:
        for key in ["candidate_ppv_micro", "candidate_recall_micro", "top_k_recall_micro", "top_k_case_hit_rate",
                    "candidate_precision_macro", "candidate_recall_macro", "candidate_f1_macro"]:
            result[key] = None
        result["primary_cluster_intervals"] = result["source_group_sensitivity_intervals"] = None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    print(json.dumps(evaluate(read(args.gold), read(args.predictions), args.top_k, bootstrap=args.bootstrap, seed=args.seed), ensure_ascii=False, indent=2))

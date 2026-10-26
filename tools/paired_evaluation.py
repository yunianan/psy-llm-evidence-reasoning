"""Exact v1.1 paired supergroup evaluation. All inputs are local prediction rows."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prediction_contract import KINDS, keyed, normalize_for_protocol, validate_gold
from tools.mentalbench_partition import family_key

VERSION = "paired-evaluation-v1.1"
METRICS = ("candidate_recall_micro", "candidate_ppv_micro", "top_k_recall_micro",
           "final.reference_hit_rate", "candidate_count_mean", "empty_candidate_rate")


def ratio(a, b):
    return Fraction(a, b) if b else None


def exact(value):
    return None if value is None else str(value)


def shared_draws(groups, *, n_boot=1000, seed=20261015):
    """One explicit index matrix, consumed by both arms of every replicate."""
    rng = random.Random(seed)
    return [[rng.randrange(groups) for _ in range(groups)] for _ in range(n_boot)] if groups >= 2 else []


def _vector(gold, prediction, top_k):
    truth = set(gold["gold_labels"])
    labels = [c["disorder"] for c in prediction["candidates"]]
    return [len(truth & set(labels)), len(labels), len(truth), len(truth & set(labels[:top_k])),
            1, int(not labels), int(bool(truth & set(prediction["final_diagnoses"])))]


def _sum(vectors):
    return [sum(column) for column in zip(*vectors)] if vectors else [0] * 7


def _metrics(v, kind):
    h, p, g, t, n, empty, final = v
    return dict(zip(METRICS, (ratio(h, g), ratio(h, p), ratio(t, g),
        ratio(final, n) if kind == "single_reference_label" else None, ratio(p, n), ratio(empty, n))))


def _counts(v):
    return dict(zip(("hits", "predicted", "gold", "top5_hits", "cases", "empty", "final_hits"), v))


def _quality(ids, x, y):
    n = len(ids)
    a = sum(x[k]["complete"] for k in ids)
    b = sum(y[k]["complete"] for k in ids)
    both = sum(x[k]["complete"] and y[k]["complete"] for k in ids)
    passed = bool(n) and min(Fraction(a, n), Fraction(b, n), Fraction(both, n)) >= Fraction(95, 100)
    return {"cases": n, "complete_x": a, "complete_y": b, "joint_complete": both,
            "rate_x": exact(ratio(a, n)), "rate_y": exact(ratio(b, n)),
            "joint_rate": exact(ratio(both, n)), "passed": passed}


def _component(ci, margin):
    return None if ci is None else Fraction(ci[0]) > margin


def iut_verdict(recall_ci, ppv_ci):
    recall, ppv = _component(recall_ci, Fraction(0)), _component(ppv_ci, Fraction(-1, 100))
    verdict = "not_estimable" if recall is None or ppv is None else "win" if recall and ppv else "not_win"
    return {"recall_superior": recall, "ppv_noninferior": ppv, "verdict": verdict}


def _compare(gold, x, y, *, kind, hypothesis, top_k, n_boot, seed, plan=None):
    ids = sorted(k for k, row in gold.items() if row["reference_kind"] == kind)
    groups = sorted({family_key(gold[k]["source_code"]) for k in ids})
    vectors = {"x": {k: _vector(gold[k], x[k], top_k) for k in ids},
               "y": {k: _vector(gold[k], y[k], top_k) for k in ids}}
    counts = {arm: _sum(list(rows.values())) for arm, rows in vectors.items()}
    points = {arm: _metrics(value, kind) for arm, value in counts.items()}
    grouped = {arm: [_sum([rows[k] for k in ids if family_key(gold[k]["source_code"]) == group])
                     for group in groups] for arm, rows in vectors.items()}
    draws = shared_draws(len(groups), n_boot=n_boot, seed=seed) if plan is None else plan
    samples = {metric: [] for metric in METRICS}
    undefined = {metric: 0 for metric in METRICS}
    for indices in draws:
        # Both arms consume the very same indices, not separate calls to random.
        metrics = {arm: _metrics(_sum([rows[i] for i in indices]), kind) for arm, rows in grouped.items()}
        for metric in METRICS:
            left, right = metrics["x"][metric], metrics["y"][metric]
            if left is None or right is None:
                undefined[metric] += 1
            else:
                samples[metric].append(right - left)
    deltas = {}
    for metric in METRICS:
        a, b = points["x"][metric], points["y"][metric]
        values = sorted(samples[metric])
        ci = [exact(values[24]), exact(values[974])] if len(values) == n_boot and not undefined[metric] else None
        deltas[metric] = {"point": exact(b - a) if a is not None and b is not None else None,
                          "ci95": ci, "undefined_replicates": undefined[metric]}
    iut = iut_verdict(deltas[METRICS[0]]["ci95"], deltas[METRICS[1]]["ci95"])
    top5 = _component(deltas[METRICS[2]]["ci95"], Fraction(0))
    statistical = iut["verdict"] if hypothesis == "H1" else "not_estimable" if top5 is None else "win" if top5 else "not_win"
    layer_quality, all_quality = _quality(ids, x, y), _quality(sorted(gold), x, y)
    quality = layer_quality["passed"] and all_quality["passed"]
    status = "quality_gate_failed" if not quality else statistical
    failures = {arm: {"model_failure_cases": sum(rows[k]["model_failure"] for k in ids),
                       "classes": dict(sorted(Counter(rows[k]["failure_class"] for k in ids if not rows[k]["complete"]).items()))}
                for arm, rows in (("x", x), ("y", y))}
    return {"version": VERSION, "kind": kind, "hypothesis": hypothesis,
            "counts": {arm: _counts(value) for arm, value in counts.items()},
            "metrics": {arm: {m: exact(v) for m, v in values.items()} for arm, values in points.items()},
            "deltas": deltas, "sampling": {"unit": "supergroup", "groups": groups,
                "replicates": len(draws), "requested_replicates": n_boot, "seed": seed,
                "shared_across_arms": True, "quantile_indices": [24, 974]},
            "quality": {"layer": layer_quality, "all_qualified": all_quality},
            "quality_gate_passed": quality, "tests": {"iut": iut, "top5_superior": top5},
            "point_estimate_status": "empty_layer" if not ids else "ppv_undefined" if any(points[a][METRICS[1]] is None for a in points) else "computed",
            "statistical_verdict": statistical, "inference_status": status,
            "confirmatory_eligible": kind == "single_reference_label" and quality,
            "failures": failures}


def compare(gold_rows, arm_x_rows, arm_y_rows, *, kind, top_k=5, n_boot=1000, seed=20261015, hypothesis="H1"):
    """Compare y minus x, validating the complete common set before stratification."""
    if kind not in KINDS or hypothesis not in {"H1", "H2", "H3"}:
        raise ValueError("Unknown stratum or hypothesis")
    if (top_k, n_boot, seed) != (5, 1000, 20261015):
        raise ValueError("v1.1 fixes K=5, B=1000 and seed=20261015")
    gold, left, right = validate_gold(gold_rows), keyed(arm_x_rows), keyed(arm_y_rows)
    if set(gold) != set(left) or set(gold) != set(right):
        raise ValueError("Both prediction ID sets must exactly equal qualified gold")
    x, y = ({k: normalize_for_protocol(row) for k, row in rows.items()} for rows in (left, right))
    return _compare(gold, x, y, kind=kind, hypothesis=hypothesis, top_k=top_k, n_boot=n_boot, seed=seed)


def decide(comparisons):
    """Fixed H1→H2→H3 gate. Raw later intervals remain reportable."""
    opened, result = True, {}
    for name in ("H1", "H2", "H3"):
        comparison = comparisons.get(name)
        if comparison is None:
            result[name] = "not_run"
            opened = False
        elif not opened:
            result[name] = "exploratory_gate_closed"
        elif not comparison["confirmatory_eligible"]:
            result[name] = "quality_gate_failed" if not comparison["quality_gate_passed"] else "secondary_stratum"
            opened = False
        else:
            result[name] = comparison["inference_status"]
            opened = result[name] == "win"
    return {"gate_order": ["H1", "H2", "H3"], "verdicts": result}


def evaluate_study(gold_rows, arms, *, kind):
    """All available arms share one bootstrap plan, including across comparisons."""
    gold = validate_gold(gold_rows)
    normalized = {}
    for arm, rows in arms.items():
        indexed = keyed(rows)
        if set(indexed) != set(gold):
            raise ValueError("All arms must have the common qualified ID set")
        normalized[arm] = {k: normalize_for_protocol(v) for k, v in indexed.items()}
    if kind not in KINDS:
        raise ValueError("Unknown stratum")
    groups = {family_key(row["source_code"]) for row in gold.values() if row["reference_kind"] == kind}
    plan = shared_draws(len(groups))
    comparisons = {}
    for name, left, right in (("H1", "A0", "A1"), ("H2", "A1", "A2"), ("H3", "A2", "A3")):
        if left in normalized and right in normalized:
            comparisons[name] = _compare(gold, normalized[left], normalized[right], kind=kind, hypothesis=name,
                                         top_k=5, n_boot=1000, seed=20261015, plan=plan)
    return {"version": VERSION, "kind": kind, "comparisons": comparisons, "decision": decide(comparisons)}


def monitor_block(rows):
    """Terminal-case rates only; recovered attempt errors do not count."""
    if not rows or len(rows) > 20:
        raise ValueError("Monitoring block must contain 1..20 terminal cases")
    keyed(rows)
    classes = [r.get("failure_class", r.get("failure_kind")) or ("model_or_contract" if r.get("model_failure") else None) for r in rows]
    contract = sum(x in {"contract", "model", "model_or_contract", "contract_failure"} for x in classes)
    transport = sum(x in {"transport", "transport_or_http"} for x in classes)
    return {"cases": len(rows), "contract": contract, "transport": transport,
            "contract_stop": Fraction(contract, len(rows)) > Fraction(1, 10),
            "transport_stop": Fraction(transport, len(rows)) > Fraction(15, 100),
            "stop": Fraction(contract, len(rows)) > Fraction(1, 10) or Fraction(transport, len(rows)) > Fraction(15, 100)}


def markdown(result):
    lines = ["# v1.1 离线配对评分", "", f"分层：{result['kind']}；方向：y−x。", "",
             "| 指标 | x | y | 差值 | 95% CI |", "|---|---|---|---|---|"]
    for metric in METRICS:
        delta = result["deltas"][metric]
        lines.append(f"| {metric} | {result['metrics']['x'][metric]} | {result['metrics']['y'][metric]} | {delta['point']} | {delta['ci95']} |")
    lines += ["", f"质量门：{result['quality_gate_passed']}；统计判定：{result['statistical_verdict']}；推断状态：{result['inference_status']}。",
              "", "这是离线评分；工程示例不构成正式EVAL结果。", ""]
    return "\n".join(lines)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("arm_x", type=Path)
    parser.add_argument("arm_y", type=Path)
    parser.add_argument("--kind", choices=KINDS, default="single_reference_label")
    parser.add_argument("--hypothesis", choices=["H1", "H2", "H3"], default="H1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(read_rows(args.gold), read_rows(args.arm_x), read_rows(args.arm_y), kind=args.kind, hypothesis=args.hypothesis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

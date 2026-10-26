"""Build static fixtures from hand-declared count vectors, not scorer output.

The independent oracle uses only stdlib Fraction/random and those vectors;
it never imports paired_evaluation or the normalization under test.
"""
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2] / "L1-L5系统/tests/fixtures/paired"
METRICS = ("candidate_recall_micro", "candidate_ppv_micro", "top_k_recall_micro",
           "final.reference_hit_rate", "candidate_count_mean", "empty_candidate_rate")


def row(case, labels, *, final=(), confidence=None, failure=None):
    return {"case_id": case, "candidates": [{"disorder": d, "confidence": (confidence or {}).get(d, .8)} for d in labels],
            "final_diagnoses": list(final), "model_failure": failure in {"contract", "transport"},
            "run_failure": failure is not None, "failure_class": failure}


def gold(case, label, group=None):
    return {"case_id": case, "gold_labels": [label], "reference_kind": "single_reference_label", "source_code": group or label}


def sums(vectors):
    return [sum(v[j] for v in vectors) for j in range(7)]


def ratios(v):
    h, p, g, t, n, empty, final = v
    return [F(h, g) if g else None, F(h, p) if p else None, F(t, g) if g else None,
            F(final, n) if n else None, F(p, n) if n else None, F(empty, n) if n else None]


def text(value):
    return None if value is None else str(value)


def oracle(g, x, y, vx, vy):
    groups = sorted({"-".join(sorted(r["source_code"].split("-"))) for r in g})
    totals = {"x": sums(vx), "y": sums(vy)}
    point = {a: ratios(v) for a, v in totals.items()}
    by_group = {a: [sums([v for r, v in zip(g, vectors) if "-".join(sorted(r["source_code"].split("-"))) == key])
                    for key in groups] for a, vectors in (("x", vx), ("y", vy))}
    series, missing = [[] for _ in METRICS], [0] * 6
    rng = random.Random(20261015)
    if len(groups) >= 2:
        for _ in range(1000):
            selected = [rng.choice(range(len(groups))) for _ in groups]
            left = ratios(sums([by_group["x"][i] for i in selected]))
            right = ratios(sums([by_group["y"][i] for i in selected]))
            for j in range(6):
                if left[j] is None or right[j] is None: missing[j] += 1
                else: series[j].append(right[j] - left[j])
    deltas = {}
    for j, name in enumerate(METRICS):
        values = sorted(series[j])
        deltas[name] = {"point": text(point["y"][j] - point["x"][j]) if point["x"][j] is not None and point["y"][j] is not None else None,
                        "ci95": [text(values[24]), text(values[974])] if len(values) == 1000 else None,
                        "undefined_replicates": missing[j]}
    rc, pc, tc = (deltas[name]["ci95"] for name in METRICS[:3])
    rpass = None if rc is None else F(rc[0]) > 0
    ppass = None if pc is None else F(pc[0]) > F(-1, 100)
    verdict = "not_estimable" if rpass is None or ppass is None else "win" if rpass and ppass else "not_win"
    n = len(g)
    complete_x = sum(not r["run_failure"] for r in x)
    complete_y = sum(not r["run_failure"] for r in y)
    both = sum(not a["run_failure"] and not b["run_failure"] for a, b in zip(x, y))
    passed = min(F(complete_x, n), F(complete_y, n), F(both, n)) >= F(95, 100)
    quality = {"cases": n, "complete_x": complete_x, "complete_y": complete_y, "joint_complete": both,
               "rate_x": str(F(complete_x, n)), "rate_y": str(F(complete_y, n)), "joint_rate": str(F(both, n)), "passed": passed}
    return {"version": "paired-evaluation-v1.1", "kind": "single_reference_label", "hypothesis": "H1",
        "counts": {a: dict(zip(("hits", "predicted", "gold", "top5_hits", "cases", "empty", "final_hits"), v)) for a, v in totals.items()},
        "metrics": {a: dict(zip(METRICS, map(text, values))) for a, values in point.items()}, "deltas": deltas,
        "sampling": {"unit": "supergroup", "groups": groups, "replicates": 1000 if len(groups) >= 2 else 0,
                     "requested_replicates": 1000, "seed": 20261015, "shared_across_arms": True, "quantile_indices": [24, 974]},
        "quality": {"layer": quality, "all_qualified": quality}, "quality_gate_passed": passed,
        "tests": {"iut": {"recall_superior": rpass, "ppv_noninferior": ppass, "verdict": verdict},
                  "top5_superior": None if tc is None else F(tc[0]) > 0},
        "point_estimate_status": "ppv_undefined" if any(point[a][1] is None for a in point) else "computed",
        "statistical_verdict": verdict, "inference_status": verdict if passed else "quality_gate_failed",
        "confirmatory_eligible": passed,
        "failures": {a: {"model_failure_cases": sum(r["model_failure"] for r in rows),
                         "classes": dict(sorted(Counter(r["failure_class"] for r in rows if r["run_failure"]).items()))}
                     for a, rows in (("x", x), ("y", y))}}


def save(number, g, x, y, vx, vy):
    root = ROOT / f"toy-{number}"
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("gold", g), ("arm_x", x), ("arm_y", y)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    expected = oracle(g, x, y, vx, vy)
    if number == 9:
        def state(contract, transport):
            return {"cases": 20, "contract": contract, "transport": transport, "contract_stop": contract >= 3,
                    "transport_stop": transport >= 4, "stop": contract >= 3 or transport >= 4}
        expected = {"comparison": expected, "monitor_variants": {"contract2": state(2, 0), "contract3": state(3, 0),
                    "transport3": state(0, 3), "transport4": state(0, 4), "recovered_only": state(0, 0)}}
    (root / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    save(1, [gold("toy-1", "D001", "D001-D002")], [row("toy-1", ["D001", "D004"], final=["D001"])],
         [row("toy-1", ["D001"], final=["D001"])], [[1, 2, 1, 1, 1, 0, 1]], [[1, 1, 1, 1, 1, 0, 1]])
    save(2, [gold("toy-2", "D002", "D002-D001")], [row("toy-2", [])], [row("toy-2", ["D002", "D006"], final=["D002"])],
         [[0, 0, 1, 0, 1, 1, 0]], [[1, 2, 1, 1, 1, 0, 1]])
    save(3, [gold("toy-3", "D003")], [row("toy-3", ["D003", "D005"])], [row("toy-3", ["D003", "D007"], final=["D003"])],
         [[1, 2, 1, 1, 1, 0, 0]], [[1, 2, 1, 1, 1, 0, 1]])
    g = [gold("a", "D001"), gold("b", "D002")]
    save(4, g, [row("a", ["D001"]), row("b", ["D002"])], [row("a", ["D001"]), row("b", [], failure="contract")],
         [[1, 1, 1, 1, 1, 0, 0]] * 2, [[1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 0]])
    save(5, g, [row("a", ["D001"]), row("b", [])], [row("a", ["D001"]), row("b", ["D002"])],
         [[1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 0]], [[1, 1, 1, 1, 1, 0, 0]] * 2)
    g, x, y, vx, vy = [], [], [], [], []
    for label in ("D001", "D003"):
        for i in range(6):
            case = f"boundary-{label}-{i}"
            g.append(gold(case, label))
            x.append(row(case, [label, "D010", "D011", "D012"] if i < 3 else []))
            y.append(row(case, [label, "D010", "D011", "D012"] + (["D013"] if i == 5 else [])))
            vx.append([1, 4, 1, 1, 1, 0, 0] if i < 3 else [0, 0, 1, 0, 1, 1, 0])
            vy.append([1, 5 if i == 5 else 4, 1, 1, 1, 0, 0])
    save(6, g, x, y, vx, vy)
    labels = [f"D{i:03d}" for i in range(1, 7)]
    save(7, [gold("rank", "D001")], [row("rank", labels, confidence={"D001": .1})],
         [row("rank", labels, confidence={"D001": .99})], [[1, 6, 1, 0, 1, 0, 0]], [[1, 6, 1, 1, 1, 0, 0]])
    save(8, [gold("s1", "D001"), gold("s2", "D002"), gold("s3", "D003")],
         [row("s1", ["D001"]), row("s2", ["D002"]), row("s3", ["D004"])],
         [row("s1", ["D001", "D005"]), row("s2", ["D002", "D006"]), row("s3", ["D004", "D003"])],
         [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0, 0]], [[1, 2, 1, 1, 1, 0, 0]] * 3)
    g, x, y, vx, vy = [], [], [], [], []
    for i in range(20):
        case, label = f"block-{i}", f"D{i+1:03d}"
        g.append(gold(case, label))
        x.append(row(case, [] if i < 2 else [label], failure="contract" if i < 2 else None))
        y.append(row(case, [] if i < 3 else [label], failure="transport" if i < 3 else None))
        vx.append([0, 0, 1, 0, 1, 1, 0] if i < 2 else [1, 1, 1, 1, 1, 0, 0])
        vy.append([0, 0, 1, 0, 1, 1, 0] if i < 3 else [1, 1, 1, 1, 1, 0, 0])
    save(9, g, x, y, vx, vy)
    print("Wrote nine fixtures from protocol count-vector oracle; no model calls")


if __name__ == "__main__":
    main()

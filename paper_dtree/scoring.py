"""Figure 1 stage 7 and Appendix D: per-case sets, then macro averages."""
from __future__ import annotations


def _metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "PPV": precision, "sensitivity": sensitivity, "F1": f1}


def score_case(predictions, reference):
    if predictions["status"] != "completed" or reference["status"] != "completed":
        raise ValueError("Unresolved reconciliation is not a diagnostic miss; finish stage 6 first")
    if any(row["kind"] == "non_dsm" for row in reference["accepted"]):
        raise ValueError("Author reference diagnoses cannot be unresolved/non-DSM outputs")
    gold = {r["name"] for r in reference["accepted"]}
    predicted = {r["name"] for r in predictions["accepted"] if r["kind"] != "non_dsm"}
    non_dsm = {r["name"] for r in predictions["accepted"] if r["kind"] == "non_dsm"}
    tp, fp, fn = predicted & gold, (predicted - gold) | non_dsm, gold - predicted
    diagnosis = {**_metrics(len(tp), len(fp), len(fn)),
                 "TP_list": sorted(tp), "FP_list": sorted(fp), "FN_list": sorted(fn)}
    gold_cat = {r["category"] for r in reference["accepted"]}
    pred_cat = {r["category"] for r in predictions["accepted"]}
    cat_tp, cat_fp, cat_fn = pred_cat & gold_cat, pred_cat - gold_cat, gold_cat - pred_cat
    category = {**_metrics(len(cat_tp), len(cat_fp), len(cat_fn)),
                "TP_list": sorted(cat_tp), "FP_list": sorted(cat_fp), "FN_list": sorted(cat_fn)}
    return {"status": "completed", "diagnosis": diagnosis, "category": category}


def paired_macro(records):
    """Project policy: only common successful cases; pending/failed are separate."""
    if not isinstance(records, dict) or len(records) < 2:
        raise ValueError("At least two named paired arms are required")
    valid, excluded = {}, {}
    for arm, rows in records.items():
        valid[arm], excluded[arm] = {}, []
        seen = set()
        for row in rows:
            case_id = row["case_id"]
            if case_id in seen:
                raise ValueError("Duplicate case is not a new independent sample")
            seen.add(case_id)
            if row.get("status") == "completed" and row.get("scoring", {}).get("status") == "completed":
                valid[arm][case_id] = row["scoring"]
            else:
                excluded[arm].append({"case_id": case_id, "status": row.get("status", "not_started")})
    common = set.intersection(*(set(cases) for cases in valid.values()))
    macro = {}
    for arm, cases in valid.items():
        excluded[arm].extend({"case_id": case_id, "status": "successful_but_not_common"}
                             for case_id in sorted(set(cases) - common))
        macro[arm] = {granularity: {
            metric: sum(cases[key][granularity][metric] for key in common) / len(common) if common else None
            for metric in ("PPV", "sensitivity", "F1")}
            for granularity in ("diagnosis", "category")}
    return {"common_successful_case_ids": sorted(common), "n": len(common), "macro": macro,
            "excluded": excluded, "aggregation": "mean_of_per_case_metrics"}

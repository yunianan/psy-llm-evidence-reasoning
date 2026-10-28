"""Select 100 criterion-pending records before source-text review; no API calls."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import random
import sys

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parents[1]
sys.path.insert(0, str(WORKSPACE / "L1-L5系统"))
from layers.layer3.diagnosis.decision_tree import criterion_nodes
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider

SEED, SAMPLE_SIZE = 20261015, 100


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    target = BASE / "sampled-items.jsonl"
    if target.exists():
        raise FileExistsError("Do not resample after review")
    source = WORKSPACE / "phase1/p1.4-dev15-case2/scoring/mechanism-audit-v1/pending-reasons.jsonl"
    rows = read_rows(source)
    assert len(rows) == 1115
    strata = defaultdict(list)
    for row in rows:
        strata[row["condition_family"], row["primary_reason"]].append(row)
    counts = {key: min(2, len(group)) for key, group in strata.items()}
    remaining = SAMPLE_SIZE - sum(counts.values())
    residual = sum(len(group) - counts[key] for key, group in strata.items())
    quotas = {key: Fraction(remaining * (len(group) - counts[key]), residual) for key, group in strata.items()}
    additions = {key: int(quota) for key, quota in quotas.items()}
    for key in sorted(strata, key=lambda key: (-(quotas[key] - additions[key]), key))[:remaining - sum(additions.values())]:
        additions[key] += 1
    rng = random.Random(SEED)
    selected = []
    table = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda row: (row["case_id"], row["node_id"]))
        n = counts[key] + additions[key]
        table.append({"condition_family": key[0], "primary_reason": key[1], "population": len(group), "sample": n,
                      "population_weight_per_selected": str(Fraction(len(group), n))})
        for row in rng.sample(group, n):
            selected.append({**row, "stratum_population": len(group), "stratum_sample": n,
                             "population_weight": str(Fraction(len(group), n))})
    selected.sort(key=lambda row: (row["case_id"], row["node_id"]))
    assert len(selected) == SAMPLE_SIZE and len({(row["case_id"], row["node_id"]) for row in selected}) == SAMPLE_SIZE
    provider = JsonDiagnosticKnowledgeProvider()
    definitions = {}
    for row in selected:
        label = row["node_id"].split(".")[1]
        if label not in definitions:
            definitions[label] = criterion_nodes(label, knowledge=provider)
    for i, row in enumerate(selected, 1):
        definition = definitions[row["node_id"].split(".")[1]][row["node_id"]]
        row.update(audit_id=f"P{i:03d}", source_group=definition["source_group"],
                   symptom_pool=definition["symptom_pool"])
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    registration = {"version": "pending-source-review100-v1", "seed": SEED, "population": 1115, "sample": 100,
                    "stratification": ["condition_family", "primary_reason"],
                    "allocation": "min(2,Nh) per occupied stratum, residual proportional largest remainders; lexicographic ties",
                    "source": str(source.relative_to(WORKSPACE)), "strata": table,
                    "sampling_unit": "one case_id + one criterion node with pending; not 100 independent patients",
                    "case_counts": dict(Counter(row["case_id"] for row in selected)),
                    "review_categories": {
                        "missing_required_fact": "Full patient text does not establish the required threshold/core symptom/duration/attribution; unknown does not prove absence.",
                        "evidence_present_not_used": "Patient text explicitly establishes this node's complete requirement, but saved judgment remains pending.",
                        "explicit_counterevidence": "Patient text explicitly denies the relevant requirement or establishes a value below its threshold; not evidence of a missed positive.",
                        "ambiguous_requires_review": "Mapping, source condition or attribution cannot be resolved confidently from the text."},
                    "reporting": "Show raw counts/100 and stratum-weighted point estimates; retain counterevidence/ambiguity outside the two-way comparison.",
                    "reviewer": "assistant source-text review, not independent clinician gold", "model_api_calls": 0, "eval_accessed": False}
    (BASE / "registration.json").write_text(json.dumps(registration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    profiles = read_rows(WORKSPACE / "phase1/p1.4-dev15-case2/dev15.inputs.jsonl")
    catalog = provider.get_symptoms()
    lines = ["# 100条pending原文复核材料（抽样先于归类）", ""]
    for case in profiles:
        case_rows = [row for row in selected if row["case_id"] == case["case_id"]]
        if not case_rows:
            continue
        lines += [f"## {case['case_id']}", ""]
        for node in case["profile"]["evidence_nodes"]:
            lines += [f"[{node['evidence_id']}] {node['quote']}", ""]
        for row in case_rows:
            lines += [f"### {row['audit_id']} {row['node_id']}",
                      "condition: " + json.dumps(row["source_condition"], ensure_ascii=False),
                      "group: " + json.dumps(row["source_group"], ensure_ascii=False),
                      "original_rationale: " + row["rationale"],
                      "original_refs: " + json.dumps(row["evidence_refs"], ensure_ascii=False), ""]
    (BASE / "review-material.md").write_text("\n".join(lines), encoding="utf-8")
    ids = sorted({sid for row in selected for sid in row["symptom_pool"]})
    (BASE / "symptom-catalog.json").write_text(json.dumps({sid: catalog[sid] for sid in ids}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"sample": len(selected), "strata": table, "case_counts": registration["case_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

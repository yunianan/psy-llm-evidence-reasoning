"""Select a fixed engineering cohort from previously untouched DEV supergroups."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "L1-L5系统"
sys.path.insert(0, str(REPO))
from tools.mentalbench_partition import family_key


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    output = HERE / "smoke.inputs.jsonl"
    assert not output.exists(), "Do not change the preregistered cohort"
    root = REPO / "artifacts/datasets/mentalbench_split_v2"
    gold = read(root / "development/dev.gold.jsonl")  # Sampling metadata only; not passed to engine.
    inputs = {r["case_id"]: r for r in read(root / "engine/dev.inputs.jsonl")}
    old = {r["case_id"] for r in read(HERE.parent / "p1.0.6/inputs.jsonl")}
    old_groups = {family_key(r["source_code"]) for r in gold if r["case_id"] in old}
    groups = defaultdict(list)
    for row in gold:
        group = family_key(row["source_code"])
        if group not in old_groups:
            groups[group].append(row)
    rng = random.Random(20261015)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected = []
    for index, key in enumerate(keys):
        rows = sorted(groups[key], key=lambda r: r["case_id"])
        rng.shuffle(rows)
        # Equal group allocation plus a deterministic remainder: 50 total.
        selected.extend(rows[:50 // len(keys) + (index < 50 % len(keys))])
    ids = sorted(r["case_id"] for r in selected)
    assert len(ids) == len(set(ids)) == 50 and not set(ids) & old
    output.write_text("".join(json.dumps(inputs[key], ensure_ascii=False) + "\n" for key in ids), encoding="utf-8")
    manifest = {"seed": 20261015, "purpose": "engineering smoke, not formal endpoint scoring", "case_ids": ids,
                "excluded_previous_supergroups": sorted(old_groups), "selected_supergroups": sorted(keys),
                "allocation": dict(Counter(family_key(r["source_code"]) for r in selected)),
                "reference_kind_counts": dict(Counter(r["reference_kind"] for r in selected)),
                "metadata_gold_used_for_sampling_only": True, "gold_labels_passed_to_engine": False,
                "formal_budget_yuan": 300, "execution_pending_platform_unit_prices": True,
                "new_group_cohort_not_a_population_weighted_estimate": True}
    (HERE / "smoke-cohort.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["seed", "allocation", "reference_kind_counts", "formal_budget_yuan", "execution_pending_platform_unit_prices"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

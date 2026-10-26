"""Freeze a fresh DEV-only cohort before inspecting model outcomes."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1.1.1-infrastructure"))
from common import REPO, read, cost
from tools.mentalbench_partition import family_key


def main():
    assert not (HERE / "inputs.jsonl").exists(), "Never overwrite a frozen holdout"
    excluded = set()
    source_files = []
    for directory in HERE.parent.iterdir():
        if not directory.is_dir() or directory == HERE:
            continue
        for file in directory.glob("*inputs.jsonl"):
            source_files.append(str(file))
            excluded.update(r["case_id"] for r in read(file))
    dataset = REPO / "artifacts/datasets/mentalbench_split_v2"
    groups = defaultdict(list)
    for row in read(dataset / "development/dev.gold.jsonl"):
        if row["case_id"] not in excluded:
            groups[family_key(row["source_code"])].append(row)
    rng, selected = random.Random(20261015), []
    keys = sorted(groups)
    rng.shuffle(keys)
    for index, key in enumerate(keys):
        values = sorted(groups[key], key=lambda row: row["case_id"])
        rng.shuffle(values)
        selected += values[:20 // len(keys) + (index < 20 % len(keys))]
    ids = sorted(row["case_id"] for row in selected)
    assert len(ids) == len(set(ids)) == 20 and not set(ids) & excluded
    inputs = {r["case_id"]: r for r in read(dataset / "engine/dev.inputs.jsonl")}
    (HERE / "inputs.jsonl").write_text("".join(json.dumps(inputs[key], ensure_ascii=False) + "\n" for key in ids), encoding="utf-8")
    cost.write(HERE / "cohort.json", {"seed": 20261015, "case_ids": ids, "excluded_case_count": len(excluded),
        "excluded_source_files": source_files, "supergroups": dict(Counter(family_key(row["source_code"]) for row in selected)),
        "gold_used_for_sampling_only": True, "gold_passed_to_engine": False, "eval_opened": False})
    print(json.dumps({"new_holdout_cases": len(ids), "excluded_cases": len(excluded), "supergroups": len(keys)}))


if __name__ == "__main__":
    main()

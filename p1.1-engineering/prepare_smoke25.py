"""Reduce the planned cohort to 25 attempts, preserving four existing failures."""
from collections import Counter, defaultdict
import json
import random

from prepare_smoke import HERE, REPO, read, family_key


def main():
    assert not (HERE / "smoke25-cohort.json").exists()
    original = read(HERE / "smoke.inputs.jsonl")
    original_ids = {r["case_id"] for r in original}
    attempted = {r["case_id"] for r in read(HERE / "runs/cached-full-smoke-v1/results.jsonl") if r["attempted"]}
    assert len(attempted) == 4
    metadata = {r["case_id"]: r for r in read(REPO / "artifacts/datasets/mentalbench_split_v2/development/dev.gold.jsonl") if r["case_id"] in original_ids}
    groups = defaultdict(list)
    for key, row in metadata.items():
        groups[family_key(row["source_code"])].append(key)
    already = Counter(family_key(metadata[key]["source_code"]) for key in attempted)
    assert len(groups) == 8 and sorted(already.values()) == [4]
    selected, rng = set(attempted), random.Random(20261015)
    for group in sorted(groups):
        remaining = sorted(set(groups[group]) - attempted)
        rng.shuffle(remaining)
        selected.update(remaining[:max(0, 3 - already[group])])
    assert len(selected) == 25
    rows = [r for r in original if r["case_id"] in selected]
    pending = [r for r in rows if r["case_id"] not in attempted]
    assert len(pending) == 21
    manifest = {"authorized_total_attempts": 25, "seed": 20261015,
                "purpose": "engineering smoke, no formal diagnosis scoring",
                "selection": "retain four attempted cases; sample three from each other supergroup using metadata only",
                "case_ids": [r["case_id"] for r in rows], "prior_attempted_case_ids": sorted(attempted),
                "remaining_case_ids": [r["case_id"] for r in pending],
                "allocation": dict(Counter(family_key(metadata[key]["source_code"]) for key in selected)),
                "reference_kind_counts": dict(Counter(metadata[key]["reference_kind"] for key in selected)),
                "medical_gold_passed_to_engine": False, "whole_case_retries": 0,
                "official_peak_reference_ceiling_yuan": 30, "actual_platform_price_verified": False}
    (HERE / "smoke25.inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (HERE / "smoke25-cohort.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

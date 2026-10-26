from collections import Counter, defaultdict
import json
import random

from common import HERE, REPO, read, cost
from tools.mentalbench_partition import family_key


def main():
    assert not (HERE / "audit20.inputs.jsonl").exists()
    previous = HERE.parent / "p1.1-engineering"
    plan = json.loads((previous / "smoke25-cohort.json").read_text(encoding="utf-8"))
    records = [r for run in ["cached-full-smoke-v1", "cached-full-smoke25-continuation-v1", "cached-full-smoke25-bounded-v1"]
               for r in read(previous / "runs" / run / "results.jsonl") if r["attempted"]]
    failed = {r["case_id"] for r in records if not r["ok"]}
    assert len(failed) == 9
    all25 = read(previous / "smoke25.inputs.jsonl")
    repair = [r for r in all25 if r["case_id"] in failed]
    (HERE / "repair9.inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in repair), encoding="utf-8")
    excluded = set(plan["case_ids"]) | {r["case_id"] for r in read(HERE.parent / "p1.0.6/inputs.jsonl")}
    dataset = REPO / "artifacts/datasets/mentalbench_split_v2"
    groups = defaultdict(list)
    for row in read(dataset / "development/dev.gold.jsonl"):
        if row["case_id"] not in excluded:
            groups[family_key(row["source_code"])].append(row)
    rng, selected = random.Random(20261015), []
    keys = sorted(groups)
    rng.shuffle(keys)
    for index, key in enumerate(keys):
        values = sorted(groups[key], key=lambda r: r["case_id"])
        rng.shuffle(values)
        selected += values[:20 // len(keys) + (index < 20 % len(keys))]
    ids = sorted(r["case_id"] for r in selected)
    assert len(ids) == 20 and not set(ids) & excluded
    inputs = {r["case_id"]: r for r in read(dataset / "engine/dev.inputs.jsonl")}
    (HERE / "audit20.inputs.jsonl").write_text("".join(json.dumps(inputs[key], ensure_ascii=False) + "\n" for key in ids), encoding="utf-8")
    manifest = {"seed": 20261015, "audit20_case_ids": ids, "repair9_case_ids": sorted(failed),
                "audit20_supergroups": dict(Counter(family_key(r["source_code"]) for r in selected)),
                "gold_metadata_used_for_sampling_only": True, "gold_passed_to_engine": False,
                "formal_scoring": False, "eval_opened": False}
    cost.write(HERE / "cohorts.json", manifest)
    requests = [r for record in records for r in record["requests"]]
    audit = []
    for record in records:
        stage_counts = Counter()
        for request_number, request in enumerate(record["requests"], 1):
            stage_counts[request["stage"]] += 1
            if request.get("error_type"):
                audit.append({"case_id": record["case_id"], "case_request_number": request_number,
                              "stage_batch_number": stage_counts[request["stage"]],
                              **{k: request.get(k) for k in ["stage", "node_ids", "elapsed_seconds", "error_type", "status_code", "transport_attempts", "usage"]},
                              "classification": "client ReadTimeout; server HTTP status unavailable",
                              "historical_response_request_id": None})
    known = [r["usage"] for r in requests if r.get("usage")]
    reasoning = sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for u in known)
    completion = sum(u["completion_tokens"] for u in known)
    cost.write(HERE / "failure-audit.json", {"timeouts": audit, "reasoning_tokens": reasoning,
        "completion_tokens": completion, "reasoning_share": reasoning / completion,
        "legacy_explicit_deepseek_thinking_override": None,
        "budget_failures": [{"case_id": r["case_id"], "kind": r["failure_kind"]} for r in records if r.get("failure_kind") == "reference_budget_stop"],
        "actual_invoice": None, "invoice_status": "provider billing evidence not supplied; generic Aliyun URLs do not establish proxy billing"})
    print(json.dumps({"audit_cases": len(ids), "repair_cases": len(failed), "reasoning_share": reasoning / completion}))


if __name__ == "__main__":
    main()

"""Phase 0 数据前置检查；逐项失败即停止，真实 EVAL 仅调用 is_file()。"""
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1] / "L1-L5系统"
sys.path.insert(0, str(ROOT))
from layers.layer3.api import DiagnoseRequest


DATA = ROOT / "artifacts/datasets/mentalbench_split_v2"
RESULT = Path(__file__).with_name("data-checks.json")
LABELS = {f"D{i:03d}" for i in range(1, 24)}
PROFILE_FIELDS = {
    "key_symptoms_extracted", "symptom_duration_days", "functional_impairment",
    "mania_or_hypomania_history", "substance_or_medical_cause",
    "differential_diagnoses_assessed", "evidence_conflicts", "diagnostic_details",
    "patient_goals", "emotions", "cognitive_patterns", "key_narratives",
    "functional_impairments", "motivation", "evidence_nodes",
}
results = []


def record(name, details):
    results.append({"check": name, "status": "pass", **details})
    RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results[-1], ensure_ascii=False))


def rows(relative, count):
    data = []
    with (DATA / relative).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            assert line.strip(), f"{relative}:{number}: blank line"
            data.append(json.loads(line))
    assert len(data) == count, f"{relative}: expected {count}, got {len(data)}"
    return data


def unique_ids(data):
    ids = [row["case_id"] for row in data]
    assert all(isinstance(i, str) and i.strip() for i in ids), "Invalid case_id"
    assert len(set(ids)) == len(ids), "Duplicate case_id"
    return set(ids)


def inputs(relative, count):
    data = rows(relative, count)
    ids = unique_ids(data)
    for row in data:
        assert set(row) == {"case_id", "profile"}, f"{row['case_id']}: envelope fields"
        profile = row["profile"]
        assert PROFILE_FIELDS <= profile.keys(), f"{row['case_id']}: missing profile fields"
        DiagnoseRequest.model_validate(profile)
        facts = profile["key_narratives"]
        assert isinstance(facts, list) and facts, f"{row['case_id']}: empty narratives"
        nodes = profile["evidence_nodes"]
        evidence_ids = [node["evidence_id"] for node in nodes]
        assert nodes and len(set(evidence_ids)) == len(nodes), f"{row['case_id']}: invalid evidence IDs"
        assert all(isinstance(ref, str) and ref.strip() for ref in evidence_ids)
        for index, fact in enumerate(facts):
            assert isinstance(fact["value"], str) and fact["value"].strip()
            assert fact["sources"] and set(fact["sources"]) <= set(evidence_ids)
            matching = [node for node in nodes if node["evidence_id"] in fact["sources"]]
            assert all(node["target_path"] == f"key_narratives.{index}" for node in matching)
            assert all(node["quote"] == fact["value"] for node in matching)
    return ids


def gold(relative, count, expected_ids):
    data = rows(relative, count)
    assert unique_ids(data) == expected_ids, f"{relative}: input/gold IDs differ"
    for row in data:
        labels = row["gold_labels"]
        assert isinstance(labels, list) and labels and set(labels) <= LABELS
        assert len(labels) == len(set(labels)), f"{row['case_id']}: duplicate label"
    return {label for row in data for label in row["gold_labels"]}


def main():
    raw = subprocess.check_output([
        "git", "-c", "core.quotePath=false", "status", "--porcelain=v1", "-z", "--untracked-files=all",
    ], cwd=ROOT).decode("utf-8")
    entries = [line for line in raw.split("\0") if line]
    nonbusiness = [line for line in entries if line[3:].startswith((
        ".tmp/", "artifacts/developer_profiles/",
    ))]
    business = [line for line in entries if line not in nonbusiness]
    assert not business, f"Uncommitted business files: {business}"
    record("repository", {
        "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "business_changes": 0,
        "historical_nonbusiness_changes": len(nonbusiness),
    })

    dev_ids = inputs("engine/dev.inputs.jsonl", 4719)
    record("dev_inputs", {"records": len(dev_ids), "required_profile_fields": len(PROFILE_FIELDS),
                          "schema_valid": True, "narrative_evidence_refs_valid": True})
    dev_labels = gold("development/dev.gold.jsonl", 4719, dev_ids)
    record("dev_gold", {"records": 4719, "one_to_one": True, "nonempty_labels": True,
                        "label_coverage": len(dev_labels)})

    kb_ids = inputs("engine/kb.inputs.jsonl", 12364)
    assert not kb_ids & dev_ids, "KB/DEV active case IDs overlap"
    record("kb_inputs", {"records": len(kb_ids), "nonempty_narratives": True,
                         "schema_valid": True, "narrative_evidence_refs_valid": True})
    kb_labels = gold("development/kb.gold.jsonl", 12364, kb_ids)
    record("kb_gold", {"records": 12364, "one_to_one": True, "nonempty_labels": True,
                       "labels": sorted(kb_labels), "label_coverage": len(kb_labels)})

    quarantine = rows("development/kb.quarantine.jsonl", 11)
    quarantine_ids = {row["input"]["case_id"] for row in quarantine}
    assert len(quarantine_ids) == 11
    assert not quarantine_ids & (kb_ids | dev_ids), "Quarantine IDs are in active pools"
    record("kb_quarantine", {"records": 11, "overlap_with_active_kb_dev": 0,
                             "pilot_policy": "KB-side samples use active engine/kb.inputs.jsonl and exclude quarantine IDs; pilot is deferred to P0.5/P0.6."})

    # Do not open, hash, decrypt, or recursively inspect the real EVAL bundle.
    assert (DATA / "sealed/eval.bundle.dpapi").is_file(), "Missing sealed EVAL bundle"
    record("eval_seal", {"exists": True, "contents_read": False, "decrypted": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        results.append({"check": "stopped", "status": "fail", "error": str(error)})
        RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results[-1], ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)

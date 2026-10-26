"""Historical G1/A1 replay plus synthetic A0 responses; zero inference calls.

No output here is an actual A0 model performance estimate or a new DEV pilot.
"""
import json
from pathlib import Path
import socket
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / "L1-L5系统"
sys.path.insert(0, str(REPO))

from tools import phase1_research_entry as entry
from tools import paired_evaluation as paired
from tools.a0_no_knowledge import known_patient_refs
from shared.disorder_catalog import disorder_metadata

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/offline-example-v2"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    if RUN.exists():
        raise RuntimeError("Preserve the existing example; use its --resume CLI for repeatability checks")
    def blocked(*args, **kwargs):
        raise AssertionError("All network access disabled for offline example")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    saved = paired.read_rows(WORKSPACE / "phase1/p1.1.1-infrastructure/engineering-repaired25.predictions.jsonl")
    ids = {r["case_id"] for r in saved}
    assert len(saved) == len(ids) == 25
    dataset = REPO / "artifacts/datasets/mentalbench_split_v2"
    inputs = [r for r in paired.read_rows(dataset / "engine/dev.inputs.jsonl") if r["case_id"] in ids]
    gold = [r for r in paired.read_rows(dataset / "development/dev.gold.jsonl") if r["case_id"] in ids]
    assert {r["case_id"] for r in inputs} == {r["case_id"] for r in gold} == ids
    replies = []
    for row in inputs:
        refs = known_patient_refs(row["profile"])
        candidates = [] if not refs else [{"disorder": "D001", "label": disorder_metadata("D001")["label"],
            "symptom_match": {"matched": 0, "total": 1}, "core_evidence_summary": "合成接口夹具，不是对该患者的模型推断",
            "evidence_refs": [refs[0]], "confidence": .4, "decision_paths": []}]
        replies.append({"case_id": row["case_id"], "responses": [json.dumps({"candidates": candidates}, ensure_ascii=False)]})
    data = RUN / "inputs"
    write_rows(data / "dev25.inputs.jsonl", inputs)
    write_rows(data / "dev25.gold.jsonl", gold)
    write_rows(data / "synthetic-a0-responses.jsonl", replies)
    write_rows(data / "historical-a1.predictions.jsonl", saved)
    common = ["--dev-list", str(data / "dev25.inputs.jsonl"), "--out", str(RUN)]
    a0 = ["--arm", "A0", *common, "--a0-responses", str(data / "synthetic-a0-responses.jsonl")]
    a1 = ["--arm", "A1", *common, "--replay-predictions", str(data / "historical-a1.predictions.jsonl"),
          "--g1-cache-dir", str(WORKSPACE / "phase1/p1.1-engineering/runs/cached-full-smoke25-bounded-v1/g1-nodes"),
          "--g1-cache-dir", str(WORKSPACE / "phase1/p1.1.1-infrastructure/runs/repair9-v1/g1-nodes")]
    assert entry.main(a0) == 0
    assert entry.main(a1) == 0
    before = {str(p.relative_to(RUN)): p.read_bytes() for p in (RUN / "runs").glob("*/*.json")}
    assert entry.main([*a0, "--resume"]) == 0
    assert entry.main([*a1, "--resume"]) == 0
    assert before == {str(p.relative_to(RUN)): p.read_bytes() for p in (RUN / "runs").glob("*/*.json")}
    for kind in ("single_reference_label", "admissible_candidates"):
        path = RUN / "scoring" / f"{kind}.json"
        paired.main([str(data / "dev25.gold.jsonl"), str(RUN / "predictions/A0.jsonl"),
                     str(RUN / "predictions/A1.jsonl"), "--kind", kind, "--out", str(path)])
        report = path.with_suffix(".md")
        report.write_text("**人工合成A0＋历史A1，仅验证接口和评分流程，不能解释为真实A0/H1诊断性能。**\n\n" + report.read_text(encoding="utf-8"), encoding="utf-8")
    summary = {"cases": 25, "new_model_calls": 0, "network_disabled": True, "synthetic_a0": True,
               "historical_a1_only": True, "read_only_g1_replays": 25,
               "resume_terminal_files_unchanged": True, "formal_scoring": False, "eval_opened": False}
    entry.write_json(RUN / "verification.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Read-only DEV-15 format annotation: no diagnostic run or gold access."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "L1-L5系统"))
from layers.layer3.diagnosis.information_report import build_information_report, VERSION
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider


def main():
    target = BASE / "dev15-information.jsonl"
    if target.exists():
        raise FileExistsError("Do not overwrite this annotation evidence")
    source = ROOT / "phase1/p1.4-dev15-case2/runs/p1.4-dev15-case2-recovery-v1/predictions/A1.jsonl"
    original_bytes = source.read_bytes()
    rows = [json.loads(line) for line in original_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    provider = JsonDiagnosticKnowledgeProvider()
    outputs, cases, counts, priority_counts = [], [], Counter(), Counter()
    path_count = 0
    for row in rows:
        report = row["report"]
        before = deepcopy(report)
        info = build_information_report(report, knowledge=provider)
        assert report == before
        readiness, plan = info["diagnostic_readiness"], info["information_collection_plan"]
        ranked = sorted(report["candidates"], key=lambda c: (-c["confidence"], c["disorder"]))
        assert plan["candidate_order"] == [c["disorder"] for c in ranked]
        assert len(plan["candidate_order"]) == len(report["candidates"])
        assert set(plan["top_3_candidates"] + plan["remaining_candidates"]) == set(plan["candidate_order"])
        for c in report["candidates"]:
            required = [q for q in plan["items"] if c["disorder"] in q["required_by"]]
            described = next(p for p in readiness["candidates"] if p["disorder"] == c["disorder"])
            assert {q["node_id"] for q in required} == set(described["pending_required_nodes"])
        counts.update(readiness["counts"])
        priority_counts.update(q["priority"] for q in plan["items"])
        path_count += len(report["candidates"])
        cases.append({"case_id": row["case_id"], "status": readiness["status"], "counts": readiness["counts"],
                      "top_1": plan["top_1_candidates"], "top_3": plan["top_3_candidates"],
                      "remaining": plan["remaining_candidates"], "question_count": len(plan["items"]),
                      "original_routing": report["verifier_routing"],
                      "original_final_diagnoses": report["final_diagnoses"]})
        outputs.append({"case_id": row["case_id"], "source_arm": "A1", "format_annotation_only": True, **info})
    assert len(rows) == 15 and path_count == 216
    assert counts == Counter(supported=0, known_blocker=29, insufficient_information=187)
    assert all(c["status"] == "insufficient_information" for c in cases)
    assert source.read_bytes() == original_bytes
    summary = {
        "version": VERSION, "mode": "offline_report_annotation_not_engine_rerun", "cases": 15,
        "retained_candidate_paths": path_count, "path_status_counts": dict(counts),
        "question_priority_counts": dict(priority_counts), "per_case": cases,
        "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "original_report_objects_unchanged": True, "original_prediction_file_unchanged": True,
        "candidate_set_confidence_order_coverage_and_final_routing_unchanged": True,
        "all_remaining_candidate_questions_retained": True, "top3_is_not_candidate_cutoff": True,
        "functional_v1_intervention_activated": False, "new_diagnostic_api_calls": 0,
        "new_diagnostic_api_cost_yuan": 0, "patient_agent_called": False, "gold_read": False, "EVAL_accessed": False,
    }
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")
    (BASE / "dev15-verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["cases", "path_status_counts", "question_priority_counts", "original_prediction_file_unchanged", "new_diagnostic_api_calls"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

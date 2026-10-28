"""Describe existing judgments and order questions without changing diagnosis gates."""
from __future__ import annotations

from collections import Counter
from typing import Any

from shared.diagnostic_knowledge import DiagnosticKnowledgeProvider
from shared.candidate_ordering import candidate_sort_key

from .decision_tree import criterion_nodes, load_candidate_graph

VERSION = "l3-diagnostic-information-v1"
LABELS = {
    "supported": "必需判据已满足",
    "known_blocker": "存在已知不满足条件",
    "insufficient_information": "信息不足",
    "not_evaluated": "尚未进行路径评估",
}


def _priority(rank: int) -> str:
    return "top_1" if rank == 1 else "top_3" if rank <= 3 else "remaining"


def build_information_report(
    result: dict[str, Any], *, knowledge: DiagnosticKnowledgeProvider,
) -> dict[str, dict[str, Any]]:
    """Read candidate paths only; a pending judgment never becomes not_met.

    Optional criteria do not block readiness. Guard judgments are interpreted
    against each path's expected state, rather than against the literal 'met'.
    The original candidate list, coverage, assessments, routing and scores are
    never mutated. No model or patient-agent call is made.
    """
    graph = load_candidate_graph()
    assessed = {a["disorder"]: a for a in result.get("checked", [])}
    ranked = sorted(result.get("candidates", []), key=candidate_sort_key)
    paths, questions = [], {}
    for rank, candidate in enumerate(ranked, 1):
        label = candidate["disorder"]
        definitions = criterion_nodes(label, knowledge=knowledge)
        candidate_paths = [p for p in candidate.get("decision_paths", []) if p["path_id"] == label]
        records = {}
        for path in candidate_paths:
            for record in path["decisions"]:
                if record["node_id"] in records:
                    raise ValueError(f"Duplicate report path record: {record['node_id']}")
                records[record["node_id"]] = record
        pending, blockers, optional_pending = [], [], []
        checks = [(nid, definition, "met") for nid, definition in definitions.items() if definition["mandatory"]]
        optional_pending = [nid for nid, d in definitions.items()
                            if not d["mandatory"] and records.get(nid, {}).get("status", "pending") == "pending"]
        checks += [(g["node_id"], graph["nodes"][g["node_id"]], g["expected"])
                   for g in graph["paths"][label]["guards"]]
        for nid, definition, expected in checks:
            record = records.get(nid)
            raw = record["status"] if record else "pending"
            if raw == "pending":
                pending.append(nid)
                if nid not in questions:
                    questions[nid] = {
                        "field": f"decision_tree.{nid}", "node_id": nid,
                        "purpose": definition["question"], "suggested_question": definition["follow_up"],
                        "stage": definition["stage"], "priority": _priority(rank),
                        "candidate_rank": rank, "required_by": [], "expected_by_candidate": {},
                        "source_refs": list(definition["source_refs"]),
                    }
                questions[nid]["required_by"].append(label)
                questions[nid]["expected_by_candidate"][label] = expected
            elif raw != expected:
                blockers.append({"node_id": nid, "stage": definition["stage"],
                                 "observed_status": raw, "expected_status": expected,
                                 "rationale": record["rationale"], "evidence_refs": list(record["evidence_refs"])})
        record_issues = []
        if not candidate_paths:
            record_issues.append("candidate_path_missing")
        elif any(p["provisional"] for p in candidate_paths) and not pending and not blockers:
            record_issues.append("path_still_marked_provisional")
        assessment = assessed.get(label)
        if not pending and not blockers and (not assessment or assessment["conclusion"] != "meets_criteria"):
            record_issues.append("assessment_not_confirmed")
        status = ("known_blocker" if blockers else "insufficient_information"
                  if pending or record_issues else "supported")
        paths.append({"disorder": label, "label": candidate["label"], "candidate_rank": rank,
                      "status": status, "status_label": LABELS[status],
                      "pending_required_nodes": pending, "known_blockers": blockers,
                      "optional_pending_nodes": optional_pending, "record_issues": record_issues})
    counts = Counter(p["status"] for p in paths)
    excluded = sum(a["conclusion"] == "ruled_out" for a in result.get("checked", []))
    overall = ("supported" if counts["supported"] else "insufficient_information"
               if counts["insufficient_information"] else "known_blocker"
               if counts["known_blocker"] or excluded else "not_evaluated")
    diagnosis = result.get("diagnosis", {})
    readiness = {
        "version": VERSION, "scope": "mandatory_criteria_and_guards",
        "status": overall, "status_label": LABELS[overall],
        "counts": {key: counts[key] for key in ("supported", "known_blocker", "insufficient_information")},
        "candidate_path_count": len(paths), "excluded_path_count": excluded, "candidates": paths,
        "final_diagnosis_reported": bool(diagnosis.get("primary") or diagnosis.get("secondary")),
        "verifier_routing": result.get("verifier_routing"),
        "note": "判据状态不等于最终诊断；最终输出仍须经过原置信度门和Verifier。",
    }
    plan = {
        "version": VERSION, "mode": "planning_only", "priority_window": 3,
        "candidate_order": [c["disorder"] for c in ranked],
        "top_1_candidates": [c["disorder"] for c in ranked[:1]],
        "top_3_candidates": [c["disorder"] for c in ranked[:3]],
        "remaining_candidates": [c["disorder"] for c in ranked[3:]],
        "items": list(questions.values()),
        "note": "Top-1/Top-3只影响补采顺序，不截断候选、不改变判据、不自动执行追问；已知阻塞见diagnostic_readiness。",
    }
    return {"diagnostic_readiness": readiness, "information_collection_plan": plan}


def input_information_report(missing: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """An input guard can report missing fields without touching KG or CLINICR."""
    return {
        "diagnostic_readiness": {
            "version": VERSION, "scope": "input_contract_only", "status": "insufficient_information",
            "status_label": LABELS["insufficient_information"],
            "candidate_path_count": 0, "excluded_path_count": 0, "candidates": [],
            "counts": {key: 0 for key in ("supported", "known_blocker", "insufficient_information")},
            "final_diagnosis_reported": False, "verifier_routing": "return_to_l2",
            "note": "输入字段或证据缺失，尚未进行路径评估；未知不表示病症不存在。",
        },
        "information_collection_plan": {
            "version": VERSION, "mode": "planning_only", "priority_window": 3,
            "candidate_order": [], "top_1_candidates": [], "top_3_candidates": [], "remaining_candidates": [],
            "items": [{**item, "priority": "input_contract", "candidate_rank": None,
                       "required_by": [], "expected_by_candidate": {}} for item in missing],
            "note": "先补充输入字段及患者证据；门禁不会自动产生事实，也未调用诊断模型。",
        },
    }

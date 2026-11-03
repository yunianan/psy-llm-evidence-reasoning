"""Describe existing judgments and order questions without changing diagnosis gates."""
from __future__ import annotations

from collections import Counter
from typing import Any

from shared.diagnostic_knowledge import DiagnosticKnowledgeProvider
from shared.candidate_ordering import candidate_sort_key

from .decision_tree import criterion_nodes, load_candidate_graph

VERSION = "l3-diagnostic-information-v2"
CORE_CONDITIONS = {"min_count", "must_include_one_of", "must_include_all"}
DISPOSITIONS = {
    "evidence_ready": "依据齐备，待终验",
    "needs_follow_up": "待追问",
    "needs_review": "待复核",
}
LABELS = {
    "supported": "必需判据已满足",
    "known_blocker": "存在已知不满足条件",
    "insufficient_information": "信息不足",
    "not_evaluated": "尚未进行路径评估",
}


def _priority(rank: int) -> str:
    return "top_1" if rank == 1 else "top_3" if rank <= 3 else "remaining"


def _coverage(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Count required checks, not independent facts or disease probabilities."""
    total = len(checks)
    counts = Counter(check["state"] for check in checks)
    return {
        "total": total,
        **{f"{state}_count": counts[state] for state in ("confirmed", "unknown", "contradicted")},
        **{f"{state}_ratio": counts[state] / total if total else None
           for state in ("confirmed", "unknown", "contradicted")},
    }


def _evidence_check(nid, definition, expected, record):
    raw = record["status"] if record else "pending"
    refs = list(record.get("evidence_refs", [])) if record else []
    # Live CLINICR has already validated refs against the patient allowlist.
    # An unreferenced known judgment in an archived/direct report is not proof.
    unknown_reason = ("record_missing" if record is None else "node_pending" if raw == "pending"
                      else "patient_evidence_refs_missing" if not refs else None)
    state = "unknown" if unknown_reason else "confirmed" if raw == expected else "contradicted"
    core = definition["stage"] == "criterion" and nid.rsplit("/", 1)[-1] in CORE_CONDITIONS
    check = {
        "node_id": nid, "stage": definition["stage"],
        "kind": "core" if core else "criterion" if definition["stage"] == "criterion" else "guard",
        "observed_status": raw, "expected_status": expected, "state": state,
        "unknown_reason": unknown_reason, "evidence_refs": refs,
        "source_refs": list(definition["source_refs"]),
        "rationale": record.get("rationale") if record else None,
    }
    if definition["stage"] == "criterion":
        check["source_condition"] = definition["source_condition"]
    return check


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
        pending, blockers, evidence_checks = [], [], []
        record_issues = []
        checks = [(nid, definition, "met") for nid, definition in definitions.items() if definition["mandatory"]]
        optional_pending = [nid for nid, d in definitions.items()
                            if not d["mandatory"] and records.get(nid, {}).get("status", "pending") == "pending"]
        checks += [(g["node_id"], graph["nodes"][g["node_id"]], g["expected"])
                   for g in graph["paths"][label]["guards"]]
        for nid, definition, expected in checks:
            record = records.get(nid)
            check = _evidence_check(nid, definition, expected, record)
            evidence_checks.append(check)
            if check["unknown_reason"] == "patient_evidence_refs_missing":
                record_issues.append(f"patient_evidence_refs_missing:{nid}")
            if check["state"] == "unknown":
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
            elif check["state"] == "contradicted":
                blockers.append({"node_id": nid, "stage": definition["stage"],
                                 "observed_status": check["observed_status"], "expected_status": expected,
                                 "rationale": record["rationale"], "evidence_refs": list(record["evidence_refs"])})
        if not candidate_paths:
            record_issues.append("candidate_path_missing")
        elif any(p["provisional"] for p in candidate_paths) and not pending and not blockers:
            record_issues.append("path_still_marked_provisional")
        assessment = assessed.get(label)
        if not pending and not blockers and (not assessment or assessment["conclusion"] != "meets_criteria"):
            record_issues.append("assessment_not_confirmed")
        status = ("known_blocker" if blockers else "insufficient_information"
                  if pending or record_issues else "supported")
        core_checks = [check for check in evidence_checks if check["kind"] == "core"]
        core = _coverage(core_checks)
        disposition = ("needs_review" if blockers else "needs_follow_up" if pending
                       else "needs_review" if record_issues else "evidence_ready")
        paths.append({"disorder": label, "label": candidate["label"], "candidate_rank": rank,
                      "status": status, "status_label": LABELS[status],
                      "disposition": disposition, "disposition_label": DISPOSITIONS[disposition],
                      "anchor_confidence": candidate["confidence"],
                      "evidence_coverage": {
                          "required": _coverage(evidence_checks), "core": core,
                          "core_requirements_confirmed": (
                              core["confirmed_count"] == core["total"] if core["total"] else None),
                          "retained_with_unknowns_only": bool(pending) and not blockers,
                          "checks": evidence_checks,
                      },
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
        "candidate_groups": {group: [p["disorder"] for p in paths if p["disposition"] == group]
                             for group in DISPOSITIONS},
        "evidence_policy": {
            "unit": "required_node_checks", "core_scope": "mandatory_min_count_and_must_include",
            "ready_rule": "all_mandatory_criteria_and_expected_guards_confirmed_with_refs_and_assessment",
            "unknown_rule": "needs_follow_up", "contradiction_rule": "needs_review",
            "ratio_is_diagnostic_probability": False, "raw_candidate_pool_preserved": True,
        },
        "final_diagnosis_reported": bool(diagnosis.get("primary") or diagnosis.get("secondary")),
        "verifier_routing": result.get("verifier_routing"),
        "note": ("比例按现行必需判据与鉴别检查计数，非独立事实数或诊断概率；已证实指既有节点判定有患者引用，"
                 "未重新核验原文。核心为来源规定的必需计数/必含症状条件，可选条件不进分母。"
                 "原candidates为G1召回池；使用candidate_groups区分待追问、待复核与依据齐备，"
                 "比例不替代诊断门、置信度门和Verifier。"),
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
            "candidate_groups": {group: [] for group in DISPOSITIONS},
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

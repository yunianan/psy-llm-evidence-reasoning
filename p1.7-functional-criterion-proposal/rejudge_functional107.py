"""Registered, offline functional-node review and overlay accounting.

The manual review table is input, not inferred by keyword matching. No diagnosis
client, model, gold labels, or sealed evaluation input is used.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / "rejudgment-v1"
INPUTS = ROOT / "phase1/p1.4-dev15-case2/dev15.inputs.jsonl"
PREDICTIONS = ROOT / "phase1/p1.4-dev15-case2/runs/p1.4-dev15-case2-recovery-v1/predictions/A1.jsonl"
SCOPE = ROOT / "phase1/p1.6-pending-audit100/disputes-18-v1/functional107-scope.json"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def three_states(counter):
    return {state: counter[state] for state in ("met", "pending", "not_met")}


def prepare():
    if OUT.exists():
        raise FileExistsError("Registration already exists; do not overwrite it")
    scope = json.loads(SCOPE.read_text(encoding="utf-8-sig"))["items"]
    inputs, predictions = rows(INPUTS), rows(PREDICTIONS)
    assert len(scope) == 107 and len({(r["case_id"], r["node_id"]) for r in scope}) == 107
    assert len(inputs) == len(predictions) == 15
    assert [r["case_id"] for r in inputs] == [r["case_id"] for r in predictions]
    OUT.mkdir()
    for name in ("proposal-v1.md", "spec-v1.json"):
        shutil.copyfile(BASE / name, OUT / name)
    registration = {
        "version": "functional107-rejudgment-v1",
        "registered_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status_at_registration": "before_new_status_assignment",
        "design": "retrospective_development_impact_assessment",
        "authorization": "User explicitly replied 重判 to the reviewable functional-criterion proposal v1",
        "rule_version": "functional-criterion-proposal-v1",
        "rule_adoption_scope": "this_offline_overlay_only_not_A1_default_or_r2_amendment",
        "rule_snapshots": {name: digest(OUT / name) for name in ("proposal-v1.md", "spec-v1.json")},
        "frozen_inputs": {str(p.relative_to(ROOT)): digest(p) for p in (INPUTS, PREDICTIONS, SCOPE)},
        "case_order": [r["case_id"] for r in inputs],
        "target_keys": [{"case_id": r["case_id"], "node_id": r["node_id"]} for r in scope],
        "target_nodes": 107, "mandatory": 77, "optional": 30,
        "independent_second_reviewer": False, "clinical_gold": None, "kappa": None,
        "new_diagnostic_api_calls": 0, "EVAL_accessed": False,
        "excluded_interventions": ["R3_symptom_counting", "L2_fact_collection", "G2_C_pruning", "computational_criteria"],
    }
    write(OUT / "registration.json", registration)
    by_case = {r["case_id"]: r["report"] for r in predictions}
    lines = ["# 固定107条功能损害原文复核材料", "", "登记先于新状态赋值；不含gold。", ""]
    for case in inputs:
        selected = [r for r in scope if r["case_id"] == case["case_id"]]
        if not selected:
            continue
        lines += [f"## {case['case_id']}", ""]
        for node in case["profile"]["evidence_nodes"]:
            lines += [f"[{node['evidence_id']}] {node['quote']}", ""]
        trace = {n["node_id"]: n for n in by_case[case["case_id"]]["tree_trace"]}
        for node in selected:
            record = trace[node["node_id"]]
            lines += [f"### {node['node_id']}",
                      f"mandatory={node['executor_mandatory']}; source_condition={node['source_condition']}",
                      "original_rationale: " + record["rationale"], ""]
    (OUT / "review-material.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"registered": 107, "cases": 15, "new_statuses_assigned": 0}, ensure_ascii=False))


GROUPS = {
    "adhd_met": ("met", "functional_component_supported", "L1_explicit_link",
                 "明确注意干扰与工作后果的直接连接，满足F2局部功能组件；不据此补齐ADHD全部标准。"),
    "adhd_motor_link_missing": ("pending", "link_missing", "L3_cooccurrence_only",
                                "已给注意分散相关工作干扰，但未明确多动/冲动本身造成的具体后果；不跨亚型搬用归属。"),
    "adhd_domain_unresolved": ("pending", "source_scope_unresolved", "L1_explicit_link",
                              "家务/兴趣活动的后果明确，但v1的S1描述社交/学习/工作范围；未给该范围的后果，不扩大源允许领域。"),
    "depressive_met": ("met", "functional_component_supported", "L1_explicit_link",
                       "low/heaviness与专注困难有直接原文连接，结合反复阅读/日常任务质量下降，满足F2局部组件；病程和完整发作仍由其他节点决定。"),
    "depressive_target_missing": ("pending", "target_feature_missing", "L0_unknown",
                                  "功能后果已给，但疲乏/heavy/off/flat等未无歧义建立本节点的目标抑郁体验或兴趣变化及其连接。"),
    "depressive_link_missing": ("pending", "link_missing", "L3_cooccurrence_only",
                                "功能后果与低落线索都已给，但未明确目标时期连接；功能困难引起低落的方向也不能反向使用。"),
    "schizoaffective_context_missing": ("pending", "target_feature_missing", "L0_unknown",
                                       "一般注意/情绪功能困难不能自动归入顶层精神病性-心境病程；全文未给能识别该目标病程的精神病性关联事实。"),
    "mania_target_and_marked_missing": ("pending", "target_feature_missing", "L0_unknown",
                                       "没有可识别的异常心境/精力发作及其与后果的关系；已有F2也不能替代v1对manic组件的F3要求。"),
    "trauma_target_missing": ("pending", "target_feature_missing", "L0_unknown",
                              "一般功能困难不补作创伤暴露、创伤特异表现或其与损害的时间/因果连接。"),
    "worry_target_missing": ("pending", "target_feature_missing", "L0_unknown",
                             "没有可识别的过度/不可控担忧与活动后果的连接；一般压力、分心和不宁不自动满足该归属。"),
    "stressor_target_missing": ("pending", "target_feature_missing", "L0_unknown",
                                "没有可识别的应激事件及其与目标反应、功能后果的连接。"),
}


def node_id(short):
    parts = short.split(".", 1)
    suffix = (parts[1] + "/") if len(parts) == 2 else ""
    return f"criterion.{parts[0]}.{suffix}functional_impairment_required"


def build():
    sys.path.insert(0, str(ROOT / "L1-L5系统"))
    from layers.layer3.diagnosis.decision_tree import criterion_nodes, load_candidate_graph
    from layers.layer3.diagnosis.assessment import assessments_from_step1
    from layers.layer3.diagnosis.schemas import CandidateGeneration, DiagnosticCandidate
    from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider

    if (OUT / "summary.json").exists():
        raise FileExistsError("Completed review already exists; do not overwrite or tune it")
    registration = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    for name, expected in registration["rule_snapshots"].items():
        assert digest(OUT / name) == expected, name
    for name, expected in registration["frozen_inputs"].items():
        assert digest(ROOT / name) == expected, name
    manual = json.loads((OUT / "manual-review.json").read_text(encoding="utf-8"))
    spec = json.loads((OUT / "spec-v1.json").read_text(encoding="utf-8"))
    inputs, predictions = rows(INPUTS), rows(PREDICTIONS)
    scope = json.loads(SCOPE.read_text(encoding="utf-8-sig"))["items"]
    source_text = {r["case_id"]: {n["evidence_id"]: n["quote"] for n in r["profile"]["evidence_nodes"]} for r in inputs}
    for row in inputs:
        quotes = source_text[row["case_id"]]
        for narrative in row["profile"]["key_narratives"]:
            assert any(narrative["value"] == quotes[ref] for ref in narrative["sources"])
    assignments = {}
    for cid, case in manual["cases"].items():
        for group, nodes in case["groups"].items():
            assert group in GROUPS
            for short in nodes:
                key = cid, node_id(short)
                assert key not in assignments, key
                assignments[key] = group
    assert set(assignments) == {(r["case_id"], r["node_id"]) for r in scope}
    provider = JsonDiagnosticKnowledgeProvider()
    definitions = {label: criterion_nodes(label, knowledge=provider) for label in {r["node_id"].split(".")[1] for r in scope}}
    graph = load_candidate_graph()
    original_trace = {r["case_id"]: {n["node_id"]: n for n in r["report"]["tree_trace"]} for r in predictions}
    reviewed = []
    for index, item in enumerate(scope, 1):
        cid, nid = item["case_id"], item["node_id"]
        group, case = assignments[cid, nid], manual["cases"][cid]
        status, reason_code, link, reason = GROUPS[group]
        definition = definitions[nid.split(".")[1]][nid]
        assert definition["mandatory"] == item["executor_mandatory"]
        assert definition["source_condition"] is True
        assert definition["source_refs"] == item["source_refs"]
        assert original_trace[cid][nid]["status"] == "pending"
        assert original_trace[cid][nid]["symptom_ids"] == []
        spans = []
        for suffix, quote in case["spans"]:
            ref = cid + "-" + suffix
            text = source_text[cid][ref]
            start = text.index(quote)
            spans.append({"evidence_ref": ref, "quote": quote, "start_char": start,
                          "end_char": start + len(quote), "offset_basis": "Unicode_codepoints_half_open"})
        row = {**item, "review_id": f"F{index:03d}", "manual_group": group,
               "source_group": definition["source_group"], "functional_fact": case["functional_fact"],
               "severity_code": "F2", "severity_rationale": case["severity_rationale"],
               "domains": case["domains"], "target_feature": case["target_note"], "link_code": link,
               "link_fact": case["target_note"], "denial_scope": "none", "evidence_spans": spans,
               "reviewed_narrative_refs": list(source_text[cid]), "whole_case_narrative_reviewed": True,
               "rule_ids": ["R1_functional_threshold", "R2_component_scope"], "new_status": status,
               "reason_code": reason_code, "rationale": reason + " " + case["target_note"],
               "secondary_reason_codes": (["severity_below_or_unclear"] if group == "mania_target_and_marked_missing" else []),
               "missing_question": None if status == "met" else case["missing_question"],
               "reviewer": "same_assistant_source_text_review_not_independent_clinician",
               "uncertainty": "single_reviewer_provisional_no_clinical_gold",
               "negation_note": case.get("negation_note"), "new_diagnostic_api_calls": 0}
        assert set(spec["required_review_fields"]) <= row.keys()
        if status == "met":
            assert link in {"L1_explicit_link", "L2_explicit_shared_period"}
        reviewed.append(row)

    changes = {(r["case_id"], r["node_id"]): r for r in reviewed}
    path_rows, overlay_rows, case_rows = [], [], []
    criteria_counts = {"before": Counter(), "after": Counter()}
    original_all_functional, new_all_functional = Counter(), Counter()
    for row in predictions:
        cid, report = row["case_id"], row["report"]
        updated_trace = deepcopy(report["tree_trace"])
        by_id = {}
        for node in updated_trace:
            original = deepcopy(node)
            change = changes.get((cid, node["node_id"]))
            if change:
                node.update(status=change["new_status"], rationale="功能判据v1离线复核：" + change["rationale"],
                            evidence_refs=list(dict.fromkeys(s["evidence_ref"] for s in change["evidence_spans"])))
            else:
                assert node == original
            assert node["confidence"] == original["confidence"]
            assert node["symptom_ids"] == original["symptom_ids"]
            by_id[node["node_id"]] = node
            if node["stage"] == "criterion":
                criteria_counts["before"][original["status"]] += 1
                criteria_counts["after"][node["status"]] += 1
                if node["node_id"].endswith("functional_impairment_required"):
                    original_all_functional[original["status"]] += 1
                    new_all_functional[node["status"]] += 1
        candidates = deepcopy(report["candidates"])
        coverage = deepcopy(report["candidate_coverage"])
        metrics = {when: {m: 0 for m in ["C", "E", "T"]} for when in ["before", "after"]}
        for candidate in candidates:
            label = candidate["disorder"]
            defs = criterion_nodes(label, knowledge=provider)
            guards = graph["paths"][label]["guards"]
            paths = candidate["decision_paths"]
            assert len(paths) == 1 and paths[0]["path_id"] == label
            path = paths[0]
            before_records = {n["node_id"]: n for n in path["decisions"]}
            for nid, record in before_records.items():
                assert record == original_trace[cid][nid]
            path["decisions"] = [deepcopy(by_id[n["node_id"]]) for n in path["decisions"]]
            after_records = {n["node_id"]: n for n in path["decisions"]}
            impact = {"case_id": cid, "disorder": label, "confidence": candidate["confidence"],
                      "original_coverage": report["candidate_coverage"][label]}
            for when, records in [("before", before_records), ("after", after_records)]:
                blockers = [nid for nid, d in defs.items() if d["mandatory"] and records.get(nid, {}).get("status") != "met"]
                guard_blockers = [{"node_id": g["node_id"], "expected": g["expected"],
                                   "actual": records.get(g["node_id"], {}).get("status", "pending")}
                                  for g in guards if records.get(g["node_id"], {}).get("status") != g["expected"]]
                C = bool(defs) and not blockers
                E = C and not guard_blockers
                T = E and candidate["confidence"] >= 0.5
                impact[when] = {"C": C, "E": E, "T": T, "mandatory_blockers": blockers, "guard_blockers": guard_blockers}
                for metric in metrics[when]:
                    metrics[when][metric] += int(impact[when][metric])
            path["provisional"] = bool(impact["after"]["mandatory_blockers"]) or any(
                after_records[g["node_id"]]["status"] == "pending" for g in guards)
            coverage[label] = "provisional_candidate" if path["provisional"] else "candidate"
            candidate["core_evidence_summary"] = "；".join(f"{n['node_id']}={n['status']}: {n['rationale']}" for n in path["decisions"])
            candidate["evidence_refs"] = list(dict.fromkeys(ref for n in path["decisions"] if n["status"] == "met" for ref in n["evidence_refs"]))
            impact["overlay_coverage"] = coverage[label]
            impact["target_nodes_changed_to_met"] = [nid for nid in before_records if (cid, nid) in changes and changes[cid, nid]["new_status"] == "met"]
            impact["non_target_mandatory_blockers_after"] = [nid for nid in impact["after"]["mandatory_blockers"] if (cid, nid) not in changes]
            path_rows.append(impact)
        assert [(c["disorder"], c["confidence"], c["symptom_match"]) for c in candidates] == [
            (c["disorder"], c["confidence"], c["symptom_match"]) for c in report["candidates"]]
        parsed = [DiagnosticCandidate.model_validate({k: c[k] for k in DiagnosticCandidate.model_fields if k in c}) for c in candidates]
        generation = CandidateGeneration(considered_disorders=report["considered_disorders"], candidates=parsed,
                                         tree_trace=updated_trace, missing_dimensions=[], candidate_coverage=coverage)
        assessed = assessments_from_step1(generation, parsed, knowledge=provider)
        actual_eligible = {a.disorder for a in assessed if a.conclusion == "meets_criteria"}
        expected_eligible = {p["disorder"] for p in path_rows if p["case_id"] == cid and p["after"]["E"]}
        assert actual_eligible == expected_eligible
        no_eligible = not actual_eligible
        case_rows.append({"case_id": cid, "target_nodes": sum(r["case_id"] == cid for r in reviewed),
                          "new_met": sum(r["case_id"] == cid and r["new_status"] == "met" for r in reviewed),
                          "candidate_paths": len(candidates), "metrics": metrics,
                          "original_final_diagnoses": report["final_diagnoses"],
                          "counterfactual_final_diagnoses": [] if no_eligible else None,
                          "counterfactual_routing": "return_to_l2" if no_eligible else None,
                          "routing_basis": "no_eligible_path_deterministic_branch" if no_eligible else "new_Verifier_required",
                          "verifier_status": "not_run_for_intervention"})
        overlay_rows.append({"version": "functional107-node-path-overlay-v1", "case_id": cid,
                             "artifact_type": "offline_overlay_not_complete_engine_prediction", "source_arm": "A1",
                             "tree_trace": updated_trace, "candidates": candidates, "candidate_coverage": coverage,
                             "verifier_status": "not_run_for_intervention"})
    transitions = Counter(r["new_status"] for r in reviewed)
    strata = {"mandatory" if mandatory else "optional": three_states(Counter(r["new_status"] for r in reviewed if r["executor_mandatory"] == mandatory)) for mandatory in (True, False)}
    gates = {when: {metric: {"paths": sum(p[when][metric] for p in path_rows), "path_denominator": len(path_rows),
                             "cases": sum(c["metrics"][when][metric] > 0 for c in case_rows), "case_denominator": 15}
                    for metric in ("C", "E", "T")} for when in ("before", "after")}
    summary = {"version": "functional107-rejudgment-summary-v1", "completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
               "design": registration["design"], "Q1": {"transitions_from_pending": three_states(transitions), "denominator": len(reviewed), "strata": strata,
                    "primary_pending_reasons": dict(Counter(r["reason_code"] for r in reviewed if r["new_status"] == "pending"))},
               "Q2_Q3": gates, "criteria_status_counts": {k: three_states(v) for k, v in criteria_counts.items()},
               "all_functional_node_status_counts": {"before": three_states(original_all_functional), "after": three_states(new_all_functional)},
               "per_case": case_rows,
               "by_node": {nid: three_states(Counter(r["new_status"] for r in reviewed if r["node_id"] == nid)) for nid in sorted({r["node_id"] for r in reviewed})},
               "candidate_set_scores_order_and_symptom_counts_unchanged": True,
               "candidate_recall_ppv_top5_change": "structurally_zero_not_new_H1_evidence",
               "changed_cases": sum(c["new_met"] > 0 for c in case_rows), "cohort_cases": 15, "target_cases": 13,
               "independent_second_reviewer": False, "clinical_gold": None, "kappa": None, "formal_CI": None,
               "new_diagnostic_api_calls": 0, "new_diagnostic_api_cost_yuan": 0, "baseline_A1_or_r2_changed": False,
               "EVAL_accessed": False, "G2_C_pruning_approved": False,
               "integrity": {"unique_keys": len(assignments), "reviewed_records": len(reviewed),
                             "exact_spans_checked": sum(len(r["evidence_spans"]) for r in reviewed),
                             "all_key_narratives_joined_to_evidence": True, "non_target_records_unchanged": True,
                             "node_path_copies_consistent": True, "existing_assessment_eligibility_matched_paths": len(path_rows)}}
    assert len(reviewed) == 107 and len(path_rows) == 216 and len(case_rows) == 15
    for name, expected in registration["frozen_inputs"].items():
        assert digest(ROOT / name) == expected, name
    for name, data in [("items.jsonl", reviewed), ("path-impact.jsonl", path_rows), ("overlay-generations.jsonl", overlay_rows)]:
        (OUT / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")
    write(OUT / "summary.json", summary)
    lines = ["# 107条功能损害：逐条离线重判", "", "按用户确认的v1；同助手复核，无独立临床金标准。", "",
             "| ID | 病例 / 节点 | 必需 | 新状态 | 原因码 | 依据及缺失事实 | 原文span |", "|---|---|---|---|---|---|---|"]
    for r in reviewed:
        spans = "; ".join(f"[{s['evidence_ref']}:{s['start_char']}..{s['end_char']}] {s['quote']}" for s in r["evidence_spans"])
        lines.append(f"| {r['review_id']} | {r['case_id']} / {r['node_id']} | {r['executor_mandatory']} | {r['new_status']} | {r['reason_code']} | {r['rationale'].replace('|','/')} | {spans.replace('|','/')} |")
    (OUT / "items.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"Q1": summary["Q1"], "Q2_Q3": gates, "changed_cases": summary["changed_cases"], "criteria": summary["criteria_status_counts"], "integrity": summary["integrity"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["build"]:
        build()
    else:
        raise SystemExit("Use prepare or build")

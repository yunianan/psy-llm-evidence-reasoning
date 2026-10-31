"""以既有DEV证据重算病程契约；不调用模型、不读EVAL、不覆盖历史结果。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.clinical_contract import CONTRACT_VERSION, TEMPORAL_POLICY_VERSION
from shared.temporal import parse_duration
from shared.temporal_gate import evaluate_temporal_node
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from layers.layer3.diagnosis.decision_tree import criterion_nodes


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def audit(output: Path):
    workspace = ROOT.parent
    pairs = read_rows(workspace / "phase2/p2.2-reference-gate-audit/pairs.jsonl")
    profiles = {r["case_id"]: r["profile"] for r in read_rows(workspace / "phase1/p1.4-dev15-case2/dev15.inputs.jsonl")}
    old_summary = json.loads((workspace / "phase2/p2.2-source-extraction-audit/summary.json").read_text(encoding="utf-8"))
    positives = {r["case_id"]: r for r in old_summary["positive_controls"]}
    # 复用上一轮已经人工定位并保存的病程span；不按参考标签生成新事实。
    spec = importlib.util.spec_from_file_location("prior_source_audit", workspace / "phase2/p2.2-source-extraction-audit/review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = JsonDiagnosticKnowledgeProvider()
    rows, augmented = [], {}
    for pair in pairs:
        case, label = pair["case_id"], pair["disorder"]
        profile = augmented.setdefault(case, deepcopy(profiles[case]))
        profile["contract_version"] = CONTRACT_VERSION
        nodes = {n["evidence_id"]: n for n in profile["evidence_nodes"]}
        for check in pair["checks"]:
            if check["gate"] != "duration":
                continue
            nid = check["node_id"]
            definition = {"node_id": nid, **criterion_nodes(label, knowledge=provider)[nid]}
            if label == "D001":
                anchor = module.CASE_NOTES[case]["time_anchor"]
                matches = [n for n in nodes.values() if anchor in n["quote"]]
                if len(matches) != 1:
                    raise ValueError(f"Prior source span cannot be verified: {case}")
                record = parse_duration(anchor, scope=nid, evidence_refs=[matches[0]["evidence_id"]], source="L3_writeback")
                profile.setdefault("duration_evidence", []).append(record.model_dump(mode="json"))
            before_records = [d for d in profiles[case].get("duration_evidence", []) if d.get("scope") == nid]
            decision = evaluate_temporal_node(profile, definition, evidence_refs=check["evidence_refs"])
            status = decision.status if decision.status in ("met", "not_met") else "pending"
            records = [d for d in profile.get("duration_evidence", []) if d["scope"] == nid]
            invariant_errors = []
            if status in ("met", "not_met") and not any(d["parse_status"] == "ok" and d["evidence_refs"] for d in records):
                invariant_errors.append("known_duration_without_structured_evidence")
            for d in records:
                if not all(ref in nodes and d["raw_span"] in nodes[ref]["quote"] for ref in d["evidence_refs"]):
                    invariant_errors.append("writeback_quote_not_in_source")
            parse_statuses = {d["parse_status"] for d in records}
            reason = ("symptom_scope_unresolved" if "其他症状群" in decision.reason else
                      "approx_boundary_requires_clarification" if decision.status == "met_approx" else
                      "crosses_threshold_or_scope_unresolved" if status == "pending" and "ok" in parse_statuses else
                      "source_time_unquantified" if parse_statuses & {"vague", "unresolved"} else
                      "parse_failed_or_no_cited_time" if status == "pending" else "comparable_interval")
            rows.append({"case_id": case, "disorder": label, "node_id": nid,
                "original_status": check["observed_status"], "effective_status": status,
                "decision": decision.model_dump(mode="json"), "duration_evidence": records,
                "structured_empty_before": not before_records, "structured_empty_after": not records,
                "reason_category": reason, "invariant_errors": invariant_errors,
                "historical_node_scope_reused": True, "new_clinical_scope_adjudication": False})
    before = Counter(r["original_status"] for r in rows)
    after = Counter(r["effective_status"] for r in rows)
    controls = [r for r in rows if r["case_id"] in positives and r["disorder"] == "D001"]
    summary = {"contract_version": CONTRACT_VERSION, "temporal_policy": TEMPORAL_POLICY_VERSION,
        "cases": len(profiles), "duration_checks": len(rows), "before": dict(before), "after": dict(after),
        "met_approx": sum(r["decision"]["status"] == "met_approx" for r in rows),
        "positive_controls": {"total": len(controls), "empty_before": sum(r["structured_empty_before"] for r in controls),
                              "empty_after": sum(r["structured_empty_after"] for r in controls)},
        "original_unknown_now_met": sum(r["original_status"] == "pending" and r["effective_status"] == "met" for r in rows),
        "original_unknown_now_not_met": sum(r["original_status"] == "pending" and r["effective_status"] == "not_met" for r in rows),
        "unknown_reasons": dict(Counter(r["reason_category"] for r in rows if r["effective_status"] == "pending")),
        "remaining_consistency_errors": sum(len(r["invariant_errors"]) for r in rows),
        "new_api_calls": 0, "eval_accessed": False, "historical_files_changed": False,
        "scope": "deterministic temporal overlay of historical node citations; no new clinical association adjudication"}
    if summary["remaining_consistency_errors"]:
        raise ValueError("Temporal audit invariants failed")
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in ("temporal-checks.jsonl", "profiles.jsonl", "summary.json", "report.md")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Choose a new audit directory; do not overwrite a prior audit")
    targets[0].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    targets[1].write_text("".join(json.dumps({"case_id": case, "profile": p}, ensure_ascii=False) + "\n" for case, p in augmented.items()), encoding="utf-8")
    targets[2].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# DEV-15 病程契约重算", "", "日期：2026-10-30。保留旧结果，零API调用；只重算病程区间与引用一致性。", "",
        f"30条参考路径中，病程未知从{before['pending']}/30变为{after['pending']}/30；约数边界met_approx有{summary['met_approx']}项。",
        f"6例正对照的结构化空值从{summary['positive_controls']['empty_before']}降为{summary['positive_controls']['empty_after']}；剩余一致性错误{summary['remaining_consistency_errors']}。", "",
        "|病例|候选|原状态|区间比较|生效状态|原因|", "|---|---|---|---|---|---|"]
    lines += [f"|{r['case_id']}|{r['disorder']}|{r['original_status']}|{r['decision']['status']}|{r['effective_status']}|{r['reason_category']}|" for r in rows]
    lines += ["", "解析为区间不等于门槛已满足；模糊、缺锚点和约数边界仍需追问。历史节点的症状归属沿用原记录，未借助参考诊断补事实，也未新增临床归属裁定。",
        "这30项是原参考路径的病程节点，其他门及正式PPV评分未重跑。原文未量化与解析失败分别计数；不能把缺少精确病程都归因于抽取器。"]
    targets[3].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().output), ensure_ascii=False, indent=2))

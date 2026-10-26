"""Offline, descriptive DEV scoring of the completed engineering recovery.

Uses the unchanged v1.1 scorer; never invokes a model or opens sealed data.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1] / "L1-L5系统"
sys.path.insert(0, str(REPO))

from tools.paired_evaluation import compare, read_rows, markdown, VERSION
from tools.prediction_contract import keyed, normalize_for_protocol, validate_gold
from tools.l3_observability_summary import _distribution
from tools.mentalbench_partition import family_key


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def percent(value):
    return "不适用" if value is None else f"{float(Fraction(value)) * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "scoring/dev15-v1")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise ValueError("Use a fresh scoring directory; never overwrite an earlier score")
    run = BASE / "runs/p1.4-dev15-case2-recovery-v1"
    protocol = REPO.parent / "phase0/evaluation-protocol-v1.1.md"
    gold_path = REPO / "artifacts/datasets/mentalbench_split_v2/development/dev.gold.jsonl"
    profiles = read_rows(BASE / "dev15.inputs.jsonl")
    ids = [r["case_id"] for r in profiles]
    quarantined = {r.get("case_id") for r in read_rows(gold_path.with_name("dev.quarantine.jsonl"))}
    if set(ids) & quarantined or any(not r.get("profile") for r in profiles):
        raise ValueError("Unexpected ineligible case in the registered cohort; do not silently change its denominator")
    gold_all = keyed(read_rows(gold_path))
    gold = [gold_all[case] for case in ids]
    validate_gold(gold)
    arm_paths = {a: run / f"predictions/{a}.jsonl" for a in ("A0", "A1")}
    arm_rows = {a: read_rows(path) for a, path in arm_paths.items()}
    arms = {a: {k: normalize_for_protocol(v) for k, v in keyed(rows).items()} for a, rows in arm_rows.items()}
    raw_a1 = keyed(arm_rows["A1"])
    strata = {kind: compare(gold, arm_rows["A0"], arm_rows["A1"], kind=kind)
              for kind in ("single_reference_label", "admissible_candidates")}
    per_case, stage_counts, coverage_counts, pending_counts = [], {}, Counter(), []
    for reference in gold:
        case, truth = reference["case_id"], set(reference["gold_labels"])
        item = {"case_id": case, "reference_kind": reference["reference_kind"],
                "supergroup": family_key(reference["source_code"]), "gold_labels": reference["gold_labels"]}
        for arm, data in arms.items():
            prediction = data[case]
            labels = [c["disorder"] for c in prediction["candidates"]]
            item[arm] = {"gold_count": len(truth), "hits": len(truth & set(labels)),
                         "predicted": len(labels), "top5_hits": len(truth & set(labels[:5])),
                         "empty": not labels, "complete": prediction["complete"],
                         "candidate_labels": labels, "top5_labels": labels[:5],
                         "gold_ranks": {g: labels.index(g) + 1 if g in labels else None for g in sorted(truth)},
                         "final_diagnoses": prediction["final_diagnoses"]}
        trace = {}
        for node in raw_a1[case]["report"]["tree_trace"]:
            key = node["node_id"]
            value = (node["stage"], node["status"])
            if key in trace and trace[key] != value:
                raise ValueError("Conflicting judgments for a node")
            trace[key] = value
        pending = sum(status == "pending" for _, status in trace.values())
        item["A1_pending_nodes"] = pending
        pending_counts.append(pending)
        for stage, status in trace.values():
            stage_counts.setdefault(stage, Counter())[status] += 1
        coverage = raw_a1[case]["report"]["candidate_coverage"]
        assert len(coverage) == 24
        coverage_counts.update("excluded" if s.startswith("excluded:") else s for s in coverage.values())
        per_case.append(item)
    # Check every disclosed integer against independent set-counting per-case rows.
    for kind, result in strata.items():
        selected = [r for r in per_case if r["reference_kind"] == kind]
        for arm, slot in (("A0", "x"), ("A1", "y")):
            counts = result["counts"][slot]
            assert counts["cases"] == len(selected)
            for field in ("hits", "predicted", "gold", "top5_hits", "empty"):
                key = "gold_count" if field == "gold" else field
                assert counts[field] == sum(r[arm][key] for r in selected)
    pending = {"A0": None, "A0_status": "not_applicable_no_tree", "A1": {
        "cases_with_g1": len(ids), "cases_with_pending": sum(v > 0 for v in pending_counts),
        "pending_nodes_per_case": _distribution(pending_counts),
        "per_stage": {stage: {"counts": dict(sorted(counts.items())), "executed": sum(counts.values()),
                               "pending_fraction": str(Fraction(counts["pending"], sum(counts.values())))}
                      for stage, counts in sorted(stage_counts.items())},
        "coverage_denominator": 24 * len(ids), "coverage_counts": dict(sorted(coverage_counts.items())),
        "no_final_diagnosis_cases": sum(not arms["A1"][case]["final_diagnoses"] for case in ids)}}
    paths = {"gold_dev": gold_path, "cohort": BASE / "dev15.inputs.jsonl", "protocol": protocol,
             "scorer": REPO / "tools/paired_evaluation.py", **arm_paths}
    metadata = {"scored_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "version": VERSION,
                "source_run_id": run.name, "arm_x": "A0", "arm_y": "A1", "direction": "A1-A0",
                "planned": len(ids), "excluded": {"quarantined": 0, "missing_profile": 0, "missing_gold": 0},
                "qualified": len(ids), "gold_dev_rows": len(gold_all),
                "stratum_counts": dict(Counter(r["reference_kind"] for r in gold)),
                "supergroups": dict(Counter(family_key(r["source_code"]) for r in gold)),
                "scope": "descriptive_DEV_engineering_recovery", "confirmatory_claim": False,
                "primary_status": "empty_primary_stratum", "secondary_ci_status": "insufficient_supergroups",
                "new_model_calls": 0, "eval_accessed": False, "formal_eval_score": False,
                "source_files": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                 for key, path in paths.items()}}
    out.mkdir(parents=True)
    write_json(out / "metadata.json", metadata)
    for kind, result in strata.items():
        write_json(out / f"{kind}.json", result)
        (out / f"{kind}.md").write_text(markdown(result), encoding="utf-8")
    (out / "dev15.gold.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold), encoding="utf-8")
    (out / "per-case.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_case), encoding="utf-8")
    write_json(out / "pending.json", pending)
    result = strata["admissible_candidates"]
    names = {"candidate_recall_micro": "候选召回率", "candidate_ppv_micro": "候选 PPV",
             "top_k_recall_micro": "Top-5 召回率", "candidate_count_mean": "每例平均候选数",
             "empty_candidate_rate": "空候选率"}
    lines = ["# DEV-15 诊断效果描述性评分", "",
             "已获用户授权，对完成的 A0/A1 工程恢复结果离线评分；不新增模型请求、不访问 EVAL。", "",
             "## 样本与统计边界", "",
             "固定名单 15/15 均保留，两臂完整率均为 100%，资格排除 0。全部为 D001+D013 双标签病例，每例贡献 2 个 gold，共 30 个 gold 标签。只有 1 个 supergroup，单标签主层为 0 例。", "",
             "按 v1.1 §5，M<2 不计算聚类 CI，也不把组拆成病例做 bootstrap。H1 单标签主比较没有样本，不能作获胜或正式劣效判定；双标签层仅报告描述性点估计。H2/H3 未运行。", "",
             "## 双标签层结果", "", "| 指标 | A0 | A1 | A1−A0 |", "|---|---:|---:|---:|"]
    for metric, label in names.items():
        left, right, delta = result["metrics"]["x"][metric], result["metrics"]["y"][metric], result["deltas"][metric]["point"]
        if metric == "candidate_count_mean":
            values = [f"{float(Fraction(v)):.2f}" for v in (left, right, delta)]
        else:
            values = [percent(left), percent(right), f"{float(Fraction(delta))*100:+.2f}pp"]
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines += ["", "精确分子分母：A0 命中 24、预测 53、gold 30、Top-5 命中 24；A1 命中 30、预测 216、gold 30、Top-5 命中 18。所有候选均依 candidate/provisional_candidate 口径，按置信度降序、标签字典序打破并列；未应用最终诊断的 0.5 门槛删减候选。", "",
              "候选召回提高 20pp，同时候选数由 3.53 增至 14.40，PPV 降低约 31.39pp，Top-5 召回降低 20pp。本批表现为更宽的候选召回，候选筛选和前五排序仍有明显问题，不能只凭 100% 召回宣称知识图谱提高了总体诊断效果。H1 的 PPV 非劣界值仍为 −1pp，不据本批结果更改。", "",
              "## 最终诊断与 pending", "",
              f"A1 有 {pending['A1']['no_final_diagnosis_cases']}/15 例最终诊断为空。两个 gold 标签在全部病例均为 provisional_candidate；候选覆盖不等于已完成最终诊断。双标签层 final.reference_hit_rate 按协议为 null；A0 最终诊断按入口契约为空，也不派生 top-1 冒充最终诊断。", "",
              f"A1 的 {pending['A1']['cases_with_pending']}/15 例有 pending 节点；每例 pending 数 p50/p95={pending['A1']['pending_nodes_per_case']['p50']}/{pending['A1']['pending_nodes_per_case']['p95']}。A0 没有树轨迹，pending 记不适用。", "",
              "| 阶段 | met | not_met | pending | 已执行 |", "|---|---:|---:|---:|---:|"]
    for stage, value in pending["A1"]["per_stage"].items():
        c = value["counts"]
        lines.append(f"| {stage} | {c.get('met',0)} | {c.get('not_met',0)} | {c.get('pending',0)} | {value['executed']} |")
    lines += ["", f"coverage 分母为 24×15=360，计数：`{dict(coverage_counts)}`。未执行节点不补成 pending。", "",
              "## 复算与来源", "", "原预测文件未修改；原失败批次和恢复批次继续分开保存。metadata.json 记录协议、评分器、名单、gold 与两臂预测的路径和摘要；per-case.jsonl 提供逐例分子分母、排序、gold 名次；pending.json 提供伴随指标整数计数。", "",
              "此结论只适用于该固定疾病组合的 15 例 DEV 工程样本。正式样本量、分层抽样与研究版本仍未 freeze，不扩展到全部疾病、正式 EVAL 或临床准确性。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "counts": result["counts"], "metrics": result["metrics"],
                      "primary_status": metadata["primary_status"], "ci_status": metadata["secondary_ci_status"],
                      "pending": pending["A1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

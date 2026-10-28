"""Recompute DEV-15 candidate/ranking mechanisms and audit pending explanations offline."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1] / "L1-L5系统"
sys.path.insert(0, str(REPO))
from tools.paired_evaluation import read_rows
from tools.prediction_contract import keyed, normalize_for_protocol
from layers.layer3.diagnosis.decision_tree import criterion_nodes, load_candidate_graph
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from shared.disorder_catalog import SUPPORTED_DISORDERS


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def field_type(node_id):
    return node_id.rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def main():
    out = BASE / "scoring/mechanism-audit-v1"
    if out.exists():
        raise ValueError("Audit output already exists")
    run = BASE / "runs/p1.4-dev15-case2-recovery-v1"
    graph = load_candidate_graph()
    provider = JsonDiagnosticKnowledgeProvider()
    definitions = {k: v for label in SUPPORTED_DISORDERS for k, v in criterion_nodes(label, knowledge=provider).items()}
    arms = {arm: keyed(read_rows(run / f"predictions/{arm}.jsonl")) for arm in ("A0", "A1")}
    gold = keyed(read_rows(BASE / "scoring/dev15-v1/dev15.gold.jsonl"))
    raw, rows, pending = arms["A1"], [], []
    confidence = Counter()
    missing_pattern = re.compile("未提供|未提及|未报告|未描述|未明确|未说明|未记载|未记录|缺乏|缺少|缺失|未知|不详|不清|无法确定|无法确认|不能确认|不能确定|不确定|无法判断|不能判断")
    insufficient_pattern = re.compile("不足|未达|不到|少于|不满|不够|未够|仅见|仅有|少数|计数")
    unclear_pattern = re.compile("模糊|不明确|可能|似乎|疑似|归因|区分|不足以|不等于|难以")
    matched_confidence, candidates_total = 0, 0
    for case in sorted(gold):
        x, y = (normalize_for_protocol(arms[a][case]) for a in ("A0", "A1"))
        left, right = ({c["disorder"] for c in p["candidates"]} for p in (x, y))
        truth = set(gold[case]["gold_labels"])
        added = right - left
        records = {n["node_id"]: n for n in raw[case]["report"]["tree_trace"]}
        anchor_met = {label for label, path in graph["paths"].items()
                      if any(records.get(n, {}).get("status") == "met" for n in path["anchors"])}
        labels = [c["disorder"] for c in y["candidates"]]
        gold_candidate = next(c for c in y["candidates"] if c["disorder"] == "D013")
        strict_higher = sum(c["confidence"] > gold_candidate["confidence"] for c in y["candidates"])
        boundary_tie = y["candidates"][4]["confidence"] == gold_candidate["confidence"]
        rows.append({"case_id": case, "A0_subset_A1": left <= right, "added_candidates": len(added),
                     "added_reference": len(added & truth), "added_nonreference": len(added - truth),
                     "anchor_met_equals_retained": anchor_met == right,
                     "D001_rank": labels.index("D001") + 1, "D013_rank": labels.index("D013") + 1,
                     "D013_confidence": gold_candidate["confidence"], "D013_strictly_higher_count": strict_higher,
                     "D013_tied_at_top5_boundary": boundary_tie,
                     "D013_missed_by_boundary_tie": labels.index("D013") >= 5 and strict_higher < 5 and boundary_tie})
        for candidate in y["candidates"]:
            anchors = [records[n]["confidence"] for n in graph["paths"][candidate["disorder"]]["anchors"]
                       if records[n]["status"] == "met"]
            matched_confidence += candidate["confidence"] == max(anchors)
            candidates_total += 1
            confidence[str(candidate["confidence"])] += 1
        for node in records.values():
            if node["stage"] != "criterion" or node["status"] != "pending":
                continue
            definition = definitions[node["node_id"]]
            part, reason = field_type(node["node_id"]), node["rationale"]
            family = ("symptom_count_or_core" if part in {"min_count", "must_include_one_of", "must_include_all"} else
                      "duration" if "duration" in part else "functional_impairment" if "impairment" in part else
                      "stressor_event" if "stressor" in part else "other_source_requirement")
            flags = {"rationale_reports_missing_or_undetermined": bool(missing_pattern.search(reason)),
                     "rationale_reports_insufficient_evidence_or_count": bool(insufficient_pattern.search(reason)),
                     "rationale_reports_semantic_or_attribution_ambiguity": bool(unclear_pattern.search(reason))}
            primary = ("missing_or_undetermined_reported" if flags["rationale_reports_missing_or_undetermined"] else
                       "insufficient_evidence_or_count_reported" if flags["rationale_reports_insufficient_evidence_or_count"] else
                       "semantic_ambiguity_reported" if flags["rationale_reports_semantic_or_attribution_ambiguity"] else
                       "manual_review_required")
            pending.append({"case_id": case, "node_id": node["node_id"], "field_type": part, "condition_family": family,
                            "source_condition": definition["source_condition"], "source_refs": definition["source_refs"],
                            "rationale": reason, "evidence_refs": node["evidence_refs"], "reason_flags": flags,
                            "primary_reason": primary, "clinical_adjudication": "not_performed_by_keyword_audit"})
    assert len(pending) == 1115 and candidates_total == 216
    assert all(r["A0_subset_A1"] and r["anchor_met_equals_retained"] for r in rows)
    assert matched_confidence == candidates_total
    incremental = {k: sum(r[k] for r in rows) for k in ("added_candidates", "added_reference", "added_nonreference")}
    audit = {"cases": len(rows), "new_model_calls": 0, "eval_accessed": False, "clinical_adjudication": False,
             "incremental": incremental, "A0_subset_A1_cases": sum(r["A0_subset_A1"] for r in rows),
             "anchor_met_equals_retained_cases": sum(r["anchor_met_equals_retained"] for r in rows),
             "confidence_matches_anchor_max": matched_confidence, "confidence_total": candidates_total,
             "confidence_value_counts": dict(sorted(confidence.items(), key=lambda p: -float(p[0]))),
             "D001_top1": sum(r["D001_rank"] == 1 for r in rows), "D001_top5": sum(r["D001_rank"] <= 5 for r in rows),
             "D013_top5": sum(r["D013_rank"] <= 5 for r in rows),
             "D013_outside_top5": sum(r["D013_rank"] > 5 for r in rows),
             "D013_outside_due_to_boundary_tie": sum(r["D013_missed_by_boundary_tie"] for r in rows),
             "pending_by_field": dict(Counter(r["field_type"] for r in pending)),
             "pending_by_condition_family": dict(Counter(r["condition_family"] for r in pending)),
             "pending_by_rationale_category": dict(Counter(r["primary_reason"] for r in pending)),
             "rationale_categories_are_not_patient_fact_adjudication": True}
    cost = json.loads((run / "observability/joint.summary.json").read_text(encoding="utf-8"))
    total = Fraction(str(cost["per_arm"]["A0"]["cost_reference"]["upper_bound_yuan"])) + Fraction(str(cost["per_arm"]["A1"]["cost_reference"]["upper_bound_yuan"]))
    a1 = Fraction(str(cost["per_arm"]["A1"]["cost_reference"]["upper_bound_yuan"])) / len(rows)
    old_peak = Fraction("0.874690")
    budget = {"reference_only": True, "actual_bill": None, "includes_carried_source_reference_yuan": "6.007656",
              "all_chain_reference_yuan": str(total), "A1_peak_per_case": str(a1), "old_peak_per_case": str(old_peak),
              "same_basis_cost_multiplier": str(a1 / old_peak), "same_basis_cost_increase_fraction": str(a1 / old_peak - 1),
              "cache_hit_over_known_total": str(Fraction(2734592, 4946806)),
              "cache_hit_over_known_prompt": str(Fraction(2734592, 3511551)),
              "cache_miss": None, "full_chain_cache_hit_rate": None, "cache_cost_reduction": None,
              "planning_assumptions": "300 future yuan, 20% reserve, equal DEV/EVAL N; no G2/C cost; offpeak is hypothetical half official reference",
              "A1_only_N_peak_floor": int(Fraction(120) / a1), "A1_only_N_offpeak_floor": int(Fraction(120) / (a1 / 2)),
              "A0A1_N_peak_floor": int(Fraction(120) / (total / len(rows))),
              "A0A1_N_offpeak_floor": int(Fraction(120) / (total / len(rows) / 2)), "formal_N_frozen": False}
    out.mkdir(parents=True)
    write(out / "summary.json", audit)
    write(out / "budget-reference.json", budget)
    for filename, data in (("per-case.jsonl", rows), ("pending-reasons.jsonl", pending)):
        (out / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    text = ["# DEV-15 离线机制审计与预算口径", "", "本次新增模型调用 0、EVAL 访问 0，不改变 prompt、判定、排序、候选或正式协议。", "",
            "## 候选集合与排序", "",
            "15/15 例 A0⊆A1；新增候选 163，其中参考标签 6、非参考标签 157。157/6≈26.17 是新增非参考/参考的数量比，不是经临床裁定的误诊比。", "",
            "D001 在 13/15 例排名第一，15/15 均入前五；D013 仅 3/15 入前五，12 例落选贡献全部 Top-5 缺口。其中 9 例 D013 与第五名同分且严格更高分者不足 5 个，固定标签字典序直接影响边界入选；另 3 例已有至少 5 个严格更高分候选。18 个不同置信度取值不能排除局部并列的作用。", "",
            "216/216 候选 confidence 等于其 met anchor 的最大 confidence（decision_tree.py 的构造公式）。criterion/guard 的 pending 没有在这个候选排序字段中独立折算或扣分；confidence 是 anchor 支持程度，不是完整诊断概率。报告顶层 candidate_confidence 与逐候选 confidence 不能混称。", "",
            "15/15 例最终候选集合等于 anchor-met 保留集合。guard 全 pending，本批未追加剪枝；criteria 仍决定暂定状态及最终诊断门槛，不因候选阶段未缩小集合就说 criteria 完全没有执行或没有作用。", "",
            "## 1,115 个 criterion pending 的初筛", "",
            "| 条件家族 | pending 数 |", "|---|---:|"]
    text += [f"| {k} | {v} |" for k, v in audit["pending_by_condition_family"].items()]
    text += ["", "| rationale 表述分型（互斥优先级） | 数量 |", "|---|---:|"]
    text += [f"| {k} | {v} |" for k, v in audit["pending_by_rationale_category"].items()]
    text += ["", "上述分型仅标记原模型解释中的缺失/不足/含混表述，不是独立临床复核。逐节点原条件、来源、解释、evidence_refs 均见 pending-reasons.jsonl；不能仅凭这些表述断定事实确实未出现，或断定模型过度保守。严谨区分二者仍需对患者原文与节点条件做带引用的人工事实审计。unknown 不等于 absent，anchor 也不是完备症状事实表。", "",
             "知识与病例检索不能补造患者病程、年龄、损害或事件事实；缺失的患者事实应由 L2 追问补采，G2/C 只能解释或重新核对患者已经提供的证据。", "",
             "## 预算与缓存", "",
             f"A1 峰时全未命中参考上界={float(a1):.9f}元/例；旧峰时0.874690元/例，同口径为{float(a1/old_peak):.6f}倍（+{float(a1/old_peak-1)*100:.2f}%）。2.8897倍来自新峰时与旧空闲0.437345比较，不能称同口径上涨。", "",
             "按未来可用300元、留20%、DEV/EVAL各同N、仅A1估算，向下取整峰时94/空闲假设189例；同时计A0则92/184例。均未计G2/C/Embedding和真实平台价，不是正式N。全链19.536700元包含承接6.007656元；不得再加一次。", "",
             "2,734,592/4,946,806=55.28% 是已知 hit/总token（含输出），不是输入缓存命中率；已知 hit/已知 prompt=77.87%，也只描述有usage的子集。完整cache miss、全链缓存命中率及费用降幅保持null。", "",
             "## 停批与协议边界", "",
             "真实停批证据为单例预算/运输误分类终态导致完成率不可能达标的联合停止；DEV-25也在前两例即停。完整20例块中的契约>10%/传输>15%阈值尚无本轮真实触发证据，只在toy-9及离线测试中验证；分别披露。", "",
             "r2正式协议已禁止C加法，保留G1候选准入集合；G2/C只解释和重排。剪枝会改变该设计，只作为待批准修订建议，不在本次实施。排序能影响H2/H3的Top-5，却不会改变冻结A1的H1完整候选PPV。", "",
             "单supergroup、15个非随机前缀样本不能证明任一统计分量通过，也不能外推N再大必不通过；若更广样本重复相同方向，单纯增加样本量不会修复系统性的筛选/排序机制。正式抽样须预注册supergroup与reference_kind分层、覆盖和超额规则，不能继续使用此前缀当正式代表样本。", ""]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"out": str(out), "audit": audit, "budget": budget}, ensure_ascii=False))


if __name__ == "__main__":
    main()

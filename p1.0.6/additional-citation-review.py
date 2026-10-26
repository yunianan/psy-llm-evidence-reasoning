"""Post-review of extra citations, without changing the frozen node reference."""
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONTEXT_ONLY = {
    ("mb-006200", "screen.sleep", "mb-006200-p002"): "少睡且精力旺盛不是入睡/维持/早醒困难；同节点另引 p003 有明确入睡困难。",
    ("mb-007540", "screen.sleep", "mb-007540-p003"): "减少睡眠需要仅作背景；同节点 p002 的夜醒/不能再睡才是直接依据。",
    ("mb-014901", "screen.anxiety", "mb-014901-p002"): "惊跳/活动变化仅作背景，直接惊恐样呼吸困难在同节点 p001。",
    ("mb-014901", "screen.stress", "mb-014901-p002"): "当前训练和情绪变化需结合 p001/p003 的创伤与时间关系。",
    ("mb-017541", "screen.stress", "mb-017541-p002"): "当前空虚与行为变化需结合 p001 的丧偶后时间信息。",
    ("mb-017395", "screen.appearance", "mb-017395-p001"): "体重下降和进食变化不独立说明反复外观关注，主要依据是同节点 p002 的称重。",
    ("mb-006200", "screen.anxiety", "mb-006200-p002"): "既往背景焦虑与不太担忧并存，仅作背景；当前明确持续担忧在 p003。",
    ("mb-007587", "screen.stress", "mb-007587-p002"): "事故前后奇异时期的关联不独立确立当前反应，直接当前时间关系见同节点 p003。",
}


def main():
    rows = [json.loads(line) for line in (HERE / "nonpending-quote-review.jsonl").read_text(encoding="utf-8").splitlines()]
    output = []
    for row in rows:
        for ref in row["additional_unmatched_citations"]:
            key = (row["case_id"], row["node_id"], ref)
            output.append({"case_id": row["case_id"], "arm": row["arm"], "node_id": row["node_id"], "ref": ref,
                           "reviewer_type": "agent_manual_post_review", "classification": "context_only" if key in CONTEXT_ONLY else "additional_entry_support",
                           "note": CONTEXT_ONLY.get(key, "已结合完整患者段落复核，存在该入口的注意/情绪/担忧/重复关注/事件/进食线索；不要求达到疾病标准。"),
                           "quote": next(item["quote"] for item in row["cited_quotes"] if item["ref"] == ref)})
    (HERE / "additional-citation-review.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in output), encoding="utf-8")
    print(json.dumps({"extra_citation_occurrences": len(output), "classification_counts": dict(Counter(row["classification"] for row in output)),
                      "unique_case_node_refs": len({(row["case_id"], row["node_id"], row["ref"]) for row in output})}, ensure_ascii=True))


if __name__ == "__main__":
    main()

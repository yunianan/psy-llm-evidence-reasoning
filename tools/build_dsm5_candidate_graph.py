"""Build the project's sourced 24-label candidate subgraph; never edit MentalKG."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.disorder_catalog import SUPPORTED_DISORDERS
from shared.diagnostic_knowledge import candidate_knowledge
from layers.layer3.diagnosis.decision_tree import _symptom_pool


def build_graph():
    nodes, paths = {}, {}
    # Entry summaries are project paraphrases of First (2014), chapter 2.
    entries = {
        "attention": ("注意力易分散、学业困难或多动冲动的线索", "2.1/2.2/2.4/2.24", "22-29,35-37,122-125"),
        "psychosis": ("妄想、幻觉、言语或行为紊乱、阴性症状的线索", "2.5/2.6", "38-48"),
        "mood": ("心境高涨、易激惹、抑郁或兴趣下降的当前或既往线索", "2.8/2.9/2.10", "52-66"),
        "anxiety": ("焦虑、担忧、恐惧、惊恐或回避的线索", "2.13/2.14/2.15", "75-86"),
        "obsession": ("强迫观念、重复行为、焦虑或回避的线索", "2.13/2.15/2.25", "75-79,83-86,126-128"),
        "appearance": ("外貌或躯体相关的担忧和反复关注的线索", "2.13/2.17", "75-79,91-93"),
        "stress": ("创伤或应激之后的情绪、行为或回避线索", "2.16", "87-90"),
        "eating": ("进食、体重、体形、暴食或补偿行为的线索", "2.18/2.24", "94-98,122-125"),
        "sleep": ("入睡、睡眠维持或早醒困难的线索", "2.19", "99-103"),
    }
    book = lambda trees, pages: f"First2014:chapter2/trees/{trees}#print-pages={pages}"
    for key, (description, trees, pages) in entries.items():
        nodes[f"screen.{key}"] = {
            "stage": "screening", "question": f"画像是否存在{description}？只判断入口是否适用。",
            "follow_up": f"请补充是否存在{description}及相关病史。",
            "source_refs": [book(trees, pages)], "symptom_pool": [], "disorders": [],
        }
    groups = {
        "attention": ["D001", "D002", "D003"], "psychosis": ["D004", "D005", "D006", "D007"],
        "mood": ["D008", "D009", "D010", "D013", "D014", "D015"],
        "anxiety": ["D011", "D012"], "obsession": ["D016"], "appearance": ["D017"],
        "stress": ["D018", "D019", "D020"], "eating": ["D021", "D022", "D023"], "sleep": ["Insomnia"],
    }
    for group, labels in groups.items():
        for label in labels:
            knowledge = candidate_knowledge(label)
            # Recall anchor is an explicitly engineering admission rule, not a
            # replacement for the source's min_count or clinical criteria.
            pool = sorted(_symptom_pool(knowledge["criteria"]))
            anchor = f"anchor.{label}"
            nodes[anchor] = {
                "stage": "anchor", "question": "来源症状集合中是否至少有一个症状获得患者事实支持？此处仅召回候选，不表示满足诊断门槛。",
                "follow_up": "请描述相关症状的具体表现及发生情境。",
                "source_refs": [knowledge["source_ref"]], "symptom_pool": pool, "disorders": [label],
            }
            screens = [f"screen.{group}"]
            if label in {"D006", "D007"}: screens += ["screen.mood"]
            if label in {"D010", "D015"}: screens += ["screen.psychosis"]
            if label in {"D018", "D019"}: screens += ["screen.anxiety"]
            if label == "D020": screens += ["screen.mood", "screen.anxiety", "screen.sleep"]
            paths[label] = {"screens": screens, "anchors": [anchor], "guards": [],
                            "tree_refs": list(dict.fromkeys(ref for n in screens for ref in nodes[n]["source_refs"]))}
            for screen in screens:
                nodes[screen]["disorders"].append(label)

    def guard(key, question, refs, labels, expected="not_met"):
        nodes[key] = {"stage": "exclusion", "question": question,
                      "follow_up": "请补充本节点所需的发作史、症状时间关系和病因判断依据。",
                      "source_refs": refs, "symptom_pool": [], "disorders": labels}
        for label in labels:
            paths[label]["guards"].append({"node_id": key, "expected": expected})

    guard("history.mania", "是否有证据确认既往或当前满足完整躁狂发作条件（非仅高兴、易激惹、失眠或筛查阳性）？需考虑原发发作与物质/躯体因素的区别。",
          [book("2.8/2.10", "52-55,61-66"), "First2014:table3.3.2#print-pages=185-187"],
          ["D009", "D013", "D014", "D015", "D007"])
    guard("history.hypomania", "是否有证据确认既往或当前满足完整轻躁狂发作条件（非仅单个症状或筛查阳性）？",
          [book("2.10", "61-66"), "First2014:table3.4.1/3.4.2#print-pages=189-193"],
          ["D013", "D014", "D015"])
    nodes["history.mania"]["context_disorders"] = ["D008"]
    nodes["history.hypomania"]["context_disorders"] = ["D009"]
    guard("psychosis.only_mood", "是否已确认整个病程中精神病性症状只出现于完整心境发作期间？必须有纵向病史；未提到发作外症状不等于确认仅限发作内。",
          [book("2.5/2.6", "38-48"), "First2014:table3.2.2#print-pages=177"], ["D005", "D006", "D007"])
    # Scope causal exclusions to the presenting syndrome; merely having a
    # comorbid medical condition or substance use is not causal evidence.
    for group in ["psychosis", "mood", "anxiety", "obsession", "appearance", "stress"]:
        labels = groups[group]
        description, trees, pages = entries[group]
        guard(f"cause.{group}", f"对于本次所鉴别的{description}，是否已有明确临床证据表明其直接由物质/药物或另一躯体疾病引起？仅共存、用药史、物质使用史或时间相关不够；病因关系未知须 pending。",
              [book(trees, pages), "First2014:chapter1/steps2-3#print-pages=2-9"], labels)
    insomnia = candidate_knowledge("Insomnia")
    for code, key, question in [
        ("F", "sleep.other_disorder", "失眠是否被另一睡眠觉醒障碍更好解释且仅发生在该障碍病程中？"),
        ("G", "sleep.substance", "是否有明确证据表明睡眠困难可归因于物质或药物的生理效应？"),
        ("H", "sleep.explained", "共存精神或躯体疾病是否足以解释主要失眠主诉？仅共存不能回答是。"),
    ]:
        guard(key, question, [book("2.19", "99-103"), insomnia["criterion_source_refs"][code]], ["Insomnia"])
    return {
        "standard": "DSM-5", "version": "project-candidate-subgraph-3-insomnia-restored",
        "scope": "L3 Step 1; project adaptation, not all 29 original trees or author source code",
        "policy": {"unknown": "preserve branch; never infer absence", "admission": "one evidenced source symptom; engineering recall heuristic only",
                   "diagnosis": "endpoint remains candidate; downstream full criteria review required"},
        "nodes": nodes, "paths": {key: paths[key] for key in SUPPORTED_DISORDERS},
    }


if __name__ == "__main__":
    target = ROOT / "knowledge/dsm5_decision_tree/candidate_paths.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_graph(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)

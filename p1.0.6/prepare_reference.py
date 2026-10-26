"""Agent pre-audit against entry questions, written before either model arm runs."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Positive entry cues, not disorder labels or diagnostic criteria. Each tuple is
# (paragraph number, verbatim substring). All other entries remain unknown.
POSITIVE = {
    "mb-000750": {
        "attention": [(2, "get distracted")], "mood": [(3, "feel down")],
    },
    "mb-000820": {
        "attention": [(2, "distracted mid-thought")], "mood": [(3, "felt down lately")],
        "stress": [(3, "cyberbullying")],
    },
    "mb-001132": {
        "attention": [(2, "get sidetracked")],
        "anxiety": [(2, "want to run and hide")], "obsession": [(2, "want to run and hide")],
    },
    "mb-001248": {
        "attention": [(1, "get distracted")], "anxiety": [(3, "feel anxious a lot")],
        "obsession": [(3, "feel anxious a lot")],
        "stress": [(3, "especially about not having a job and about money")],
    },
    "mb-006200": {
        "attention": [(3, "lose track mid-conversation")], "mood": [(2, "on top of the world"), (3, "heavy emptiness")],
        "anxiety": [(3, "The worry is constant now")], "obsession": [(3, "The worry is constant now")],
        "sleep": [(3, "lie in bed for hours unable to fall asleep")],
    },
    "mb-007540": {
        "mood": [(2, "feel flat and sad"), (3, "overly cheerful")],
        "sleep": [(2, "wake up at 2 or 3 a.m.")],
    },
    "mb-007587": {
        "attention": [(1, "losing my focus")], "mood": [(1, "pervasive sadness"), (2, "on top of the world")],
        "anxiety": [(3, "constantly on edge")], "obsession": [(3, "constantly on edge")],
        "stress": [(3, "steady since that day")], "sleep": [(1, "waking up in the middle of the night")],
    },
    "mb-008771": {
        "attention": [(2, "attention on one thing")], "anxiety": [(2, "keep worrying")],
        "obsession": [(2, "keep worrying")], "stress": [(3, "tied to what happened")],
    },
    "mb-014901": {
        "mood": [(2, "go through the motions")], "anxiety": [(1, "like I can\u2019t breathe")],
        "obsession": [(1, "like I can\u2019t breathe")], "stress": [(1, "about that night at the trainee dorm")],
    },
    "mb-017376": {
        "attention": [(2, "hard to focus")], "mood": [(2, "easy to get irritated")],
        "anxiety": [(2, "the idea of gaining weight scares me")],
        "obsession": [(2, "the idea of gaining weight scares me")],
        "appearance": [(1, "balloon up overnight")], "eating": [(1, "stick to salads, plain chicken, black coffee")],
    },
    "mb-017395": {
        "mood": [(3, "enjoy things like I used to")], "obsession": [(2, "every morning")],
        "appearance": [(2, "stepping on the scale at home every morning")],
        "eating": [(1, "feel hungry like I used to")], "stress": [(1, "divorce talks started")],
    },
    "mb-017541": {
        "mood": [(2, "feel hollow")], "eating": [(1, "skip breakfast and only eat a few bites at dinner")],
        "stress": [(1, "since my husband passed three years ago")],
    },
    "mb-018770": {
        "mood": [(1, "I'm so down")], "sleep": [(1, "My sleep's been all over the place")],
        "eating": [(1, "huge amounts")],
    },
    "mb-020605": {"mood": [(1, "continuous depressive syndrome")]},
    "mb-020609": {
        "mood": [(1, "chronic depressive syndrome")], "sleep": [(1, "early morning awakening with an inability to return to sleep")],
    },
    "mb-022130": {
        "attention": [(3, "I pace a lot")], "psychosis": [(2, "hearing voices pretty regularly")],
        "mood": [(3, "snapping at people over nothing")],
    },
    "mb-022133": {
        "psychosis": [(2, "odd experiences with taste"), (5, "a fact I can\u2019t shake")],
        "mood": [(3, "feeling elated"), (4, "feel completely hollow")],
        "eating": [(4, "sit and eat mindlessly")],
    },
    "mb-023252": {
        "attention": [(1, "my focus")], "mood": [(1, "I just feel... flat")],
    },
    "mb-024466": {
        "attention": [(1, "so distracted")], "anxiety": [(2, "terrified of gaining weight")],
        "obsession": [(2, "terrified of gaining weight")], "appearance": [(2, "terrified of gaining weight")],
        "eating": [(2, "feeling totally out of control")],
    },
    "mb-024530": {
        "anxiety": [(3, "afford to gain weight")], "obsession": [(3, "punish myself at the gym")],
        "appearance": [(3, "softness in the mirror")], "eating": [(2, "absurd amount of food")],
        "stress": [(2, "after a stressful day")],
    },
}

NOTES = {
    "mb-001132": "任务回避按入口的回避线索宽召回，不视为焦虑或强迫诊断。",
    "mb-007540": "脑力疲惫不直接等同注意缺陷；既往创伤未明确解释当前起伏，不自动触发 stress。",
    "mb-014901": "练舞兴趣下降支持 mood；重复噩梦/惊醒未明确入睡、维持或早醒困难，sleep 暂 pending。",
    "mb-017395": "身体不适和不确定是否害怕未直接确认 anxiety；反复称重支持重复检查/外观关注入口。",
    "mb-017541": "否认体重恐惧不能推广为否认所有焦虑；丧偶后情绪与进食变化作为 stress 线索。",
    "mb-020605": "没有完整入口的明确反驳；未报告症状仍为 pending，不据单项否认躁狂排除 mood。",
    "mb-020609": "学业参与减少与动力低下不直接当作注意易分散；睡眠维持困难有明确描述。",
    "mb-022130": "少睡伴精力充沛不直接等于失眠；活动增加和踱步支持宽注意/多动入口。",
    "mb-022133": "少睡伴精力充沛不直接等于失眠；灾难确信明确被描述为非恐惧，不推定 anxiety。",
    "mb-023252": "充足睡眠不等于明确否认所有睡眠困难；没有该入口线索时仍 pending。",
    "mb-024530": "压力日后暴食按 stress 行为入口宽召回；体形挫败感不直接当作 mood。",
}


def main():
    destination = HERE / "reference.jsonl"
    assert not destination.exists(), "Do not overwrite pre-run references"
    assert not list(HERE.glob("runs/*/results.jsonl")), "References must precede both arms"
    inputs = [json.loads(line) for line in (HERE / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    graph = json.loads((ROOT / "L1-L5系统/knowledge/dsm5_decision_tree/candidate_paths.json").read_text(encoding="utf-8"))
    ids = list(dict.fromkeys(node for path in graph["paths"].values() for node in path["screens"]))
    assert set(POSITIVE) == {row["case_id"] for row in inputs}
    rows = []
    for row in inputs:
        texts = {ref: item["value"] for item in row["profile"]["key_narratives"] for ref in item["sources"]}
        for node in ids:
            cues = POSITIVE[row["case_id"]].get(node.removeprefix("screen."), [])
            evidence = []
            for paragraph, phrase in cues:
                ref = f"{row['case_id']}-p{paragraph:03d}"
                assert phrase in texts[ref], (row["case_id"], node, phrase)
                evidence.append({"ref": ref, "quote": phrase})
            rows.append({"case_id": row["case_id"], "node_id": node, "reference_status": "met" if cues else "pending",
                         "support": evidence, "case_note": NOTES.get(row["case_id"], "按入口原问句判线索；未提及不等于反证。"),
                         "reviewer_type": "agent_pre_audit", "gold_used": False})
    destination.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding="utf-8")
    metadata = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
                "reviewer_type": "agent_pre_audit_not_independent_clinician_gold", "reference_version": "screen-reference-v1",
                "case_count": len(inputs), "node_count": len(rows), "status_counts": dict(Counter(row["reference_status"] for row in rows)),
                "model_outputs_seen": False, "gold_used": False}
    (HERE / "reference-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=True))


if __name__ == "__main__":
    main()

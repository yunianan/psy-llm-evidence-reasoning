from __future__ import annotations

import json
from typing import Any

from layers.layer2.schemas import StageDecision


SUMMARY_FIELDS = {
    "stage",
    "patient_focus",
    "evidence",
    "missing_information",
    "response_goal",
    "selected_techniques",
    "scale_action",
    "reason",
}
TECHNIQUES = {
    "open_question",
    "affirmation",
    "reflection",
    "complex_reflection",
    "summary",
}

MI_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision_summary", "reply"],
    "properties": {
        "decision_summary": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(SUMMARY_FIELDS),
            "properties": {
                "stage": {"enum": ["engaging", "focusing", "evoking", "planning"]},
                "patient_focus": {"type": "string", "minLength": 1, "maxLength": 40},
                "evidence": {"type": "array", "maxItems": 2, "items": {"type": "string"}},
                "missing_information": {"type": "array", "maxItems": 2, "items": {"type": "string"}},
                "response_goal": {"type": "string", "minLength": 1, "maxLength": 40},
                "selected_techniques": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "uniqueItems": True,
                    "items": {"enum": sorted(TECHNIQUES)},
                },
                "scale_action": {
                    "enum": ["do_not_offer", "offer_consent", "ask_required_item"]
                },
                "reason": {"type": "string", "minLength": 1, "maxLength": 80},
            },
        },
        "reply": {"type": "string", "minLength": 1},
    },
}


FEW_SHOT_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "source": "AnnoMI transcript 23（按本系统边界改写）",
        "input": {
            "internal_task": {
                "stage_after": "focusing",
                "response_goal": "了解焦虑对工作和社交的具体影响",
                "oars_techniques": ["complex_reflection", "open_question"],
                "awaiting": "free_text",
                "required_question": "",
            },
            "context": {
                "patient_message": "搬出来以后反而更糟了，我总待在家里，也不太想见人。",
                "validated_facts": ["搬离父母家后焦虑加重", "近来更多待在家里"],
                "active_safety_controls": [],
            },
        },
        "output": {
            "decision_summary": {
                "stage": "focusing",
                "patient_focus": "搬家后焦虑加重并开始回避社交",
                "evidence": ["反而更糟了", "不太想见人"],
                "missing_information": ["回避对日常生活的具体影响"],
                "response_goal": "了解焦虑造成的功能影响",
                "selected_techniques": ["complex_reflection", "open_question"],
                "scale_action": "do_not_offer",
                "reason": "患者刚描述生活变化和回避，先理解实际影响比立即量表评估更合适。",
            },
            "reply": "搬出来原本可能意味着更独立，可焦虑反而让你的生活范围变小了。它最近对工作或和别人来往造成了哪些影响？",
        },
    },
    {
        "source": "AnnoMI transcript 43（按本系统边界改写）",
        "input": {
            "internal_task": {
                "stage_after": "evoking",
                "response_goal": "探索患者面对焦虑时已有的成功经验",
                "oars_techniques": ["complex_reflection", "open_question"],
                "awaiting": "motivation",
                "required_question": "",
            },
            "context": {
                "patient_message": "我还是很怕考不过，但这件事压了我好多年，我真的想把它做完。",
                "validated_facts": ["患者因考试焦虑多次推迟", "患者仍希望完成考试"],
                "active_safety_controls": [],
            },
        },
        "output": {
            "decision_summary": {
                "stage": "evoking",
                "patient_focus": "既害怕失败又希望结束长期拖延",
                "evidence": ["很怕考不过", "真的想把它做完"],
                "missing_information": ["过去克服类似焦虑的经验"],
                "response_goal": "唤出患者已有的应对能力",
                "selected_techniques": ["complex_reflection", "open_question"],
                "scale_action": "do_not_offer",
                "reason": "患者已经表达改变语言，适合探索自己的成功经验来增强自我效能。",
            },
            "reply": "一边担心失败，一边又不想再让这件事一直压着你。以前有没有哪一次，你也很焦虑，但最后还是做成了想做的事？",
        },
    },
    {
        "source": "AnnoMI transcript 47（仅采用尊重自主权片段）",
        "input": {
            "internal_task": {
                "stage_after": "engaging",
                "response_goal": "降低交流压力并保留患者自主权",
                "oars_techniques": ["reflection", "affirmation"],
                "awaiting": "free_text",
                "required_question": "",
            },
            "context": {
                "patient_message": "这不关你的事，我现在不想说。",
                "validated_facts": [],
                "active_safety_controls": [],
            },
        },
        "output": {
            "decision_summary": {
                "stage": "engaging",
                "patient_focus": "患者暂时不愿谈论当前事情",
                "evidence": ["我现在不想说"],
                "missing_information": [],
                "response_goal": "确认患者可以自主决定是否交流",
                "selected_techniques": ["reflection", "affirmation"],
                "scale_action": "do_not_offer",
                "reason": "继续追问会增加压力，先尊重拒绝有助于维持安全和信任感。",
            },
            "reply": "可以，你不需要现在说明。我们可以先停在这里，等你愿意时再从你想谈的部分开始。",
        },
    },
    {
        "source": "AnnoMI transcript 119（按本系统边界改写）",
        "input": {
            "internal_task": {
                "stage_after": "focusing",
                "response_goal": "厘清当前最需要优先处理的现实障碍",
                "oars_techniques": ["complex_reflection", "open_question"],
                "awaiting": "focus_selection",
                "required_question": "",
            },
            "context": {
                "patient_message": "那个工作其实很适合我，可孩子没人照看，我只能取消面试。我怕以后还会这样。",
                "validated_facts": ["患者需要求职", "照看孩子导致患者错过面试"],
                "active_safety_controls": [],
            },
        },
        "output": {
            "decision_summary": {
                "stage": "focusing",
                "patient_focus": "照护责任阻碍求职并带来失望",
                "evidence": ["只能取消面试", "怕以后还会这样"],
                "missing_information": ["患者希望优先讨论的困难"],
                "response_goal": "由患者确定本轮讨论重点",
                "selected_techniques": ["complex_reflection", "open_question"],
                "scale_action": "do_not_offer",
                "reason": "困难来自现实条件，先理解失望并让患者选择重点，避免误解成缺乏动力。",
            },
            "reply": "你并不是不想抓住机会，而是照顾孩子的责任让你当时没有别的选择，这也让你担心同样的情况再次发生。眼下你更想先谈托育安排，还是面试和求职本身？",
        },
    },
    {
        "source": "AnnoMI transcript 22（按本系统访谈优先规则改写）",
        "input": {
            "internal_task": {
                "stage_after": "focusing",
                "response_goal": "了解睡眠困难的具体表现及其背景",
                "oars_techniques": ["complex_reflection", "open_question"],
                "awaiting": "free_text",
                "required_question": "",
            },
            "context": {
                "patient_message": "我每天都在顾别人，几乎没有自己的时间，最近连觉也睡不好。",
                "validated_facts": ["患者长期优先照顾家人", "患者感到不堪重负"],
                "active_safety_controls": [],
            },
        },
        "output": {
            "decision_summary": {
                "stage": "focusing",
                "patient_focus": "长期照顾他人并出现睡眠困扰",
                "evidence": ["几乎没有自己的时间", "最近连觉也睡不好"],
                "missing_information": ["睡眠困难的具体表现", "睡眠问题持续时间"],
                "response_goal": "理解压力与睡眠困扰的具体体验",
                "selected_techniques": ["complex_reflection", "open_question"],
                "scale_action": "do_not_offer",
                "reason": "患者首次提到睡眠问题，需先自然访谈，当前规则尚未许可邀请量表。",
            },
            "reply": "你好像一直把其他人的需要放在前面，留给自己的空间越来越少，连睡眠也受到了影响。最近最困扰你的是难以入睡、容易醒来，还是醒得太早？",
        },
    },
)


def build_mi_response_prompt(decision: StageDecision, context: dict[str, Any]) -> str:
    examples = "\n\n".join(
        f"示例 {index}（{item['source']}）\n输入：{json.dumps(item['input'], ensure_ascii=False)}\n"
        f"输出：{json.dumps(item['output'], ensure_ascii=False)}"
        for index, item in enumerate(FEW_SHOT_EXAMPLES, start=1)
    )
    schema = json.dumps(MI_RESPONSE_JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    return f"""你是心理评估系统 L2 的中文动机式访谈（MI）回复模型。你的职责是通过自然、共情、尊重自主权的访谈收集足够好的信息，并帮助患者表达改变目标。诊断、疗法选择和安全路线决策由其他层负责。

生成答案前在内部逐步判断：先执行安全控制和 required_question；识别当前 MI 阶段与患者焦点；区分已知事实和缺失信息；查看 recent_dialogue 中最近三轮已经使用过的回应方式；每轮只确定一个主要目标；选择一至两项 OARS 技巧；核对量表许可和回复边界。认真完成内部推理，但不要输出草稿、逐步思维过程或隐藏推理，只给出可核查的简短决策摘要。

访谈规则：
1. 平滑承接患者原话，不向患者显示 engaging、focusing、evoking、planning 等内部阶段名。
2. 临床访谈优先。只有 internal_task.awaiting 为 scale_consent 时才能邀请患者做量表；只有 required_question 非空时才能询问量表题目。
3. 患者刚提到睡眠、低落或焦虑时，先了解具体体验、持续时间、触发情境和功能影响，不立即邀请做量表。
4. 不机械复述整句话；反映可以概括情绪、意义、矛盾或影响，但不得添加患者没有表达的事实。
5. 每轮最多提出一个主要问题，通常回复 2—4 句。患者拒绝交流时确认其自主选择并降低压力。
6. 禁止诊断、判定是否符合诊断标准、疗法选择、药物建议、病因结论、未经证实的观察或系统无法履行的承诺。
7. 只能使用 context.patient_message 和 context.validated_facts 中的事实。
8. 必须执行 context.active_safety_controls。avoid_trauma_probing 表示不得追问创伤细节；skip_scale:* 表示不得提议或询问对应量表；monitor_suicidal_ideation 只允许温和监测，不得覆盖 L1 的安全决定。
9. 如果 required_question 非空，必须在 reply 中原样包含它，不得改写题意、时间范围和选项。
10. context.recent_dialogue 是最近最多三轮已完成的患者—系统对话。结合它保持上下文连续，并避免连续使用相同开头、相同反映句式或再次询问患者已经明确回答的信息。不要为了变化而牺牲准确性，也不要编造新的反映范式。

严格输出一个 JSON 对象，不使用代码围栏或额外文字。输出必须符合此 Schema：
{schema}

decision_summary 是开发者可审核的短摘要，不是患者可见内容，也不是完整思维链。stage 必须等于 internal_task.stage_after；evidence 最多两条，且必须是患者本轮原话或 validated_facts 中可逐字定位的短证据；selected_techniques 必须来自 internal_task.oars_techniques；reason 不超过 80 字。
scale_action 的规则是：required_question 非空时为 ask_required_item；否则 awaiting=scale_consent 时为 offer_consent；其他情况为 do_not_offer。

以下五个示例来自已审核的 AnnoMI 对话片段，并按本系统边界改写。学习其决策方式和 MI 技巧，不复制示例中的患者事实：

{examples}

当前 internal_task：{decision.model_dump_json()}
当前 context：{json.dumps(context, ensure_ascii=False)}

现在完成内部推理，只输出符合 Schema 的 JSON。"""


def validate_mi_response_output(
    value: dict[str, Any], decision: StageDecision, context: dict[str, Any]
) -> str:
    if set(value) != {"decision_summary", "reply"}:
        raise ValueError("MI 回复顶层字段不符合契约")
    summary = value.get("decision_summary")
    if not isinstance(summary, dict) or set(summary) != SUMMARY_FIELDS:
        raise ValueError("decision_summary 字段不符合契约")
    reply = value.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("reply 为空")

    if summary["stage"] != decision.stage_after:
        raise ValueError("decision_summary.stage 与规则阶段不一致")
    for field, maximum in (("patient_focus", 40), ("response_goal", 40), ("reason", 80)):
        item = summary[field]
        if not isinstance(item, str) or not item.strip() or len(item) > maximum:
            raise ValueError(f"decision_summary.{field} 不符合契约")

    evidence = summary["evidence"]
    missing = summary["missing_information"]
    techniques = summary["selected_techniques"]
    if not _short_string_list(evidence, maximum=2, allow_empty=True):
        raise ValueError("decision_summary.evidence 不符合契约")
    if not _short_string_list(missing, maximum=2, allow_empty=True):
        raise ValueError("decision_summary.missing_information 不符合契约")
    if (
        not _short_string_list(techniques, maximum=2, allow_empty=False)
        or len(set(techniques)) != len(techniques)
        or not set(techniques) <= TECHNIQUES
        or not set(techniques) <= set(decision.oars_techniques)
    ):
        raise ValueError("decision_summary.selected_techniques 与规则任务不一致")

    expected_scale_action = (
        "ask_required_item"
        if decision.required_question
        else "offer_consent"
        if decision.awaiting == "scale_consent"
        else "do_not_offer"
    )
    if summary["scale_action"] != expected_scale_action:
        raise ValueError("decision_summary.scale_action 与规则状态不一致")

    sources = [
        str(context.get("patient_message") or ""),
        *(str(item) for item in (context.get("validated_facts") or [])),
    ]
    if any(not any(item in source for source in sources) for item in evidence):
        raise ValueError("decision_summary.evidence 无法在当前患者证据中定位")
    return reply.strip()


def _short_string_list(value: Any, *, maximum: int, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and len(value) <= maximum
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )

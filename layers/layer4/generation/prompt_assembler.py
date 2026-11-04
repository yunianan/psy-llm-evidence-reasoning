import json

from layers.layer4.llm.base import Message
from layers.layer4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    TurnInstruction,
)

_SYSTEM_INTRO = (
    "你是一名受过CBT训练的心理治疗师。你的唯一任务是把诊断层给出的治疗指令"
    "翻译为自然、温暖、专业的治疗对话。\n"
    "你是语言生成引擎，不做任何临床推理：不诊断、不调整治疗方案、不评判指令。"
)


def assemble_prompt(
    *,
    role: RoleInstruction,
    anchor: DiagnosticAnchor,
    boundary: BoundaryInstruction,
    turn: TurnInstruction,
    history: list[Message],
    user_message: str,
    fix_instruction: str | None = None,
) -> list[Message]:
    """四段式组装 prompt（文档第 6 节）：系统(L1+L2+L4) / 本轮(L3+插件) / 上下文 / 当前。"""
    plugin_trigger_text = "；".join(
        f"{item.get('plugin', '')}: {', '.join(item.get('conditions', []))}"
        for item in anchor.plugin_triggers
    ) or "无"
    adjustment_text = "；".join(
        anchor.strategy_adjustments.get("instructions", [])
    ) or "无"
    diagnosis_name = anchor.primary_label
    verification = (
        f"候选证据支持度：{anchor.candidate_confidence}\n"
        f"复核判定：{anchor.verifier_decision}"
    )
    system_parts = [_SYSTEM_INTRO]
    system_parts.append(f"【角色定义】(L1)\n{role.identity}\n风格：{role.style}\n边界：{'；'.join(role.boundaries)}")
    system_parts.append(
        "【诊断锚点】(L2)\n"
        f"主诊断：{diagnosis_name}（{anchor.diagnostic_standard} {anchor.primary}）\n"
        f"基础疗法：{anchor.therapy_type}\n"
        f"诊断与鉴别排除详情：{json.dumps(anchor.diagnosis_details, ensure_ascii=False)}\n"
        f"量表：{json.dumps(anchor.scale_scores, ensure_ascii=False)}\n"
        f"共病：{', '.join(anchor.comorbidities) if anchor.comorbidities else '无'}\n"
        f"已排除：{', '.join(anchor.excluded) if anchor.excluded else '无'}\n"
        f"核心信念：{', '.join(anchor.core_beliefs) if anchor.core_beliefs else '无'}\n"
        f"认知模式（未标注为核心信念时勿推断）：{json.dumps(anchor.cognitive_patterns, ensure_ascii=False)}\n"
        f"触发情境与关键叙述（原始依据）：{json.dumps(anchor.triggering_situations, ensure_ascii=False)}\n"
        f"关键叙述（不可一律视为触发情境）：{json.dumps(anchor.key_narratives, ensure_ascii=False)}\n"
        f"患者目标：{'；'.join(anchor.patient_goals)}\n"
        f"已激活插件：{', '.join(anchor.plugins) if anchor.plugins else '无'}\n"
        f"插件触发依据：{plugin_trigger_text}\n"
        f"治疗方案微调：{adjustment_text}\n"
        f"禁忌与转介：{'；'.join(anchor.contraindications) if anchor.contraindications else '无'}\n"
        f"{verification}；反馈措辞：{anchor.wording_constraint}"
    )
    bans = "\n".join(f"- {b}" for b in boundary.absolute_bans) or "- 无"
    system_parts.append(
        "【防越界指令】(L4)\n"
        f"绝对禁止：\n{bans}\n"
        f"安全触发：{boundary.safety_trigger or '无'}"
    )
    system_parts.append(
        '【输出格式】请只输出一个 JSON 对象，不要包含任何其他文字，格式：\n'
        '{"reply": "治疗师回复正文"}\n'
        '只执行本轮指令，不判断阶段是否完成，不自行推进治疗。对患者平滑衔接，不播报内部阶段。'
    )

    user_parts = [
        "【本轮指令】(L3)",
        f"目标：{turn.goal}",
        f"技术：{turn.technique}",
        f"禁止：{', '.join(turn.forbidden) if turn.forbidden else '无'}",
        f"衔接上轮：{turn.link_previous or '无'}",
        "插件嵌入指导：",
        turn.plugin_guidance or "无",
    ]
    if history:
        lines = [f"{'用户' if m.role == 'user' else '治疗师'}：{m.content}" for m in history]
        user_parts.append("【对话历史】\n" + "\n".join(lines))
    else:
        user_parts.append("【对话历史】（无）")
    user_parts.append(f"【当前消息】\n用户：{user_message}")
    user_parts.append("请生成治疗师回复（JSON）。")
    if fix_instruction:
        user_parts.append(f"【重写要求（审核反馈）】\n{fix_instruction}\n请根据以上要求重写回复，输出 JSON。")

    return [
        Message(role="system", content="\n\n".join(system_parts)),
        Message(role="user", content="\n\n".join(user_parts)),
    ]

from layers.layer4.generation.prompt_assembler import assemble_prompt
from layers.layer4.generation.responder import GenerationResult, Responder
from layers.layer4.generation.summary_builder import (
    build_evidence_refs,
    build_l5_context,
    build_summary,
)
from layers.layer4.llm.base import Message
from layers.layer4.llm.mock import MockBackend
from layers.layer4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from layers.layer4.session.history import ConversationTurn
from shared.schemas.treatment_instructions import TreatmentInstructions

ROLE = RoleInstruction(identity="你是一名受过CBT训练的心理治疗师", style="温暖、共情、结构化")
ANCHOR = DiagnosticAnchor(
    primary="F40.2", primary_label="特定恐怖症", comorbidities=["F51.0"],
    excluded=["F40.0"], plugins=["CBT-I"], candidate_confidence=.75,
    verifier_decision="clinician_confirmed", therapy_type="CBT",
)
BOUNDARY = BoundaryInstruction(absolute_bans=["禁止开药", "禁止诊断新疾病", "禁止预测未来"],
                               safety_trigger="用户表达自杀意图→立即触发安全协议")


def _turn(**kw) -> TurnInstruction:
    defaults = dict(goal="探索自动化思维", technique="苏格拉底式提问",
                    forbidden=["建议", "确诊"], link_previous="上次你提到……")
    defaults.update(kw)
    return TurnInstruction(**defaults)


def test_assemble_prompt_four_sections():
    msgs = assemble_prompt(role=ROLE, anchor=ANCHOR, boundary=BOUNDARY,
                           turn=_turn(), history=[], user_message="我这周又加班到凌晨")
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    sys_text = msgs[0].content
    # L1 角色
    assert "你是一名受过CBT训练的心理治疗师" in sys_text
    # L2 诊断锚点
    assert "特定恐怖症（WHO ICD-10 F40.2）" in sys_text
    assert "候选证据支持度：0.75" in sys_text
    assert "复核判定：clinician_confirmed" in sys_text
    # L4 防越界
    assert "禁止开药" in sys_text
    user_text = msgs[1].content
    # L3 本轮指令
    assert "苏格拉底式提问" in user_text
    # 上下文与当前消息
    assert "我这周又加班到凌晨" in user_text


def test_assemble_prompt_history_and_plugin():
    history = [Message(role="user", content="你好"),
               Message(role="assistant", content="今天想聊什么？")]
    msgs = assemble_prompt(role=ROLE, anchor=ANCHOR, boundary=BOUNDARY,
                           turn=_turn(plugin_guidance="本轮开场嵌入3分钟呼吸空间"),
                           history=history, user_message="我睡不着")
    user_text = msgs[1].content
    assert "3分钟呼吸空间" in user_text
    assert "今天想聊什么" in user_text


def test_assemble_prompt_fix_instruction_appended_at_end():
    msgs = assemble_prompt(role=ROLE, anchor=ANCHOR, boundary=BOUNDARY,
                           turn=_turn(), history=[], user_message="hi",
                           fix_instruction="删除确定性措辞，改为初步印象")
    user_text = msgs[1].content
    assert "删除确定性措辞，改为初步印象" in user_text
    # 重写要求块位于"当前消息"之后（所有内容之后）
    assert user_text.index("删除确定性措辞，改为初步印象") > user_text.index("用户：hi")


def test_responder_parses_json():
    backend = MockBackend(
        default='{"reply": "听起来工作给你带来了很多负担。", "stage_complete": true, "reason": "目标达成"}'
    )
    resp = Responder(backend, ROLE, ANCHOR, BOUNDARY)
    gen = resp.generate(_turn(), history=[], user_message="最近压力很大")
    assert gen == GenerationResult(reply="听起来工作给你带来了很多负担。",
                                   stage_complete=True, reason="目标达成")


def test_responder_json_parse_failure_falls_back_to_text():
    backend = MockBackend(default="这不是JSON，只是一句普通的回复")
    resp = Responder(backend, ROLE, ANCHOR, BOUNDARY)
    gen = resp.generate(_turn(), history=[], user_message="hi")
    assert gen.reply == "这不是JSON，只是一句普通的回复"
    assert gen.stage_complete is False
    assert gen.reason == "parse_fallback"


def test_responder_forwards_fix_instruction():
    backend = MockBackend(responses={"删除确定性措辞": '{"reply": "改写后回复", "stage_complete": false, "reason": ""}'})
    resp = Responder(backend, ROLE, ANCHOR, BOUNDARY)
    gen = resp.generate(_turn(), history=[], user_message="hi",
                        fix_instruction="删除确定性措辞，改为初步印象")
    assert gen.reply == "改写后回复"


def test_build_l5_context_fields():
    safety = SafetyPassport(risk_level=2, sensitive_topics=["trauma"],
                            flags=["monitor_suicidal_ideation"])
    instructions = TreatmentInstructions(L1=ROLE, L2=ANCHOR, L3=_turn(), L4=BOUNDARY)
    ctx = build_l5_context("你好", "L4-L2", safety, instructions, "我最近睡不好", [])
    assert ctx.l4_raw_output == "你好"
    assert ctx.l4_substage == "L4-L2"
    assert ctx.therapy_type == "CBT"
    assert ctx.diagnosis_context.primary == "F40.2"
    assert ctx.diagnosis_context.secondary == ["F51.0"]
    assert ctx.diagnosis_context.suicidal_ideation == "passive_only"
    assert ctx.risk_context.risk_level == 2
    assert ctx.risk_context.sensitive_topics == ["trauma"]


def test_build_evidence_refs_non_empty():
    history = [ConversationTurn(role="user", content="我躺床上两三个小时睡不着"),
               ConversationTurn(role="assistant", content="听起来很辛苦")]
    refs = build_evidence_refs(_turn(), history,
                               SafetyPassport(flags=["monitor_suicidal_ideation"]))
    sources = {r.source for r in refs}
    assert "L3_turn_instruction" in sources
    assert "conversation_history" in sources
    assert "L1_safety_passport" in sources


def test_build_summary_fields():
    refs = [{"source": "L3_turn_instruction", "finding": "technique=苏格拉底式提问", "basis": "目标"}]
    s = build_summary(_turn(), "L4-L3", "测试回复", refs, confidence=0.85,
                      flags=["stage_regression_override"])
    assert s.layer_id == "L4"
    assert s.structured_labels["substage"] == "L4-L3"
    assert s.structured_labels["goal"] == "探索自动化思维"
    assert s.confidence == 0.85
    assert "stage_regression_override" in s.flags

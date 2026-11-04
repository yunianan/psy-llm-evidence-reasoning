"""四段指令专项回归：受控模型不代表临床金标准。"""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from layers.layer3.instructions.instruction_generator import InstructionGenerator
from layers.layer3.output_validator import L3OutputValidator
from layers.layer3.instructions.turn_planner import (
    APITreatmentPlanner,
    TreatmentPlanningAPIError,
    TreatmentPlanningOutputInvalid,
    TreatmentPlanningUnavailable,
    TurnInstructionPlanner,
)
from layers.layer4.config import Settings
from layers.layer4.llm.mock import MockBackend
from layers.layer4.review.mock_reviewer import MockReviewer
from layers.layer4.schemas.l3_report import DiagnosticReport
from layers.layer4.service import L4Service, SessionRequest, TurnRequest
from shared.schemas.treatment_instructions import SourceQuote, TreatmentInstructions
from tests.instruction_test_double import ControlledTreatmentModel, controlled_turn


class RecordingBackend(MockBackend):
    def __init__(self):
        super().__init__(default='{"reply":"听起来这很重要，你愿意多说一些吗？","stage_complete":true}')
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        return super().chat(messages, **kwargs)


@pytest.fixture
def raw():
    return json.loads((Path(__file__).resolve().parents[1] / "examples/sample_diagnostic_report.json").read_text(encoding="utf-8"))


@pytest.fixture
def service(raw):
    service = L4Service(Settings(_env_file=None), backend=RecordingBackend(), reviewer=MockReviewer())
    service.create_session(SessionRequest(session_id="s", report=DiagnosticReport.model_validate(raw)))
    return service


def run_turn(service, message, **overrides):
    turn = controlled_turn(service, "s", message, **overrides)
    outcome = service.handle_turn(TurnRequest(session_id="s", user_message=message, turn=turn))
    return turn, outcome


def test_static_anchor_preserves_evidence_and_separate_scores(raw):
    profile = {
        "Y_BOCS": 18, "PHQ9": 16,
        "cognitive_patterns": [
            {"kind": "core_belief", "value": "我不够好", "sources": ["e1"], "patient_quotes": ["我不够好"]},
            {"value": "灾难化思维", "sources": ["e2"], "patient_quotes": ["肯定会失败"]},
        ],
        "key_narratives": [{"kind": "triggering_situation", "value": "收到邮件后崩溃", "sources": ["e3"]}],
        "patient_goals": ["恢复工作"],
    }
    bundle = InstructionGenerator().generate(raw, profile)
    assert bundle.L2.scale_scores["Y-BOCS"] == 18
    assert bundle.L2.core_beliefs == ["我不够好"]
    assert bundle.L2.cognitive_patterns == profile["cognitive_patterns"]
    assert bundle.L2.triggering_situations == profile["key_narratives"]
    assert bundle.L2.diagnosis_details["ruled_out"] == raw["diagnosis"]["ruled_out"]
    assert bundle.L2.plugin_triggers == raw["recommended_therapy"]["plugin_triggers"]
    assert bundle.L2.candidate_confidence == raw["candidate_confidence"]
    assert bundle.L2.verifier_score == raw["verifier_score"]
    assert bundle.L3 is None
    raw["instructions"] = bundle.model_dump(mode="json")
    assert DiagnosticReport.model_validate(raw).to_anchor() == bundle.L2


def test_two_rounds_use_new_context_but_reuse_static_instructions(service):
    sd = service.store.get("s")
    static = sd.instructions().model_dump()
    first, _ = run_turn(service, "邮件之后很难受")
    second, _ = run_turn(service, "我发现那时想到自己不够好")
    assert first.goal != second.goal
    assert first.turn_number == 1 and second.turn_number == 2
    assert "邮件之后" in second.link_previous
    assert sd.instructions().model_dump() == static
    assert sd.machine.current == "L4-L1"  # L4 模型 stage_complete=true 不推进。
    assert len(sd.treatment_records()) == 2
    assert service.backend.calls[0][0] == service.backend.calls[1][0]
    assert "不播报内部阶段" in service.backend.calls[-1][0].content


def test_api_planner_uses_context_and_schema(monkeypatch, service):
    monkeypatch.delenv("L3_INSTRUCTION_TIMEOUT_SECONDS", raising=False)
    captured = []
    def fake_api(prompt, **kwargs):
        captured.append((prompt, kwargs))
        context = json.loads(prompt.split("输入数据：\n", 1)[1])
        return "```json\n" + ControlledTreatmentModel().propose(context).model_dump_json() + "\n```"
    monkeypatch.setattr("layers.layer3.instructions.turn_planner.ask_llm", fake_api)
    planner = TurnInstructionPlanner(APITreatmentPlanner(model="test-planner"))
    sd = service.store.get("s")
    turn = planner.plan(session_id="s", instructions=sd.instructions(), records=[], user_message="工作压力", safety={})
    assert turn.source_message == "工作压力"
    assert captured[0][1] == {"model": "test-planner", "temperature": 0, "timeout": 60.0}
    assert "禁止重新诊断" in captured[0][0]
    assert "allowed_techniques" in captured[0][0]


@pytest.mark.parametrize("response", ["not json", "{}", "[]", '{"goal":null}', "```json\n{\n```"])
def test_bad_api_response_fails_closed(monkeypatch, response, service):
    monkeypatch.setattr("layers.layer3.instructions.turn_planner.ask_llm", lambda *a, **kw: response)
    sd = service.store.get("s")
    with pytest.raises(TreatmentPlanningOutputInvalid):
        TurnInstructionPlanner().plan(session_id="s", instructions=sd.instructions(), records=[], user_message="难过", safety={})
    assert service.backend.calls == []


def test_api_timeout_fails_closed(monkeypatch, service):
    def timeout(*args, **kwargs):
        raise TimeoutError("timeout")
    monkeypatch.setattr("layers.layer3.instructions.turn_planner.ask_llm", timeout)
    sd = service.store.get("s")
    with pytest.raises(TreatmentPlanningAPIError):
        TurnInstructionPlanner().plan(session_id="s", instructions=sd.instructions(), records=[], user_message="难过", safety={})
    assert service.backend.calls == []


def test_instruction_timeout_can_be_overridden(monkeypatch, service):
    monkeypatch.setenv("L3_INSTRUCTION_TIMEOUT_SECONDS", "90")
    captured = {}

    def fake_api(prompt, **kwargs):
        captured.update(kwargs)
        context = json.loads(prompt.split("输入数据：\n", 1)[1])
        return ControlledTreatmentModel().propose(context).model_dump_json()

    monkeypatch.setattr("layers.layer3.instructions.turn_planner.ask_llm", fake_api)
    sd = service.store.get("s")
    TurnInstructionPlanner().plan(
        session_id="s",
        instructions=sd.instructions(),
        records=[],
        user_message="难过",
        safety={},
    )
    assert captured["timeout"] == 90.0


@pytest.mark.parametrize("change,match", [
    ({"session_id": "another"}, "会话"),
    ({"turn_number": 2}, "轮次"),
    ({"source_message": "别的消息"}, "消息"),
    ({"technique_code": "emdr", "technique": "EMDR"}, "白名单"),
    ({"plugin_steps": {"EMDR": "开始"}, "plugin_guidance": "EMDR：开始"}, "未激活"),
    ({"goal": "开始 EMDR"}, "未授权"),
    ({"forbidden": []}, "防越界"),
    ({"force_substage": "L4-L3"}, "首轮"),
    ({"evidence": [SourceQuote(source_id="turn:9:user", quote="难过")]}, "不存在"),
    ({"evidence": [SourceQuote(source_id="turn:1:user", quote="编造的原话")]}, "不存在"),
])
def test_l4_guard_blocks_invalid_instructions_before_generation(service, change, match):
    turn = controlled_turn(service, "s", "难过").model_copy(update=change)
    with pytest.raises(ValueError, match=match):
        service.handle_turn(TurnRequest(session_id="s", user_message="难过", turn=turn))
    assert not service.backend.calls
    assert not service.store.get("s").history


def test_replaying_previous_instruction_is_rejected(service):
    first, _ = run_turn(service, "难过")
    with pytest.raises(ValueError, match="轮次"):
        service.handle_turn(TurnRequest(session_id="s", user_message="难过", turn=first))
    assert len(service.backend.calls) == 1


def test_stage_transition_requires_readiness_and_cannot_skip(service):
    run_turn(service, "压力大")
    with pytest.raises(ValueError, match="阶段推进"):
        controlled_turn(service, "s", "仍然很难受", stage="L4-L2")
    with pytest.raises(ValueError, match="跳级"):
        controlled_turn(service, "s", "准备好了", stage="L4-L3", previous_goal_met=True, patient_ready=True)
    second, _ = run_turn(service, "你理解了我的困扰，我愿意听听反馈", stage="L4-L2", previous_goal_met=True, patient_ready=True, technique_code="feedback")
    assert second.force_substage == "L4-L2"
    assert service.store.get("s").machine.current == "L4-L2"


def test_plugin_guidance_sent_only_for_activated_plugin(service):
    run_turn(service, "困扰睡眠")
    run_turn(service, "你理解了，我愿意听反馈", stage="L4-L2", previous_goal_met=True, patient_ready=True)
    turn, _ = run_turn(service, "我理解了反馈，愿意开始练习", stage="L4-L3", previous_goal_met=True, patient_ready=True,
                       technique_code="sleep_hygiene", plugin_steps={"CBT-I": "先讨论睡前作息，再商量一个小调整"})
    assert turn.plugin_guidance == "CBT-I：先讨论睡前作息，再商量一个小调整"
    assert turn.plugin_guidance in service.backend.calls[-1][1].content


@pytest.mark.parametrize("field,value", [("primary", "F41.1"), ("candidate_confidence", .01), ("verifier_score", .01), ("plugins", ["EMDR"])])
def test_l3_static_validation_detects_inconsistent_anchor(raw, field, value):
    raw["instructions"]["L2"][field] = value
    receipt = L3OutputValidator().validate(raw)
    assert any(item["rule_id"] == "V3-TREATMENT-INSTRUCTIONS" and item["status"] == "fail" for item in receipt["checks"])


def test_integrity_guard_covers_final_instructions(raw):
    raw["instructions"]["L1"]["style"] = "篡改后的角色"
    service = L4Service(Settings(_env_file=None), backend=RecordingBackend(), reviewer=MockReviewer())
    with pytest.raises(ValueError, match="INTEGRITY"):
        service.create_session(SessionRequest(report=DiagnosticReport.model_validate(raw)))


def test_fallback_is_not_treatment_completion(service):
    service.reviewer = MockReviewer(always_fail=True)
    _, outcome = run_turn(service, "压力大")
    assert outcome.fallback_used
    with pytest.raises(ValueError, match="阶段推进"):
        controlled_turn(service, "s", "继续", stage="L4-L2", previous_goal_met=True, patient_ready=True)


def test_new_l1_risk_blocks_before_planner_call(service):
    sd = service.store.get("s")
    model = ControlledTreatmentModel()
    with pytest.raises(ValueError, match="安全"):
        TurnInstructionPlanner(model).plan(session_id="s", instructions=sd.instructions(), records=[], user_message="危机", safety={"risk_level": 3})
    assert model.calls == []


def test_density_and_contraindications_are_enforced(service):
    sd = service.store.get("s")
    sd.anchor.plugins = ["CBT-I", "MBCT"]
    sd.anchor.strategy_adjustments["technical_density"] = "reduced"
    with pytest.raises(ValueError, match="技术密度"):
        controlled_turn(service, "s", "难过", plugin_steps={"CBT-I": "睡眠练习", "MBCT": "正念练习"})
    sd.anchor.strategy_adjustments["care_priority"] = "stabilization_first"
    with pytest.raises(ValueError, match="稳定化"):
        controlled_turn(service, "s", "难过", plugin_steps={"CBT-I": "睡眠练习"})


def test_current_message_whitespace_is_not_silently_changed(service):
    turn, _ = run_turn(service, "  有点难过  ")
    assert turn.source_message == "  有点难过  "


def test_rewrite_cannot_change_the_original_plan(service):
    first, _ = run_turn(service, "难过")
    with pytest.raises(ValueError, match="重写必须沿用"):
        service.handle_turn(TurnRequest(session_id="s", user_message="难过", turn=first.model_copy(update={"goal": "不同目标"})), persist=False)


@pytest.mark.parametrize("goal", ["本轮开药", "确诊新的疾病", "保证治愈", "安排暴露练习"])
def test_action_text_cannot_override_boundaries(service, goal):
    service.store.get("s").anchor.strategy_adjustments["challenge_interventions"] = "deferred"
    with pytest.raises(ValueError, match="禁止|暂缓"):
        controlled_turn(service, "s", "我很难受", goal=goal)


def test_top_pipeline_replans_each_round_without_rediagnosis(monkeypatch, tmp_path):
    from tests.test_icd10_orchestrator import PENDING, review_for, start_session

    pipeline, sid, _ = start_session(monkeypatch, tmp_path)
    pipeline.confirm_patient_profile(sid, "continue")
    for _, text in PENDING.values():
        pipeline.handle_message(sid, text)
    tree = pipeline.get_session_snapshot(sid).l3_raw["model_result"]
    pipeline.submit_icd10_review(sid, review_for(tree["session_id"]))

    def unexpected_diagnosis(*args, **kwargs):
        raise AssertionError("治疗每轮不应重新运行诊断树")

    pipeline.icd10_tree_factory = unexpected_diagnosis
    assert "小行动" in pipeline.handle_message(sid, "我很在意工作").reply
    assert "小行动" in pipeline.handle_message(sid, "今天的邮件让我想起上次失败").reply
    calls = pipeline.instruction_planner.model.calls
    assert len(calls) == 2
    assert calls[1]["progress"][0]["user_message"] == "我很在意工作"
    sd = pipeline.l4.store.get(sid)
    assert len(sd.l3_history) == 2
    assert sd.l3_history[0].goal != sd.l3_history[1].goal
    crisis = pipeline.handle_message(sid, "立即危机")
    assert crisis.terminal
    assert len(calls) == 2


def test_top_pipeline_model_failure_does_not_call_l4_or_commit_progress(monkeypatch, tmp_path, raw):
    from tests.test_top_level_orchestrator import _safe_passport, _orchestrator
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    pipeline = _orchestrator(tmp_path)
    pipeline.create_session("failure")
    pipeline._start_l4(pipeline.store.get("failure"), DiagnosticReport.model_validate(raw))
    def unavailable(context):
        raise TreatmentPlanningUnavailable("模型超时")
    monkeypatch.setattr(pipeline.instruction_planner.model, "propose", unavailable)
    def unexpected_l4(*args, **kwargs):
        raise AssertionError("无合法指令时不得调用 L4")
    monkeypatch.setattr(pipeline.l4, "handle_turn", unexpected_l4)
    result = pipeline.handle_message("failure", "仍然很困扰")
    assert "本轮未进入治疗生成" in result.reply
    assert pipeline.l4.store.get("failure").treatment_records() == []

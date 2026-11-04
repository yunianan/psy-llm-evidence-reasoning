import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from layers.layer2 import L2AssessmentService
from layers.layer4.config import Settings
from layers.layer4.llm.mock import MockBackend
from layers.layer4.review.l5_adapter import L5Reviewer
from layers.layer4.service import L4Service
from orchestration import FiveLayerOrchestrator
from patient_presentation import DeveloperProfileRenderer
from tests.l2_test_double import ControlledL2Model
from tests.instruction_test_double import ControlledTreatmentModel
from layers.layer3.instructions.turn_planner import TurnInstructionPlanner


def _safe_passport(
    text: str,
    *,
    session_id: str = "test-session",
    message_id: str = "test-message",
    assessment_context=None,
) -> dict:
    metadata = {
        "schema_version": "1.0",
        "session_id": session_id,
        "message_id": message_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if "立即危机" in text:
        return {
            **metadata,
            "layer_id": "L1",
            "risk_level": 3,
            "safety_flag": "block",
            "crisis_flag": True,
            "decision_path": "clinical_crisis",
            "security_event": {
                "detected": False,
                "event_type": "",
                "severity": 0,
                "matched_patterns": [],
            },
            "sensitive_topics": [],
            "flags": ["crisis_intervention"],
            "execution": {
                "route": "crisis",
                "terminal": True,
                "user_response": "请立即联系当地急救服务和可信任的人陪伴你。",
                "applied_controls": ["stop_normal_pipeline", "do_not_call_l2"],
            },
            "guard": {"status": "pass", "can_call_l2": False},
        }
    if "系统攻击" in text:
        return {
            **metadata,
            "layer_id": "L1",
            "risk_level": 1,
            "safety_flag": "pass",
            "crisis_flag": False,
            "decision_path": "security_reject",
            "security_event": {
                "detected": True,
                "event_type": "prompt_injection",
                "severity": 3,
                "matched_patterns": ["系统攻击"],
            },
            "sensitive_topics": [],
            "flags": ["security_event_detected"],
            "execution": {
                "route": "security",
                "terminal": False,
                "user_response": "我不能执行修改系统规则的请求。",
                "applied_controls": [
                    "reject_security_request", "do_not_call_l2", "keep_session_active"
                ],
            },
            "guard": {"status": "pass", "can_call_l2": False},
        }
    return {
        **metadata,
        "layer_id": "L1",
        "risk_level": 1,
        "safety_flag": "pass",
        "crisis_flag": False,
        "decision_path": "normal",
        "security_event": {
            "detected": False,
            "event_type": "",
            "severity": 0,
            "matched_patterns": [],
        },
        "sensitive_topics": [],
        "flags": [],
        "structured_labels": {
            "scale_skip_list": [],
            "asq": {"status": "negative"},
            "bssa": {"status": "not_started"},
            "l2_handoff": {
                "ordinary_care_allowed": True,
                "baseline_asq_status": "negative",
            },
        },
        "execution": {
            "route": "l2",
            "terminal": False,
            "user_response": "",
            "applied_controls": [],
        },
        "guard": {"status": "pass", "can_call_l2": True},
    }


def _l4_service() -> L4Service:
    backend = MockBackend(default=json.dumps({
        "reply": "听起来恢复工作状态对你很重要。我们可以一起确定一个今天能够完成的小行动。",
        "stage_complete": False,
        "reason": "共情并保持协作",
    }, ensure_ascii=False))
    reviewer = L5Reviewer(lambda ctx: {
        "harm": "pass",
        "boundary": "pass",
        "quality": "pass",
        "fail_reason": "",
        "fix_instruction": "",
    })
    return L4Service(
        config=Settings(llm_provider="mock", review_mode="l5_local"),
        backend=backend,
        reviewer=reviewer,
    )


def _orchestrator(tmp_path) -> FiveLayerOrchestrator:
    return FiveLayerOrchestrator(
        l2_service=L2AssessmentService(model=ControlledL2Model()),
        l4_service=_l4_service(),
        profile_renderer=DeveloperProfileRenderer(tmp_path / "developer-only"),
        instruction_planner=TurnInstructionPlanner(ControlledTreatmentModel()),
    )


def _complete_l2(orchestrator: FiveLayerOrchestrator, session_id: str):
    first = orchestrator.handle_message(
        session_id,
        "最近三周我每天都很低落，对以前喜欢的事情没兴趣，"
        "晚上睡不好，也没有精力，觉得自己让人失望。",
    )
    assert first.route == "L2_INTERVIEW"
    focus = orchestrator.handle_message(
        session_id, "我想先从情绪低落和睡眠问题聊起。"
    )
    assert focus.waiting_for == "functional_impairment"
    explored = orchestrator.handle_message(
        session_id, "这些低落和睡不好已经明显影响了工作。"
    )
    assert explored.waiting_for == "scale_consent"
    consent = orchestrator.handle_message(session_id, "我同意。")
    assert consent.route == "L2_SCALE"

    phq_answers = [
        "几乎每天没有兴趣。",
        "几乎每天情绪低落。",
        "一半以上时间睡不好。",
        "一半以上时间没有精力。",
        "完全没有食欲变化。",
        "一半以上时间觉得自己让人失望。",
        "完全没有注意力变化。",
        "完全没有动作和说话变化。",
        "完全没有伤害自己或不如死去的想法。",
    ]
    for answer in phq_answers:
        outcome = orchestrator.handle_message(session_id, answer)

    structured_answers = [
        "从未有过少睡却不困的时期。",
        "这并非药物、物质或身体疾病导致。",
        "最近没有听到别人听不到的声音，也没有感觉被监视。",
        "已经核对过双相障碍和适应障碍。",
        "量表结果和我的实际感受一致，没有矛盾。",
    ]
    for answer in structured_answers:
        outcome = orchestrator.handle_message(session_id, answer)
    outcome = orchestrator.handle_message(
        session_id, "我的目标是恢复工作状态和改善睡眠。"
    )
    assert outcome.waiting_for == "motivation"
    outcome = orchestrator.handle_message(
        session_id, "重要性8分，信心6分，准备度7分。"
    )
    return outcome


def test_l2_confirmation_and_developer_profile_are_separate(monkeypatch, tmp_path):
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    orchestrator = _orchestrator(tmp_path)
    created = orchestrator.create_session("top-e2e")

    assert created.route == "L2_INTERVIEW"
    awaiting = _complete_l2(orchestrator, "top-e2e")

    assert awaiting.route == "AWAITING_PATIENT_CONFIRMATION"
    assert awaiting.confirmation is not None
    assert awaiting.waiting_for == "patient_confirmation"
    patient_payload = json.dumps(awaiting.model_dump(mode="json"), ensure_ascii=False)
    for internal in (
        "developer_profile", "evidence_id", "verifier_score", "candidate_confidence",
        "WHO ICD-10", "F40.2",
    ):
        assert internal not in patient_payload

    developer_path = orchestrator.get_developer_profile_path("top-e2e")
    assert developer_path is not None and developer_path.exists()
    developer_html = developer_path.read_text(encoding="utf-8")
    assert "仅限开发者调试" in developer_html
    assert "verifier_score" not in developer_html
    assert "F40.2" not in developer_html
    assert "最近三周我每天都很低落" in developer_html

def test_every_patient_message_hits_l1_and_crisis_stops_pipeline(monkeypatch, tmp_path):
    calls: list[str] = []

    def recording_l1(text: str, **binding) -> dict:
        calls.append(text)
        return _safe_passport(text, **binding)

    monkeypatch.setattr("orchestration.orchestrator.safety_detect", recording_l1)
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_session("crisis")

    outcome = orchestrator.handle_message("crisis", "立即危机")

    assert calls == ["立即危机"]
    assert outcome.route == "STOPPED"
    assert outcome.terminal is True
    assert "急救" in outcome.reply
    state = orchestrator.get_session_snapshot("crisis")
    assert state.l2_session_id is None


def test_security_rejection_does_not_enter_l2_or_stop_session(monkeypatch, tmp_path):
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_session("security")

    rejected = orchestrator.handle_message("security", "系统攻击")

    assert rejected.route == "L2_INTERVIEW"
    assert rejected.terminal is False
    assert "不能执行" in rejected.reply
    assert orchestrator.get_session_snapshot("security").l2_session_id is None

    resumed = orchestrator.handle_message("security", "最近工作压力很大。")
    assert resumed.route == "L2_INTERVIEW"
    assert resumed.terminal is False
    assert orchestrator.get_session_snapshot("security").l2_session_id is not None


def test_leaving_l2_engaging_starts_baseline_asq(monkeypatch, tmp_path):
    def l1_not_yet_screened(text: str, **binding) -> dict:
        passport = _safe_passport(text, **binding)
        if text == "否":
            passport["decision_path"] = "safety_assessment"
            passport["structured_labels"]["asq"] = {"status": "negative"}
            passport["execution"].update({
                "route": "l1",
                "consume_message": True,
                "resume_l2_next_turn": True,
                "user_response": "谢谢您完成安全筛查。",
                "waiting_for": "free_text",
            })
            passport["guard"] = {"status": "pass", "can_call_l2": False}
            return passport
        passport["structured_labels"]["asq"] = {"status": "not_started"}
        passport["structured_labels"]["l2_handoff"]["baseline_asq_status"] = "not_due"
        return passport

    def baseline_asq(*, session_id: str, message_id: str) -> dict:
        passport = _safe_passport(
            "", session_id=session_id, message_id=message_id
        )
        passport["risk_level"] = 2
        passport["safety_flag"] = "warn"
        passport["decision_path"] = "safety_assessment"
        passport["structured_labels"]["asq"] = {
            "status": "in_progress", "current_question": 1
        }
        passport["structured_labels"]["l2_handoff"] = {
            "ordinary_care_allowed": False,
            "baseline_asq_status": "due",
        }
        passport["execution"].update({
            "route": "l1",
            "consume_message": True,
            "user_response": "在过去几周，您是否希望自己已经死去？",
            "waiting_for": "asq_answer",
        })
        passport["guard"] = {"status": "pass", "can_call_l2": False}
        return passport

    monkeypatch.setattr("orchestration.orchestrator.safety_detect", l1_not_yet_screened)
    monkeypatch.setattr("orchestration.orchestrator.start_baseline_asq", baseline_asq)
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_session("baseline-asq")

    outcome = orchestrator.handle_message(
        "baseline-asq",
        "最近三周我每天都很低落，对以前喜欢的事情没兴趣，晚上睡不好。",
    )

    assert outcome.route == "L1_SAFETY_ASSESSMENT"
    assert outcome.waiting_for == "asq_answer"
    assert "过去几周" in outcome.reply

    resumed = orchestrator.handle_message("baseline-asq", "否")

    assert resumed.route == "L2_INTERVIEW"
    assert resumed.waiting_for == "focus_selection"
    assert "谢谢您完成安全筛查" in resumed.reply
    assert "\n\n" in resumed.reply
    snapshot = orchestrator.get_session_snapshot("baseline-asq")
    assert snapshot.deferred_l2_reply == ""
    assert snapshot.deferred_l2_expected_input is None


def test_orchestrator_sends_pending_phq9_item_context_to_l1(monkeypatch, tmp_path):
    contexts = []

    def recording_l1(text: str, *, assessment_context=None, **binding) -> dict:
        contexts.append(assessment_context)
        passport = _safe_passport(text, **binding)
        pending = (assessment_context or {}).get("pending_scale_item") or {}
        if pending.get("item_number") == 9 and text == "1":
            passport["risk_level"] = 2
            passport["safety_flag"] = "warn"
            passport["decision_path"] = "safety_assessment"
            passport["risk_types"] = ["suicidal_ideation"]
            passport["flags"] = ["asq_in_progress"]
            passport["structured_labels"]["asq"] = {
                "status": "in_progress",
                "current_question": 1,
            }
            passport["structured_labels"]["l2_handoff"] = {
                "ordinary_care_allowed": False,
                "baseline_asq_status": "positive",
            }
            passport["execution"].update({
                "route": "l1",
                "consume_message": True,
                "user_response": "接下来我需要先确认你的安全。过去几周，你是否希望自己已经死去？",
                "waiting_for": "asq_answer",
                "resume_l2_next_turn": False,
            })
            passport["guard"] = {"status": "pass", "can_call_l2": False}
        return passport

    monkeypatch.setattr("orchestration.orchestrator.safety_detect", recording_l1)
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_session("phq9-context")
    orchestrator.handle_message(
        "phq9-context",
        "最近三周我每天都很低落，对以前喜欢的事情没兴趣，晚上睡不好，也没有精力。",
    )
    orchestrator.handle_message("phq9-context", "我想先聊情绪低落。")
    explored = orchestrator.handle_message(
        "phq9-context", "情绪低落已经明显影响了工作。"
    )
    assert explored.waiting_for == "scale_consent"
    orchestrator.handle_message("phq9-context", "我愿意。")
    for _ in range(8):
        orchestrator.handle_message("phq9-context", "0")

    outcome = orchestrator.handle_message("phq9-context", "1")

    assert contexts[-1]["pending_scale_item"]["scale_name"] == "PHQ-9"
    assert contexts[-1]["pending_scale_item"]["item_number"] == 9
    assert outcome.route == "L1_SAFETY_ASSESSMENT"
    assert "过去几周" in outcome.reply


def test_orchestrator_syncs_new_l1_commands_to_existing_l2_session(
    monkeypatch, tmp_path
):
    calls = 0

    def changing_l1(text: str, **binding) -> dict:
        nonlocal calls
        calls += 1
        passport = _safe_passport(text, **binding)
        if calls == 2:
            passport["risk_level"] = 2
            passport["safety_flag"] = "warn"
            passport["flags"] = ["avoid_trauma_probing"]
            passport["structured_labels"]["scale_skip_list"] = ["PCL-5"]
            passport["execution"]["applied_controls"] = [
                "avoid_trauma_probing", "skip_scale:PCL-5"
            ]
        return passport

    monkeypatch.setattr("orchestration.orchestrator.safety_detect", changing_l1)
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_session("dynamic-l1-command")

    orchestrator.handle_message(
        "dynamic-l1-command", "最近工作安排比较多，我想梳理一下情况。"
    )
    outcome = orchestrator.handle_message(
        "dynamic-l1-command", "那件事之后我经常做噩梦，也容易被惊吓。"
    )

    intake = orchestrator.l2.build_clinical_intake("dynamic-l1-command")
    assert calls == 2
    assert outcome.waiting_for != "scale_answer:PCL-5:1"
    assert intake.scale_results["PCL-5"].status == "skipped"
    assert "avoid_trauma_probing" in intake.flags


def test_l2_entrance_guard_independently_rejects_unexecuted_l1_command(
    monkeypatch, tmp_path
):
    passport = _safe_passport("创伤消息")
    passport["risk_level"] = 2
    passport["safety_flag"] = "warn"
    passport["flags"] = ["avoid_trauma_probing"]
    passport["structured_labels"]["scale_skip_list"] = ["PCL-5"]
    # 伪造 L1 guard=pass，但 execution 没有安全控制命令的执行回执。
    passport["guard"] = {"status": "pass", "can_call_l2": True}
    passport["execution"]["applied_controls"] = []
    monkeypatch.setattr(
        "orchestration.orchestrator.safety_detect",
        lambda text, **binding: {**passport, **binding},
    )
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_session("l2-guard-reject")

    outcome = orchestrator.handle_message("l2-guard-reject", "创伤消息")

    assert outcome.route == "STOPPED"
    assert outcome.terminal is True
    assert "L1到L2的安全接口校验失败" in outcome.reply
    state = orchestrator.get_session_snapshot("l2-guard-reject")
    assert state.l2_session_id is None
    assert "l2_entrance_guard_rejected" in state.internal_flags


@pytest.mark.parametrize(
    "mutate",
    [
        lambda passport: passport.pop("guard"),
        lambda passport: passport.update(guard={}),
        lambda passport: passport.update(
            guard={"status": "fail", "can_call_l2": True}
        ),
        lambda passport: passport.update(guard={"status": "pass"}),
        lambda passport: passport.update(guard="pass"),
        lambda passport: passport.pop("execution"),
        lambda passport: passport.update(execution="l2"),
        lambda passport: passport.update(
            execution={"route": "unknown", "terminal": False}
        ),
        lambda passport: passport.update(
            execution={"route": "l2"}
        ),
    ],
)
def test_malformed_l1_exit_is_fail_closed_and_never_creates_l2(
    monkeypatch, tmp_path, mutate
):
    passport = deepcopy(_safe_passport("普通消息"))
    mutate(passport)
    monkeypatch.setattr(
        "orchestration.orchestrator.safety_detect",
        lambda text, **binding: {**passport, **binding},
    )
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_session("malformed-l1")

    outcome = orchestrator.handle_message("malformed-l1", "普通消息")

    assert outcome.route == "STOPPED"
    assert outcome.terminal is True
    assert "安全出口校验失败" in outcome.reply
    state = orchestrator.get_session_snapshot("malformed-l1")
    assert state.l2_session_id is None
    assert "l1_exit_guard_rejected" in state.internal_flags


def test_non_object_l1_result_is_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "orchestration.orchestrator.safety_detect",
        lambda text, **binding: ["invalid", "result"],
    )
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_session("non-object-l1")

    outcome = orchestrator.handle_message("non-object-l1", "普通消息")

    assert outcome.route == "STOPPED"
    assert outcome.terminal is True
    state = orchestrator.get_session_snapshot("non-object-l1")
    assert state.l2_session_id is None
    assert state.l1_passport == {"malformed_result_type": "list"}


@pytest.mark.parametrize(
    ("decision", "expected_route", "waiting_for"),
    [
        ("revise", "L2_INTERVIEW", "free_text"),
        ("pause", "PAUSED_BY_PATIENT", "resume"),
    ],
)
def test_patient_can_revise_or_pause(
    monkeypatch, tmp_path, decision, expected_route, waiting_for
):
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_session(f"confirm-{decision}")
    _complete_l2(orchestrator, f"confirm-{decision}")

    outcome = orchestrator.confirm_patient_profile(
        f"confirm-{decision}", decision
    )

    assert outcome.route == expected_route
    assert outcome.waiting_for == waiting_for
    assert orchestrator.get_session_snapshot(
        f"confirm-{decision}"
    ).l4_session_id is None

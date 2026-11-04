from __future__ import annotations

import pytest

import live_layer_test
from layers.layer1.service import L1SafetyService
from tests.l2_test_double import ControlledL2Model


@pytest.fixture(autouse=True)
def isolate_profile_output(monkeypatch, tmp_path):
    monkeypatch.setattr(live_layer_test, "PROJECT_ROOT", tmp_path)


def _scripted_input(monkeypatch, answers: list[str]):
    remaining = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(remaining))
    return remaining


def _l2_answers_before_confirmation() -> list[str]:
    return [
        "最近三周我每天都很低落，对以前喜欢的事情没兴趣，晚上睡不好，也没有精力，觉得自己让人失望。",
        "我想先从情绪低落和睡眠问题聊起。",
        "这些低落和睡不好已经明显影响了工作。",
        "我同意。",
        *(f"{score}分" for score in [3, 3, 2, 2, 0, 2, 0, 0, 0]),
        "从未有过少睡却不困的时期。",
        "这些变化并非药物、物质或身体疾病导致。",
        "最近没有听到别人听不到的声音，也没有感觉被监视。",
        "我的目标是恢复工作状态和改善睡眠。",
        "重要性8分，信心6分，准备度7分。",
    ]


def test_l1_independent_run_reaches_safety_exit_after_asq(monkeypatch, capsys):
    service = L1SafetyService()

    def detect(text, *, session_id, message_id):
        return service.handle_message(
            text,
            session_id=session_id,
            message_id=message_id,
            classifier=lambda *_args: {
                "analysis": {
                    "suicide_signal_detected": True,
                    "jailbreak_detected": False,
                    "safe_to_enter_l2": False,
                    "confidence": 0.95,
                    "evidence_quotes": [text],
                    "jailbreak_evidence": [],
                    "asq_transition_support": "谢谢您愿意告诉我。",
                    "summary": "需要进一步筛查",
                }
            },
        )

    monkeypatch.setattr("layers.layer1.detector.safety_detect", detect)
    monkeypatch.setattr(
        "layers.layer1.service.generate_support",
        lambda **_kwargs: ("谢谢您告诉我，我们先联系立即帮助。", "test"),
    )
    remaining = _scripted_input(
        monkeypatch,
        ["我最近有过自杀的想法", "是", "否", "否", "否", "是"],
    )

    live_layer_test.test_l1()

    output = capsys.readouterr().out
    assert "[L1 出口：普通疗程暂停" in output
    assert "路由：safety" in output
    assert '"status": "acute_positive"' in output
    with pytest.raises(StopIteration):
        next(remaining)


def test_l2_independent_run_reaches_confirmed_exit_guard(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("layers.layer2.llm.APIL2Model", ControlledL2Model)
    remaining = _scripted_input(
        monkeypatch,
        _l2_answers_before_confirmation() + ["这些信息基本准确，继续。"],
    )

    live_layer_test.test_l2()

    output = capsys.readouterr().out
    assert "[L2 出口：画像已确认，L2→L3 守卫通过]" in output
    assert '"PHQ9": 12' in output
    profiles = list((tmp_path / "artifacts" / "developer_profiles").glob("*.html"))
    assert len(profiles) == 1
    document = profiles[0].read_text(encoding="utf-8")
    assert '&quot;patient_confirmation&quot;: &quot;confirmed&quot;' in document
    assert "L2 患者画像" in document
    with pytest.raises(StopIteration):
        next(remaining)


def test_l2_pending_profile_can_open_without_another_model_turn(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("layers.layer2.llm.APIL2Model", ControlledL2Model)
    opened = []
    monkeypatch.setattr(live_layer_test.webbrowser, "open", lambda uri: opened.append(uri) or True)
    _scripted_input(monkeypatch, _l2_answers_before_confirmation() + ["/profile", "/exit"])

    live_layer_test.test_l2()

    profiles = list((tmp_path / "artifacts" / "developer_profiles").glob("*.html"))
    assert len(profiles) == 1
    assert opened == [profiles[0].as_uri()]
    document = profiles[0].read_text(encoding="utf-8")
    assert "AWAITING_PATIENT_CONFIRMATION" in document
    assert '&quot;patient_confirmation&quot;: &quot;pending&quot;' in document
    output = capsys.readouterr().out
    assert output.count("正在调用真实 LLM") == len(_l2_answers_before_confirmation())


def test_l2_revision_keeps_running_until_reconfirmed(monkeypatch, capsys):
    monkeypatch.setattr("layers.layer2.llm.APIL2Model", ControlledL2Model)
    remaining = _scripted_input(
        monkeypatch,
        _l2_answers_before_confirmation()
        + [
            "画像需要修改。",
            "我的目标不是恢复工作状态，而是先恢复稳定睡眠。",
            "这些信息基本准确，继续。",
        ],
    )

    live_layer_test.test_l2()

    output = capsys.readouterr().out
    assert "请说明画像中需要修改的具体信息" in output
    assert "[L2 出口：画像已确认，L2→L3 守卫通过]" in output
    assert "我的目标不是恢复工作状态，而是先恢复稳定睡眠。" in output
    with pytest.raises(StopIteration):
        next(remaining)

import json
import logging

import pytest

import main as entry
from layers.layer2.llm import client
from layers.layer2.schemas import StageDecision
from tests.test_top_level_orchestrator import _orchestrator, _safe_passport, _complete_l2


@pytest.mark.parametrize("phase", ["analyze", "generate"])
@pytest.mark.parametrize("confirming", [False, True])
def test_model_failure_stops_session_including_confirmation(monkeypatch, tmp_path, phase, confirming):
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    pipeline = _orchestrator(tmp_path)
    sid = pipeline.create_session().session_id
    if confirming:
        _complete_l2(pipeline, sid)
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(phase)
        raise client.L2ModelUnavailable("sensitive-error-text")

    monkeypatch.setattr(pipeline.l2.model, phase, unavailable)
    monkeypatch.setattr(pipeline, "icd10_tree_factory", lambda *a, **kw: pytest.fail("不可进入 L3"))
    before = pipeline.l2.get_session_snapshot(sid) if confirming else None
    result = pipeline.handle_message(sid, "信息准确，可以继续。" if confirming else "最近很低落。")
    assert result.route == "STOPPED"
    assert result.terminal is True
    assert result.waiting_for is None and result.confirmation is None
    assert "请稍后重试" not in result.reply
    assert "sensitive-error-text" not in result.reply
    after = pipeline.l2.get_session_snapshot(sid)
    assert after.messages == (before.messages if before else [])
    assert after.turn_index == (before.turn_index if before else 0)
    assert pipeline.handle_message(sid, "继续").terminal is True
    assert calls == [phase]


def test_terminal_does_not_prompt_again_after_l2_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    pipeline = _orchestrator(tmp_path)

    def unavailable(*a, **kw):
        raise client.L2ModelUnavailable("test")

    monkeypatch.setattr(pipeline.l2.model, "analyze", unavailable)
    prompts = []

    def patient_input(prompt):
        prompts.append(prompt)
        assert len(prompts) == 1, "模型失败后不应再次读取患者输入"
        return "最近很低落。"

    monkeypatch.setattr("builtins.input", patient_input)
    entry.run_session(pipeline)
    output = capsys.readouterr().out
    assert "STOPPED" in output and "本次终端会话已结束" in output
    assert len(prompts) == 1


def _invoke_api_model(operation):
    model = client.APIL2Model()
    if operation == "analyze":
        return model.analyze("patient-private-text", {})
    decision = StageDecision(
        stage_before="engaging", stage_after="engaging", transition_reason="test",
        response_goal="test", oars_techniques=["reflection"],
    )
    return model.generate(decision, {})


def _patch_model_call(monkeypatch, operation, implementation):
    name = "ask_llm_stream" if operation == "analyze" else "ask_llm"
    monkeypatch.setattr(client, name, implementation)


def _valid_generation_response(*, reply="我听到你最近承受了很多。", stage="engaging"):
    return json.dumps({
        "decision_summary": {
            "stage": stage,
            "patient_focus": "当前困扰",
            "evidence": [],
            "missing_information": [],
            "response_goal": "理解患者当前困扰",
            "selected_techniques": ["reflection"],
            "scale_action": "do_not_offer",
            "reason": "先通过反映建立理解并继续收集信息。",
        },
        "reply": reply,
    }, ensure_ascii=False)


@pytest.mark.parametrize("operation", ["analyze", "generate"])
def test_l2_uses_sixty_second_timeout_by_default(monkeypatch, operation):
    calls = []

    def valid_response(*args, **kwargs):
        calls.append(kwargs)
        if operation == "analyze":
            return json.dumps({"input_type": "free_narrative"})
        return _valid_generation_response()

    monkeypatch.delenv("L2_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3.8-max")
    _patch_model_call(monkeypatch, operation, valid_response)
    _invoke_api_model(operation)
    expected = {"timeout": 60.0}
    if operation == "analyze":
        expected["extra_body"] = {"enable_thinking": False}
    else:
        expected["extra_body"] = {
            "enable_thinking": True,
            "thinking_budget": 4096,
        }
    assert calls == [expected]


def test_semantic_extraction_and_repair_disable_thinking_on_dashscope(monkeypatch):
    calls = []
    responses = iter([
        '{"input_type":"free_narrative","symptoms":[{"value":"fatigue"}]}',
        '{"input_type":"free_narrative","symptoms":[]}',
    ])

    def respond(*args, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    result = client.APIL2Model().analyze("整个人都没劲。", {})

    assert result.symptoms == []
    assert calls == [
        {"timeout": 60.0, "extra_body": {"enable_thinking": False}},
        {"timeout": 60.0, "extra_body": {"enable_thinking": False}},
    ]


def test_semantic_extraction_does_not_send_dashscope_option_to_other_providers(monkeypatch):
    calls = []
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr(
        client,
        "ask_llm_stream",
        lambda *args, **kwargs: calls.append(kwargs)
        or json.dumps({"input_type": "free_narrative"}),
    )

    _invoke_api_model("analyze")

    assert calls == [{"timeout": 60.0}]


def test_mi_response_uses_low_thinking_budget_for_dashscope_qwen38(monkeypatch):
    calls = []
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3.8-max")
    monkeypatch.setattr(
        client,
        "ask_llm",
        lambda *args, **kwargs: calls.append(kwargs) or _valid_generation_response(),
    )

    _invoke_api_model("generate")

    assert calls == [{
        "timeout": 60.0,
        "extra_body": {"enable_thinking": True, "thinking_budget": 4096},
    }]


def test_mi_response_does_not_send_qwen38_option_to_other_providers(monkeypatch):
    calls = []
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setattr(
        client,
        "ask_llm",
        lambda *args, **kwargs: calls.append(kwargs) or _valid_generation_response(),
    )

    _invoke_api_model("generate")

    assert calls == [{"timeout": 60.0}]


def test_semantic_prompt_uses_live_schema_and_explicitly_defines_three_failed_fields(monkeypatch):
    captured = []

    def valid_response(prompt, **kwargs):
        captured.append(prompt)
        return json.dumps({"input_type": "free_narrative"})

    monkeypatch.setattr(client, "ask_llm_stream", valid_response)
    _invoke_api_model("analyze")
    prompt = captured[0]
    compact_schema = json.dumps(
        client.SemanticAnalysis.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert compact_schema in prompt
    assert "WHO ICD-10" in prompt
    assert '"careless_errors": "漏看细节或因粗心犯错"' in prompt
    assert "MentalKG" not in prompt and '"S001"' not in prompt
    assert '["PHQ-9", "ISI"]' in prompt
    assert '"field":"functional_impairment"' in prompt
    assert '["需要确认睡眠情况", "需要确认食欲变化"]' in prompt
    assert "clarification_needed 必须是字符串数组" in prompt


def test_three_observed_schema_errors_are_repaired_once(monkeypatch, caplog):
    calls = []
    invalid = {
        "input_type": "free_narrative",
        "possible_scale_domains": ["PHQ9"],
        "structured_fields": [{
            "name": "functional_impairment", "value": True,
            "evidence_quote": "工作不想做",
        }],
        "clarification_needed": [{"question": "需要确认睡眠情况"}],
    }
    repaired = {
        "input_type": "free_narrative",
        "possible_scale_domains": ["PHQ-9"],
        "structured_fields": [{
            "field": "functional_impairment", "value": True,
            "evidence_quote": "工作不想做",
        }],
        "clarification_needed": ["需要确认睡眠情况"],
    }
    responses = iter([json.dumps(invalid), json.dumps(repaired)])

    def respond(prompt, **kwargs):
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr(client, "ask_llm_stream", respond)
    with caplog.at_level(logging.WARNING):
        result = client.APIL2Model().analyze(
            "最近整个人没劲，工作不想做。", {}
        )
    assert result.possible_scale_domains == ["PHQ-9"]
    assert result.structured_fields[0].field == "functional_impairment"
    assert result.clarification_needed == ["需要确认睡眠情况"]
    assert len(calls) == 2
    assert "L2 JSON 格式修复器" in calls[1]
    assert "L2_OUTPUT_REPAIR" in caplog.text
    assert "L2_MODEL_UNAVAILABLE" not in caplog.text


def test_repair_may_not_invent_missing_evidence_quote(monkeypatch):
    responses = iter([
        '{"input_type":"free_narrative","symptoms":[{"value":"fatigue"}]}',
        '{"input_type":"free_narrative","symptoms":[]}',
    ])
    prompts = []

    def respond(prompt, **kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(client, "ask_llm_stream", respond)
    result = client.APIL2Model().analyze("整个人都没劲。", {})
    assert result.symptoms == []
    assert "无法绑定原话的候选必须删除，禁止编造" in prompts[1]


def test_two_invalid_schema_outputs_have_distinct_error(monkeypatch):
    monkeypatch.setattr(
        client,
        "ask_llm_stream",
        lambda *a, **kw: '{"input_type":"free_narrative","symptoms":[{"value":"fatigue"}]}',
    )
    with pytest.raises(client.L2ModelOutputInvalid):
        _invoke_api_model("analyze")

    from layers.layer2 import L2AssessmentService
    from tests.test_layer2_assessment import safe_passport

    service = L2AssessmentService(model=client.APIL2Model(), require_fresh_passport=False)
    sid = service.create_session(safe_passport("invalid-output"), session_id="invalid-output")
    result = service.handle_message(sid, "整个人都没劲。")
    assert result.error_code == "L2_MODEL_OUTPUT_INVALID"
    assert "返回的数据格式不符合要求" in result.reply
    assert "模型暂时不可用" not in result.reply


def test_l2_timeout_can_be_configured_without_changing_shared_default(monkeypatch):
    calls = []
    monkeypatch.setenv("L2_LLM_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr(
        client,
        "ask_llm_stream",
        lambda *args, **kwargs: calls.append(kwargs) or json.dumps({"input_type": "free_narrative"}),
    )
    _invoke_api_model("analyze")
    assert calls == [{"timeout": 90.0}]


@pytest.mark.parametrize("value", ["invalid", "0", "301"])
def test_invalid_l2_timeout_fails_closed_with_safe_diagnostic(monkeypatch, caplog, value):
    monkeypatch.setenv("L2_LLM_TIMEOUT_SECONDS", value)
    monkeypatch.setattr(client, "ask_llm_stream", lambda *a, **kw: pytest.fail("配置无效时不能请求 API"))
    with caplog.at_level(logging.ERROR), pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("analyze")
    assert "L2.analyze.api_request" in caplog.text
    assert "ValueError" in caplog.text


@pytest.mark.parametrize("operation", ["analyze", "generate"])
def test_timeout_reports_stage_but_not_exception_body(monkeypatch, caplog, operation):
    def timeout(*args, **kwargs):
        raise TimeoutError("secret-key patient-private-text")

    _patch_model_call(monkeypatch, operation, timeout)
    with caplog.at_level(logging.ERROR), pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model(operation)
    assert f"L2.{operation}.api_request" in caplog.text
    assert "TimeoutError" in caplog.text
    assert "secret-key" not in caplog.text and "patient-private-text" not in caplog.text


@pytest.mark.parametrize("operation", ["analyze", "generate"])
def test_invalid_json_reports_parse_stage_without_raw_output(monkeypatch, caplog, operation):
    _patch_model_call(monkeypatch, operation, lambda *_, **__: "secret-key patient-private-text")
    with caplog.at_level(logging.WARNING), pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model(operation)
    assert f"L2.{operation}.json_parse" in caplog.text
    assert "JSONDecodeError" in caplog.text
    assert "secret-key" not in caplog.text and "patient-private-text" not in caplog.text


def test_schema_failure_reports_missing_field_and_redacts_unknown_keys(monkeypatch, caplog):
    invalid = {
        "input_type": "free_narrative",
        "symptoms": [{"value": "fatigue"}],  # 缺少 evidence_quote
        "secret-key": "patient-private-text",
    }
    monkeypatch.setattr(client, "ask_llm_stream", lambda *_, **__: json.dumps(invalid))
    with caplog.at_level(logging.WARNING), pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("analyze")
    assert "L2.analyze.schema_validation" in caplog.text
    assert '"path": ["symptoms", 0, "evidence_quote"]' in caplog.text
    assert '"type": "missing"' in caplog.text
    assert "<unknown_field>" in caplog.text
    assert "secret-key" not in caplog.text and "patient-private-text" not in caplog.text


def test_empty_reply_reports_reply_validation(monkeypatch, caplog):
    monkeypatch.setattr(
        client, "ask_llm", lambda *_, **__: _valid_generation_response(reply="")
    )
    with caplog.at_level(logging.ERROR), pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("generate")
    assert "L2.generate.schema_validation" in caplog.text


def test_mi_generation_prompt_contains_five_examples_and_reasoning_contract(monkeypatch):
    prompts = []

    def respond(prompt, **kwargs):
        prompts.append(prompt)
        return _valid_generation_response()

    monkeypatch.setattr(client, "ask_llm", respond)
    assert _invoke_api_model("generate") == "我听到你最近承受了很多。"
    prompt = prompts[0]
    for transcript_id in (23, 43, 47, 119, 22):
        assert f"AnnoMI transcript {transcript_id}" in prompt
    assert "内部逐步判断" in prompt
    assert "不要输出草稿、逐步思维过程或隐藏推理" in prompt
    assert '"decision_summary"' in prompt and '"reply"' in prompt
    assert "临床访谈优先" in prompt


def test_mi_generation_rejects_reply_without_decision_summary(monkeypatch, caplog):
    monkeypatch.setattr(client, "ask_llm", lambda *_, **__: '{"reply":"继续聊聊。"}')
    with caplog.at_level(logging.ERROR), pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("generate")
    assert "L2.generate.schema_validation" in caplog.text


def test_mi_generation_rejects_summary_that_conflicts_with_rule_state(monkeypatch):
    monkeypatch.setattr(
        client,
        "ask_llm",
        lambda *_, **__: _valid_generation_response(stage="planning"),
    )
    with pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("generate")


def test_http_status_is_reported_without_headers_or_body(monkeypatch, caplog):
    import httpx
    from openai import AuthenticationError

    def unauthorized(*args, **kwargs):
        response = httpx.Response(401, request=httpx.Request("POST", "https://example.invalid/v1"))
        raise AuthenticationError("secret-key", response=response, body={"message": "patient-private-text"})

    monkeypatch.setattr(client, "ask_llm_stream", unauthorized)
    with caplog.at_level(logging.ERROR), pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("analyze")
    assert '"http_status": 401' in caplog.text
    assert "AuthenticationError" in caplog.text
    assert "secret-key" not in caplog.text and "patient-private-text" not in caplog.text


def test_schema_error_in_real_l2_adapter_ends_terminal(monkeypatch, tmp_path, caplog, capsys):
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    pipeline = _orchestrator(tmp_path)
    pipeline.l2.model = client.APIL2Model()
    monkeypatch.setattr(client, "ask_llm_stream", lambda *_, **__: json.dumps({
        "input_type": "free_narrative", "symptoms": [{"value": "fatigue"}],
    }))
    inputs = iter(["最近一个多月，我感觉整个人都没劲了。工作不想做，回家也不想说话，女朋友觉得我变得很冷淡。"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with caplog.at_level(logging.WARNING):
        entry.run_session(pipeline)
    assert "L2.analyze.schema_validation" in caplog.text
    assert "evidence_quote" in caplog.text
    assert "STOPPED" in capsys.readouterr().out

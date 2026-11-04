"""平台兼容性回归；使用 SDK 替身，不发送真实请求。"""

from types import SimpleNamespace

import pytest

from shared import llm_client
from layers.layer3.instructions.turn_planner import APITreatmentPlanner
from layers.layer4.config import Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
        "L3_REASONER_MODEL", "L3_VERIFIER_MODEL", "L3_INSTRUCTION_MODEL",
        "L3_REASONING_EFFORT", "L3_STEP2_MAX_WORKERS",
        "L4_LLM_BASE_URL", "L4_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("aliyun", [False, True])
def test_shared_sdk_receives_matching_key_url_model(monkeypatch, aliyun):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def with_options(self, **kwargs):
            captured.update(kwargs)
            return self

        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])

    monkeypatch.setattr(llm_client, "OpenAI", FakeClient)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "official-test-key")
    if aliyun:
        monkeypatch.setenv("LLM_API_KEY", "aliyun-test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://aliyun.example/compatible-mode/v1")
        monkeypatch.setenv("LLM_MODEL", "deepseek-v3.2")
    assert llm_client.ask_llm("虚构测试") == '{"ok":true}'
    assert captured["api_key"] == ("aliyun-test-key" if aliyun else "official-test-key")
    assert captured["base_url"] == ("https://aliyun.example/compatible-mode/v1" if aliyun else "https://api.deepseek.com")
    assert captured["model"] == ("deepseek-v3.2" if aliyun else "deepseek-chat")
    assert captured["timeout"] == 10.0 and captured["max_retries"] == 1
    llm_client.ask_llm("虚构测试", model="explicit-model")
    assert captured["model"] == "explicit-model"


def test_all_default_models_follow_shared_model_and_preserve_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "shared-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://aliyun.example/v1")
    assert APITreatmentPlanner().model == "shared-model"
    assert Settings().llm_model == "shared-model"
    assert Settings().llm_base_url == "https://aliyun.example/v1"
    monkeypatch.setenv("L3_INSTRUCTION_MODEL", "planner")
    monkeypatch.setenv("L4_LLM_MODEL", "generator")
    monkeypatch.setenv("L4_LLM_BASE_URL", "https://separate.example/v1")
    assert APITreatmentPlanner().model == "planner"
    assert Settings().llm_model == "generator"
    assert Settings().llm_base_url == "https://separate.example/v1"


def test_missing_key_stops_before_sdk(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", lambda **_: pytest.fail("不能请求 SDK"))
    with pytest.raises(RuntimeError, match="缺少"):
        llm_client.ask_llm("虚构测试")


def test_non_stream_optional_metadata_preserves_text_return(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content="response", reasoning_content="private reasoning"),
        )])

    monkeypatch.setattr(llm_client, "_bounded_client", lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ))
    metadata = {}
    assert llm_client.ask_llm("private prompt", response_metadata=metadata) == "response"
    assert metadata == {"finish_reason": "length"}
    assert "response_metadata" not in captured


def test_streaming_client_joins_content_and_ignores_reasoning_and_usage_chunks(monkeypatch):
    captured = {}

    def chunk(content=None, reasoning=None):
        delta = SimpleNamespace(content=content, reasoning_content=reasoning)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def with_options(self, **kwargs):
            captured.update(kwargs)
            return self

        def create(self, **kwargs):
            captured.update(kwargs)
            return iter([
                chunk(reasoning="private reasoning"),
                chunk('{"input_'),
                SimpleNamespace(choices=[]),
                chunk('type":"free_narrative"}'),
            ])

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "OpenAI", FakeClient)
    result = llm_client.ask_llm_stream("虚构测试", timeout=60)
    assert result == '{"input_type":"free_narrative"}'
    assert captured["stream"] is True
    assert captured["timeout"] == 60
    assert captured["max_retries"] == 1


def test_streaming_client_forwards_optional_extra_body(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def with_options(self, **kwargs):
            return self

        def create(self, **kwargs):
            captured.update(kwargs)
            return iter([])

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "OpenAI", FakeClient)
    options = {"enable_thinking": True, "reasoning_effort": "medium"}
    assert llm_client.ask_llm_stream("虚构测试", extra_body=options) == ""
    assert captured["extra_body"] == options


def test_streaming_client_forwards_optional_response_format(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def with_options(self, **kwargs):
            return self

        def create(self, **kwargs):
            captured.update(kwargs)
            return iter([])

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "OpenAI", FakeClient)
    response_format = {"type": "json_object"}

    assert llm_client.ask_llm_stream(
        "虚构测试", response_format=response_format
    ) == ""
    assert captured["response_format"] == response_format

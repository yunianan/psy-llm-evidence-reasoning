"""终端入口回归：不请求真实模型服务，不修改真实配置。"""

import os
from collections import deque

import pytest

import main as entry
from orchestration.schemas import OrchestrationOutcome, PatientConfirmation


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "PROJECT_ROOT", tmp_path)
    for name in (
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "L4_LLM_API_KEY",
        "L4_LLM_BASE_URL", "L4_LLM_PROVIDER", "L4_REVIEW_MODE",
        "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "L4_LLM_MODEL",
        "L3_REASONER_MODEL", "L3_VERIFIER_MODEL", "L3_INSTRUCTION_MODEL",
        "L3_INSTRUCTION_TIMEOUT_SECONDS", "L5_LLM_TIMEOUT_SECONDS",
    ):
        # 入口会直接写入 os.environ；即使变量原本不存在，也要登记回滚。
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def test_visible_key_reused_for_same_endpoint_without_saving(monkeypatch, tmp_path, capsys):
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or " secret-key ")
    entry.configure_api()
    assert len(prompts) == 1
    assert "明文显示" in prompts[0]
    assert os.environ["DEEPSEEK_API_KEY"] == "secret-key"
    assert os.environ["L4_LLM_API_KEY"] == "secret-key"
    assert not (tmp_path / ".env").exists()
    assert "secret-key" not in capsys.readouterr().out


def test_dotenv_loaded_from_project_root_environment_wins(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=file-key\nL4_LLM_API_KEY=l4-file-key\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("已有密钥，不应询问"))
    entry.configure_api()
    assert os.environ["DEEPSEEK_API_KEY"] == "env-key"
    assert os.environ["L4_LLM_API_KEY"] == "l4-file-key"


def test_different_l4_endpoint_requires_its_own_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "shared-key")
    monkeypatch.setenv("L4_LLM_BASE_URL", "https://other.invalid/v1")
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "separate-key")
    entry.configure_api()
    assert len(prompts) == 1 and "L4_LLM_API_KEY" in prompts[0]
    assert "明文显示" in prompts[0]
    assert os.environ["L4_LLM_API_KEY"] == "separate-key"


@pytest.mark.parametrize("name", ["L4_LLM_PROVIDER", "L4_REVIEW_MODE"])
def test_mock_mode_is_not_silently_used(monkeypatch, name):
    monkeypatch.setenv(name, "mock")
    with pytest.raises(entry.StartupError, match=name):
        entry.configure_api()


@pytest.mark.parametrize("value", ["invalid", "0", "301"])
def test_invalid_l5_timeout_stops_startup(monkeypatch, value):
    monkeypatch.setenv("L5_LLM_TIMEOUT_SECONDS", value)
    with pytest.raises(entry.StartupError, match="L5_LLM_TIMEOUT_SECONDS"):
        entry.configure_api()


@pytest.mark.parametrize("value", ["invalid", "0", "301"])
def test_invalid_l3_instruction_timeout_stops_startup(monkeypatch, value):
    monkeypatch.setenv("L3_INSTRUCTION_TIMEOUT_SECONDS", value)
    with pytest.raises(entry.StartupError, match="L3_INSTRUCTION_TIMEOUT_SECONDS"):
        entry.configure_api()


def test_empty_key_prevents_session_creation(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: " ")
    monkeypatch.setattr(entry, "create_pipeline", lambda: pytest.fail("不能创建会话"))
    assert entry.main() == 1
    assert "未提供 DEEPSEEK_API_KEY" in capsys.readouterr().out


def test_empty_separate_l4_key_is_rejected(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "shared-key")
    monkeypatch.setenv("L4_LLM_BASE_URL", "https://other.invalid/v1")
    monkeypatch.setattr("builtins.input", lambda prompt: " ")
    with pytest.raises(entry.StartupError, match="未提供 L4_LLM_API_KEY"):
        entry.configure_api()


def outcome(route="L2_INTERVIEW", **kwargs):
    return OrchestrationOutcome(session_id="terminal-test", route=route, reply="测试回复", **kwargs)


class FakePipeline:
    def __init__(self, results=()):
        self.results = deque(results)
        self.messages = []
        self.resumes = []

    def create_session(self):
        return outcome()

    def handle_message(self, sid, message):
        self.messages.append((sid, message))
        return self.results.popleft()

    def resume_session(self, sid):
        self.resumes.append(sid)
        return outcome("AWAITING_PATIENT_CONFIRMATION")

    def get_developer_profile_path(self, sid):
        return entry.PROJECT_ROOT / "artifacts" / "developer_profiles" / f"{sid}.html"


def feed(monkeypatch, messages):
    values = iter(messages)
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))


def test_aliyun_interactive_configuration_reaches_all_adapters(monkeypatch, tmp_path):
    from layers.layer3.llm.client import APICLINICRReasoner, APIRewardModelVerifier
    from shared.llm_client import get_api_key, get_base_url, get_model

    monkeypatch.delenv("DEEPSEEK_BASE_URL")
    url = "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    feed(monkeypatch, ["2", url + "/", "deepseek-v3.2", "aliyun-test-key"])
    entry.configure_api()
    assert get_base_url() == url
    assert get_model() == "deepseek-v3.2"
    assert get_api_key() == "aliyun-test-key"
    assert APICLINICRReasoner().model == "deepseek-v3.2"
    assert APIRewardModelVerifier().model == "deepseek-v3.2"
    pipeline = entry.create_pipeline()
    assert pipeline.l4.config.llm_base_url == url
    assert pipeline.l4.config.llm_model == "deepseek-v3.2"
    assert pipeline.l4.config.llm_api_key == "aliyun-test-key"
    assert not (tmp_path / ".env").exists()


def test_aliyun_dotenv_needs_no_interactive_input(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "LLM_API_KEY=aliyun-test-key\nLLM_BASE_URL=https://aliyun.example/v1\nLLM_MODEL=custom-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("不应询问"))
    entry.configure_api()
    assert entry.create_pipeline().l4.config.llm_model == "custom-model"
    assert os.environ["L4_LLM_API_KEY"] == "aliyun-test-key"


@pytest.mark.parametrize("url,model", [
    ("", "deepseek-v3.2"), ("https://aliyun.example/v1", ""),
    ("http://aliyun.example/v1", "deepseek-v3.2"),
    ("https://{WorkspaceId}.aliyun.example/v1", "deepseek-v3.2"),
    ("https://aliyun.example/v1/chat/completions", "deepseek-v3.2"),
])
def test_invalid_aliyun_configuration_stops_before_key_input(monkeypatch, url, model):
    monkeypatch.delenv("DEEPSEEK_BASE_URL")
    feed(monkeypatch, ["2", url, model])
    with pytest.raises(entry.StartupError):
        entry.configure_api()


def test_interactive_official_default_still_works(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL")
    feed(monkeypatch, ["", "official-test-key"])
    entry.configure_api()
    assert os.environ["DEEPSEEK_API_KEY"] == "official-test-key"


def test_invalid_platform_fails_before_session(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL")
    feed(monkeypatch, ["3"])
    with pytest.raises(entry.StartupError, match="平台选项"):
        entry.configure_api()


def test_text_and_confirmation_display_without_dumping_internal_state(monkeypatch, capsys):
    card = PatientConfirmation(
        title="请确认画像", profile_summary=["持续三周"], scale_summary=["量表摘要"],
        initial_impression="尚未诊断", disclaimer="辅助评估", next_step="确认后分析",
    )
    pipeline = FakePipeline([outcome("AWAITING_PATIENT_CONFIRMATION", confirmation=card), outcome("L4")])
    feed(monkeypatch, ["我最近睡不好", "信息准确并继续", "/exit"])
    entry.run_session(pipeline)
    assert pipeline.messages == [("terminal-test", "我最近睡不好"), ("terminal-test", "信息准确并继续")]
    output = capsys.readouterr().out
    assert "请确认画像" in output and "量表摘要" in output
    assert "需要修改 / 暂停" in output
    assert "developer_profiles" in output


def test_local_commands_do_not_enter_clinical_pipeline(monkeypatch, capsys):
    pipeline = FakePipeline()
    feed(monkeypatch, [" ", "/help", "/profile", "/unknown", "/resume", "/exit"])
    entry.run_session(pipeline)
    assert pipeline.messages == [] and pipeline.resumes == []
    assert "未知终端命令" in capsys.readouterr().out


def test_patient_pause_can_resume(monkeypatch):
    pipeline = FakePipeline([outcome("PAUSED_BY_PATIENT")])
    feed(monkeypatch, ["我想暂停", "/resume", "/exit"])
    entry.run_session(pipeline)
    assert pipeline.resumes == ["terminal-test"]


def test_manual_review_cannot_resume_but_new_text_still_reaches_orchestrator(monkeypatch):
    pipeline = FakePipeline([outcome("MANUAL_REVIEW"), outcome("STOPPED", terminal=True)])
    feed(monkeypatch, ["普通消息", "/resume", "新的安全风险消息"])
    entry.run_session(pipeline)
    assert pipeline.resumes == []
    assert len(pipeline.messages) == 2


def test_terminal_result_stops_reading_input(monkeypatch):
    pipeline = FakePipeline([outcome("STOPPED", terminal=True)])
    feed(monkeypatch, ["测试风险消息"])
    entry.run_session(pipeline)
    assert len(pipeline.messages) == 1


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_cancel_during_startup_is_clean(monkeypatch, capsys, error):
    def cancel(prompt):
        raise error()
    monkeypatch.setattr("builtins.input", cancel)
    assert entry.main() == 0
    assert "已退出" in capsys.readouterr().out


def test_main_reports_safe_clinicr_stage_without_arbitrary_error_text(monkeypatch, capsys):
    from layers.layer3.llm import CLINICRModelUnavailable

    unsafe_cause = ValueError("患者原话和sk-secret不得显示")
    clinicr_error = CLINICRModelUnavailable(
        "CLINICR Step 1 API不可用或输出不合法：JSONDecodeError"
    )
    clinicr_error.__cause__ = unsafe_cause
    monkeypatch.setattr(entry, "configure_api", lambda: None)
    monkeypatch.setattr(entry, "create_pipeline", lambda: object())
    monkeypatch.setattr(entry, "run_session", lambda _: (_ for _ in ()).throw(clinicr_error))

    assert entry.main() == 1
    output = capsys.readouterr().out
    assert '"stage": "L3.CLINICR.Step1"' in output
    assert '"exception": "ValueError"' in output
    assert "患者原话" not in output
    assert "sk-secret" not in output


def test_main_labels_clinicr_api_error_separately(monkeypatch, capsys):
    from layers.layer3.llm import CLINICRAPIError

    api_error = CLINICRAPIError("CLINICR Step 1 API错误：APIError")
    api_error.__cause__ = RuntimeError("provider-secret")
    monkeypatch.setattr(entry, "configure_api", lambda: None)
    monkeypatch.setattr(entry, "create_pipeline", lambda: object())
    monkeypatch.setattr(entry, "run_session", lambda _: (_ for _ in ()).throw(api_error))

    assert entry.main() == 1
    output = capsys.readouterr().out
    assert '[开发调试] API_ERROR {"stage": "L3.CLINICR.Step1"' in output
    assert "运行中止：CLINICRAPIError" in output
    assert "provider-secret" not in output


def test_unexpected_error_does_not_echo_request_or_credentials(monkeypatch, capsys):
    monkeypatch.setattr(entry, "configure_api", lambda: None)
    def unavailable():
        raise RuntimeError("secret-key patient-private-message")
    monkeypatch.setattr(entry, "create_pipeline", unavailable)
    assert entry.main() == 1
    output = capsys.readouterr().out
    assert "RuntimeError" in output
    assert "secret-key" not in output and "patient-private-message" not in output


def test_default_factory_uses_real_adapters_and_absolute_profile_path(monkeypatch, tmp_path):
    from layers.layer2.llm import APIL2Model
    from layers.layer4.review.l5_adapter import L5Reviewer
    monkeypatch.setenv("L4_LLM_API_KEY", "fake-unused-key")
    monkeypatch.chdir(tmp_path.parent)
    pipeline = entry.create_pipeline()  # 只构造对象，不创建会话、不请求 API。
    assert isinstance(pipeline.l2.model, APIL2Model)
    assert isinstance(pipeline.l4.reviewer, L5Reviewer)
    assert pipeline.l4.config.llm_provider == "openai_compat"
    assert pipeline.profile_renderer.output_dir == tmp_path / "artifacts" / "developer_profiles"


def test_entry_runs_real_orchestrator_with_controlled_models(monkeypatch, tmp_path, capsys):
    from tests.test_top_level_orchestrator import _orchestrator, _safe_passport
    pipeline = _orchestrator(tmp_path)
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    monkeypatch.setattr(entry, "configure_api", lambda: None)
    monkeypatch.setattr(entry, "create_pipeline", lambda: pipeline)
    feed(monkeypatch, ["最近三周我每天情绪低落，没有兴趣，也没有精力。", "/exit"])
    assert entry.main() == 0
    assert "本次终端会话已结束" in capsys.readouterr().out
    assert len(list((tmp_path / "developer-only").glob("*.html"))) == 1

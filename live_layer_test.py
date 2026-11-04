"""逐层调用真实 LLM 的独立手工测试入口。


在 VS Code 中右击本文件，选择“在终端中运行 Python 文件”；也可以运行：
    python live_layer_test.py l2

本文件只为目标层提供固定的虚构上游输入，不串行执行前置层。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import urlsplit
import uuid
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parent


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2)


def _configure_real_api() -> None:
    """读取本机配置；没有配置时只写入当前测试进程的环境变量。"""
    from dotenv import load_dotenv
    from shared.llm_client import get_api_key, get_base_url, get_model

    load_dotenv(PROJECT_ROOT / ".env", override=False, encoding="utf-8-sig")
    if not get_api_key():
        platform = input("选择 API 平台：1=DeepSeek 官方，2=阿里云百炼 [1]：").strip() or "1"
        if platform == "2":
            base_url = input(
                "请输入阿里 OpenAI 兼容 base_url（不要包含 chat/completions）："
            ).strip()
            model = input("请输入阿里控制台中的模型 ID：").strip()
            parsed = urlsplit(base_url)
            if (
                not base_url
                or not model
                or parsed.scheme != "https"
                or not parsed.hostname
                or parsed.path.rstrip("/").endswith("/chat/completions")
            ):
                raise ValueError("阿里平台的 base_url 或模型 ID 不合法")
            os.environ["LLM_BASE_URL"] = base_url.rstrip("/")
            os.environ["LLM_MODEL"] = model
        elif platform == "1":
            os.environ.setdefault("LLM_BASE_URL", "https://api.deepseek.com")
            os.environ.setdefault("LLM_MODEL", "deepseek-chat")
        else:
            raise ValueError("平台只能选择 1 或 2")

        key = input("请输入 API Key（明文显示，仅保存在本次进程中）：").strip()
        if not key:
            raise ValueError("没有提供 API Key")
        os.environ["LLM_API_KEY"] = key

    endpoint = urlsplit(get_base_url()).hostname or "无法识别"
    print(f"\n[连接] endpoint={endpoint}；model={get_model()}；使用真实 API")


def _run_timed(label: str, operation: Callable[[], Any]) -> Any:
    print(f"\n[{label}] 正在调用真实 LLM……", flush=True)
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"[{label}] 失败，耗时 {elapsed:.2f}s")
        _show_safe_exception_chain(exc)
        try:
            setattr(exc, "_live_error_shown", True)
        except Exception:
            pass
        raise
    elapsed = time.perf_counter() - started
    print(f"[{label}] 完成，耗时 {elapsed:.2f}s")
    return result


def _show_safe_exception_chain(error: BaseException) -> None:
    """显示错误类型和安全元数据，不打印请求体、患者原话或密钥。"""
    current: BaseException | None = error
    seen: set[int] = set()
    level = 0
    safe_messages = {
        "L2ModelUnavailable",
        "L2ModelOutputInvalid",
        "L3ToL4ValidationError",
    }
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        info: dict[str, Any] = {
            "level": level,
            "type": type(current).__name__,
        }
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            info["http_status"] = status
        if type(current).__name__ in safe_messages:
            info["message"] = str(current)
        if type(current).__name__ == "ValidationError" and hasattr(current, "errors"):
            info["validation_errors"] = [
                {"path": list(item["loc"]), "type": item["type"]}
                for item in current.errors(
                    include_input=False,
                    include_context=False,
                    include_url=False,
                )[:20]
            ]
        print("[安全错误链] " + json.dumps(info, ensure_ascii=False))
        current = current.__cause__ or current.__context__
        level += 1


def test_l1() -> None:
    """持续运行 L1，直到允许进入 L2 或给出安全交接结果。"""
    from layers.layer1.detector import safety_detect, start_baseline_asq

    session_id = "live-l1-" + uuid.uuid4().hex
    text = input(
        "患者消息 [最近工作压力很大，晚上睡不好，但我没有伤害自己的想法。]"
        "（/baseline 测常规 ASQ，/exit 退出）："
    ).strip() or "最近工作压力很大，晚上睡不好，但我没有伤害自己的想法。"
    turn = 0
    while text != "/exit":
        turn += 1
        result = _run_timed(
            f"L1 第{turn}轮",
            lambda: (
                start_baseline_asq(session_id=session_id, message_id=uuid.uuid4().hex)
                if text == "/baseline"
                else safety_detect(
                    text, session_id=session_id, message_id=uuid.uuid4().hex
                )
            ),
        )
        if result is None:
            raise RuntimeError("当前会话无法启动常规 ASQ")
        if result.get("status") == "error":
            print("\n[L1 错误结果]\n" + _json(result))
            raise RuntimeError("L1 未完成，模型或 API 返回错误")
        execution = result.get("execution") or {}
        print("\n系统：" + str(execution.get("user_response") or ""))
        print(
            "[开发调试] 路由：{0}；等待：{1}".format(
                execution.get("route"), execution.get("waiting_for") or "无"
            )
        )
        if execution.get("route") == "l2" or execution.get("resume_l2_next_turn"):
            print("\n[L1 出口：允许下一条患者消息进入 L2]\n" + _json(result))
            return
        if execution.get("route") == "safety":
            print("\n[L1 出口：普通疗程暂停，已输出安全交接指引]\n" + _json(result))
            return
        if execution.get("route") not in {"l1", "security"}:
            raise RuntimeError("L1 返回了未知路由，未到达本层出口")
        text = input("你（/exit 可中止测试）：").strip()
        while not text:
            text = input("请输入回答（/exit 可中止测试）：").strip()
    print("L1 测试由用户中止，未到达本层出口。")


def test_l2() -> None:
    """使用真实 L2 模型持续评估，直到画像确认且通过 L2→L3 守卫。"""
    from layers.layer1.service import L1SafetyService
    from layers.layer2 import L2AssessmentService, L2ToL3Adapter
    from layers.layer2.llm import APIL2Model
    from orchestration.schemas import PipelineSession
    from patient_presentation.developer_profile import DeveloperProfileRenderer

    session_id = "live-l2-" + uuid.uuid4().hex
    # 独立测试只模拟上游已放行，不调用 L1 模型；真实患者请走 main.py。
    passport = L1SafetyService().handle_message(
        "虚构测试会话的固定上游输入",
        session_id=session_id,
        message_id=uuid.uuid4().hex,
        classifier=lambda *_args: {
            "analysis": {
                "suicide_signal_detected": False,
                "jailbreak_detected": False,
                "safe_to_enter_l2": True,
                "confidence": 1.0,
                "evidence_quotes": [],
                "jailbreak_evidence": [],
                "asq_transition_support": "",
                "summary": "独立 L2 测试的固定上游安全结果",
            }
        },
    )
    service = L2AssessmentService(model=APIL2Model(), require_fresh_passport=False)
    service.create_session(passport, session_id=session_id)
    adapter = L2ToL3Adapter()
    profile_renderer = DeveloperProfileRenderer(
        PROJECT_ROOT / "artifacts" / "developer_profiles"
    )
    profile_session = PipelineSession(
        session_id=session_id, l2_session_id=session_id, l1_passport=passport,
        clinical_intake=service.build_clinical_intake(session_id),
    )
    profile_path = profile_renderer.render(profile_session)
    print("本项仅使用固定 L1 测试凭证；L2 的语义理解与 MI 回复调用真实 API。")
    print("持续对话直到患者确认画像并通过 L2→L3 守卫；/exit 可中止测试。")
    print(f"[开发者画像，仅限本机调试] {profile_path}")
    print("/profile 用浏览器打开画像；每轮更新后刷新页面。HTML 仅供查看，不能恢复会话。")
    text = input(
        "患者消息 [最近一个多月我总觉得没劲，工作也受到影响，想先改善睡眠。]："
    ).strip() or "最近一个多月我总觉得没劲，工作也受到影响，想先改善睡眠。"
    expected_input = "auto"
    turn = 0
    while text != "/exit":
        if text == "/profile":
            try:
                opened = webbrowser.open(profile_path.as_uri())
            except (OSError, webbrowser.Error):
                opened = False
            if not opened:
                print(f"请用浏览器手动打开：{profile_path}")
            text = input("你（/profile 查看画像；/exit 中止测试）：").strip()
            while not text:
                text = input("请输入回答（/profile 查看画像；/exit 中止测试）：").strip()
            continue
        turn += 1
        result = _run_timed(
            f"L2 第{turn}轮（语义提取＋规则处理＋MI 回复）",
            lambda: service.handle_message(
                session_id, text, expected_input=expected_input
            ),
        )
        print("\n系统：" + result.reply)
        if result.error_code in {"L2_MODEL_UNAVAILABLE", "L2_MODEL_OUTPUT_INVALID"}:
            raise RuntimeError(f"L2 未完成：{result.error_code}")
        if not result.accepted:
            print(f"[开发调试] 当前消息未被接受：{result.error_code or 'unknown'}")
            expected_input = result.awaiting or expected_input
        else:
            intake = service.build_clinical_intake(session_id)
            profile_session.clinical_intake = intake
            profile_session.expected_input = result.awaiting or "auto"
            profile_session.route = (
                "PAUSED_BY_PATIENT" if result.confirmation_decision == "pause"
                else "AWAITING_PATIENT_CONFIRMATION"
                if intake.patient_confirmation == "pending"
                and intake.information_sufficiency == "sufficient"
                else "L2_INTERVIEW"
            )
            profile_session.touch()
            profile_path = profile_renderer.render(profile_session)
            print(f"[开发者画像已更新] {profile_path}")
            if result.confirmation_decision == "pause":
                print("L2 已按患者选择暂停，未交接至 L3。")
                return
            if result.confirmation_decision == "revise":
                print("[开发调试] 请说明画像中需要修改的具体信息。")
                expected_input = "free_text"
            elif intake.information_sufficiency == "sufficient":
                if intake.patient_confirmation == "confirmed":
                    receipt = adapter.guard.verify(intake)
                    if receipt["status"] == "pass":
                        print("\n[L2 出口：画像已确认，L2→L3 守卫通过]")
                        print("\n[守卫回执]\n" + _json(receipt))
                        print("\n[L2 患者画像]\n" + _json(adapter.to_profile(intake)))
                        return
                    print("[开发调试] L2→L3 守卫未通过：" + "；".join(receipt["violations"]))
                    service.confirm_profile(session_id, "revise")
                    expected_input = "free_text"
                else:
                    print("\n[待患者确认的画像]\n" + _json(intake))
                    expected_input = "profile_confirmation"
            else:
                print("[开发调试] 待补充：" + "；".join(intake.missing_dimensions))
                expected_input = result.awaiting or "auto"
            if result.l2_turn and result.l2_turn.pending_scale_item:
                pending = result.l2_turn.pending_scale_item
                print(f"[开发调试] 当前量表题：{pending.scale_name} 第{pending.item_number}题")
                expected_input = "scale_answer"
        text = input("你（/exit 可中止测试）：").strip()
        while not text:
            text = input("请输入回答（/exit 可中止测试）：").strip()
    print("L2 测试由用户中止，未到达 L2→L3 出口。")


def _fictional_report():
    from layers.layer4.schemas.l3_report import DiagnosticReport

    path = Path(__file__).resolve().parent / "examples" / "sample_diagnostic_report.json"
    return DiagnosticReport.model_validate_json(path.read_text(encoding="utf-8"))


def _first_turn(instructions, session_id: str, user_message: str):
    from shared.schemas.treatment_instructions import SourceQuote, TurnInstruction

    return TurnInstruction(
        goal="理解患者本轮最主要的困扰并建立合作关系",
        technique_code="reflection",
        technique=instructions.L2.allowed_techniques["reflection"],
        forbidden=list(instructions.L4.absolute_bans),
        link_previous="",
        plugin_guidance=None,
        plugin_steps={},
        force_substage="L4-L1",
        session_id=session_id,
        turn_number=1,
        source_message=user_message,
        evidence=[SourceQuote(source_id="turn:1:user", quote=user_message)],
        observations=[],
        previous_goal_met=False,
        patient_ready=False,
        rationale="患者刚表达当前困扰，首轮应先建立治疗联盟并确认体验。",
    )


def test_l4() -> None:
    """只调用真实 L4 回复模型；使用固定审核器隔离 L5。"""
    from layers.layer4.config import Settings
    from layers.layer4.review.mock_reviewer import MockReviewer
    from layers.layer4.schemas.instructions import SafetyPassport
    from layers.layer4.service import L4Service, SessionRequest, TurnRequest
    from shared.llm_client import get_api_key, get_base_url, get_model
    from shared.schemas.treatment_instructions import TreatmentInstructions

    message = input(
        "患者治疗消息 [这周我还是提不起精神，看到工作消息就想逃避。]："
    ).strip() or "这周我还是提不起精神，看到工作消息就想逃避。"
    report = _fictional_report()
    session_id = "live-l4-" + uuid.uuid4().hex
    settings = Settings(
        _env_file=None,
        llm_provider="openai_compat",
        llm_api_key=get_api_key(),
        llm_base_url=get_base_url(),
        llm_model=get_model(),
        review_mode="mock",
        max_rewrite_attempts=1,
    )
    service = L4Service(config=settings, reviewer=MockReviewer())
    service.create_session(
        SessionRequest(
            session_id=session_id,
            report=report,
            safety=SafetyPassport(risk_level=0, sensitive_topics=[], flags=[]),
        )
    )
    instructions = TreatmentInstructions.model_validate(report.instructions)
    turn = _first_turn(instructions, session_id, message)
    outcome = _run_timed(
        "L4 治疗回复",
        lambda: service.handle_turn(
            TurnRequest(session_id=session_id, user_message=message, turn=turn)
        ),
    )
    print("\n[L4 出口：本轮治疗回复已生成]\n" + outcome.reply)
    print("\n[L4 结构化结果]\n" + _json(outcome))
    print("\n说明：本项只隔离测试 L4 的真实生成模型，L5 审核未被调用。")


def test_l5() -> None:
    """只调用真实 L5 审核模型；L4 回复与完整上下文由固定样例提供。"""
    from layers.layer4.generation.summary_builder import build_l5_context
    from layers.layer4.schemas.instructions import SafetyPassport
    from layers.layer5.reviewer import review_response
    from shared.schemas.treatment_instructions import TreatmentInstructions

    user_message = input(
        "本轮患者消息 [这周我还是提不起精神，看到工作消息就想逃避。]："
    ).strip() or "这周我还是提不起精神，看到工作消息就想逃避。"
    l4_reply = input(
        "待审核 L4 回复 [听起来工作消息会立刻带来很强的压力。我们可以先看看那一刻脑中出现了什么想法。]："
    ).strip() or "听起来工作消息会立刻带来很强的压力。我们可以先看看那一刻脑中出现了什么想法。"
    report = _fictional_report()
    session_id = "live-l5-" + uuid.uuid4().hex
    instructions = TreatmentInstructions.model_validate(report.instructions)
    instructions = instructions.model_copy(
        update={"L3": _first_turn(instructions, session_id, user_message)},
        deep=True,
    )
    context = build_l5_context(
        l4_reply,
        "L4-L1",
        SafetyPassport(risk_level=0, sensitive_topics=[], flags=[]),
        instructions,
        user_message,
        [],
    )
    result = _run_timed("L5 输出审核", lambda: review_response(context))
    print("\n[L5 出口：本轮审核判定已生成]\n" + _json(result))


TESTS: dict[str, tuple[str, Callable[[], None]]] = {
    "1": ("L1 安全层", test_l1),
    "2": ("L2 评估层", test_l2),
    "4": ("L4 治疗层", test_l4),
    "5": ("L5 审核层", test_l5),
}
ALIASES = {f"l{number}": number for number in TESTS}


def _choose_layer() -> str:
    print("\n请选择要独立测试的层级：")
    for number, (name, _) in TESTS.items():
        print(f"  {number}. {name}")
    print("  0. 退出")
    return input("选择：").strip().lower()


def _execute(choice: str) -> bool:
    choice = ALIASES.get(choice, choice)
    if choice in {"0", "exit", "quit", "q"}:
        return False
    if choice not in TESTS:
        print("无效选择，请输入 1、2、4 或 5。")
        return True
    name, test = TESTS[choice]
    print(f"\n=== {name}：独立真实 LLM 测试 ===")
    try:
        _configure_real_api()
        test()
    except Exception as exc:
        if not getattr(exc, "_live_error_shown", False):
            _show_safe_exception_chain(exc)
        print("本层测试未通过；上方安全错误链用于定位，不会继续调用其他层。")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="L1–L5 独立真实 LLM 手工测试")
    parser.add_argument(
        "layer",
        nargs="?",
        choices=["l1", "l2", "l4", "l5"],
        help="直接运行指定层；省略时显示交互菜单",
    )
    args = parser.parse_args()
    print("五层系统 · 独立真实 LLM 测试")
    print("每次只运行所选层；请求会发送到配置的模型服务并可能产生费用。")
    if args.layer:
        _execute(args.layer)
        return 0
    while _execute(_choose_layer()):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

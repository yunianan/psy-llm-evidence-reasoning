"""本地终端入口：在 VS Code 中右键“在终端中运行 Python 文件”。

只处理配置和终端输入输出；业务路由复用 FiveLayerOrchestrator。
"""

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
HELP = (
    "直接输入自然语言开始对话。\n"
    "/exit 退出；/resume 恢复患者主动暂停；/profile 查看开发者画像路径；/help 帮助。"
)


class StartupError(ValueError):
    """可以直接展示、不会包含密钥的配置错误。"""


def configure_api() -> None:
    """环境变量优先于 .env；终端明文输入的密钥仅存入当前进程。"""
    from dotenv import load_dotenv
    from shared.llm_client import get_api_key, get_base_url

    load_dotenv(PROJECT_ROOT / ".env", override=False, encoding="utf-8-sig")
    if os.environ.get("L4_LLM_PROVIDER", "openai_compat") != "openai_compat":
        raise StartupError("本入口使用真实 API，请将 L4_LLM_PROVIDER 设置为 openai_compat。")
    if os.environ.get("L4_REVIEW_MODE", "l5_local") != "l5_local":
        raise StartupError("本入口必须接入真实 L5，请将 L4_REVIEW_MODE 设置为 l5_local。")
    from layers.layer3.instructions.turn_planner import _instruction_timeout_seconds
    from layers.layer5.reviewer import _l5_timeout_seconds
    try:
        _instruction_timeout_seconds()
        _l5_timeout_seconds()
    except ValueError as exc:
        raise StartupError(str(exc)) from exc

    if not any(os.environ.get(name) for name in (
        "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
    )):
        platform = input("选择 API 平台：1=DeepSeek 官方，2=阿里云百炼 [1]：").strip() or "1"
        if platform == "2":
            base_url = input("请粘贴阿里控制台的 OpenAI 兼容 base_url（不是完整 chat/completions 地址）：").strip()
            model = input("请输入阿里控制台的模型 ID（model）：").strip()
            if not base_url or not model:
                raise StartupError("阿里平台必须提供 base_url 和 model。")
            from urllib.parse import urlsplit
            parsed = urlsplit(base_url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment
                    or "{" in base_url or "}" in base_url
                    or parsed.path.rstrip("/").endswith("/chat/completions")):
                raise StartupError("请使用阿里控制台的完整 HTTPS base_url，替换占位符，不要附带密钥或 /chat/completions。")
            os.environ["LLM_BASE_URL"] = base_url.rstrip("/")
            os.environ["LLM_MODEL"] = model
        elif platform != "1":
            raise StartupError("平台选项只能输入 1 或 2。")

    shared_key = get_api_key()
    key_name = "LLM_API_KEY" if os.environ.get("LLM_BASE_URL") or os.environ.get("LLM_MODEL") else "DEEPSEEK_API_KEY"
    if not shared_key:
        shared_key = input(f"请输入共享 API 密钥 {key_name}（明文显示，请勿分享截图）：").strip()
    if not shared_key:
        raise StartupError(f"未提供 {key_name}，未启动会话。")
    os.environ[key_name] = shared_key

    l4_key = (os.environ.get("L4_LLM_API_KEY") or "").strip()
    if not l4_key:
        shared_url = get_base_url()
        l4_url = os.environ.get("L4_LLM_BASE_URL", shared_url).strip().rstrip("/")
        if shared_url == l4_url:
            l4_key = shared_key
        else:
            l4_key = input("L4 使用不同 API 地址，请输入 L4_LLM_API_KEY（明文显示，请勿分享截图）：").strip()
    if not l4_key:
        raise StartupError("未提供 L4_LLM_API_KEY，未启动会话。")
    os.environ["L4_LLM_API_KEY"] = l4_key


def create_pipeline():
    # 延迟导入，确保先加载配置，并能提示解释器缺少的依赖。
    from orchestration import FiveLayerOrchestrator
    from patient_presentation import DeveloperProfileRenderer

    return FiveLayerOrchestrator(
        profile_renderer=DeveloperProfileRenderer(PROJECT_ROOT / "artifacts" / "developer_profiles")
    )


def show_outcome(outcome) -> None:
    print(f"\n系统：{outcome.reply}")
    print(f"[开发调试] 路由：{outcome.route}；等待：{outcome.waiting_for or '无'}")
    card = outcome.confirmation
    if card is not None:
        print("\n" + "\n".join([
            card.title,
            *card.profile_summary,
            *card.scale_summary,
            card.initial_impression,
            card.disclaimer,
            card.next_step,
            "请直接回复：信息准确并继续 / 需要修改 / 暂停。",
        ]))


def show_profile(pipeline, session_id: str) -> None:
    path = pipeline.get_developer_profile_path(session_id)
    if path is None:
        print("[开发者画像] 暂未生成，请检查输出目录的写入权限。")
    else:
        print(f"[开发者画像，仅限本机调试] {path}")
        print("可用浏览器打开；对话后刷新查看更新。")


def run_session(pipeline) -> None:
    outcome = pipeline.create_session()
    session_id = outcome.session_id
    print(f"会话：{session_id}")
    print("会话状态仅保存在内存中，退出程序后不能从画像 HTML 恢复。")
    print(HELP)
    show_outcome(outcome)
    show_profile(pipeline, session_id)

    while not outcome.terminal:
        text = input("\n你：").strip()
        if not text:
            continue
        if text == "/exit":
            break
        if text == "/help":
            print(HELP)
            continue
        if text == "/profile":
            show_profile(pipeline, session_id)
            continue
        if text == "/resume":
            if outcome.route != "PAUSED_BY_PATIENT":
                print("只能恢复患者主动暂停的会话；人工复核状态不能由此命令解除。")
                continue
            outcome = pipeline.resume_session(session_id)
        elif text.startswith("/"):
            print("未知终端命令，输入 /help 查看帮助。")
            continue
        else:
            print("处理中，请稍候……", flush=True)
            outcome = pipeline.handle_message(session_id, text)
        show_outcome(outcome)
    print("\n本次终端会话已结束。")


def main() -> int:
    if sys.version_info < (3, 11):
        print("请在 VS Code 中选择 Python 3.11 或更高版本的解释器。")
        return 1
    print("五层心理评估系统 · WHO ICD-10 · 本地终端")
    print("将调用真实 API，消息会发送到配置的模型服务并可能产生费用；建议先用虚构案例体验。")
    try:
        configure_api()
        run_session(create_pipeline())
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
        return 0
    except StartupError as exc:
        print(f"启动失败：{exc}")
        return 1
    except ModuleNotFoundError as exc:
        print(f"当前解释器缺少模块：{exc.name}。")
        print(f"解释器：{sys.executable}")
        print(f'请在项目根目录执行：& "{sys.executable}" -m pip install -e "."')
        return 1
    except Exception as exc:
        # API 异常可能携带请求原文或凭据，不直接输出任意异常内容。
        print(f"运行中止：{type(exc).__name__}。请检查 API 配置或保留此错误类型供排查。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

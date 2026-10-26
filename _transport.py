"""与系统共享客户端相同的 API 传输副本；不加载 Layer 3。"""
import os
from openai import OpenAI


def get_api_key():
    return (os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip()


def get_base_url():
    return (os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/")


def get_model():
    return (os.getenv("LLM_MODEL") or "deepseek-chat").strip()


def _bounded_client(*, timeout: float):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("生产 API 模式缺少 LLM_API_KEY 或 DEEPSEEK_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url=get_base_url(),
    )
    return client.with_options(timeout=timeout, max_retries=1)


def ask_llm_stream(
    prompt,
    model=None,
    temperature=0,
    *,
    timeout=10.0,
    extra_body: dict | None = None,
    response_format: dict | None = None,
):
    """内部流式接收并拼接文本；调用方只会取得完整结果。"""
    request = dict(
        model=model or get_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )
    if extra_body:
        request["extra_body"] = extra_body
    if response_format:
        request["response_format"] = response_format
    stream = _bounded_client(timeout=timeout).chat.completions.create(**request)
    parts: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        content = getattr(choices[0].delta, "content", None)
        if content:
            parts.append(content)
    return "".join(parts)

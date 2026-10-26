"""Phase 0：读取公开模型元数据，检查实际配置的模型 ID；不输出密钥。"""
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


ROOT = Path(__file__).resolve().parents[1] / "L1-L5系统"
sys.path.insert(0, str(ROOT))
from shared.llm_client import get_api_key, get_base_url, get_model


def main():
    load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")
    configured = {
        "chat_model": get_model(),
        "reasoner_model": os.getenv("L3_REASONER_MODEL") or get_model(),
        "verifier_model": os.getenv("L3_VERIFIER_MODEL") or get_model(),
    }
    result = {
        "configured": configured,
        "DASHSCOPE_API_KEY_configured": bool((os.getenv("DASHSCOPE_API_KEY") or "").strip()),
        "dashscope_matches_verified_embedding_key": (
            bool(os.getenv("DASHSCOPE_API_KEY"))
            and os.getenv("DASHSCOPE_API_KEY") == os.getenv("EMBEDDING_API_KEY")
        ),
    }
    try:
        with OpenAI(api_key=get_api_key(), base_url=get_base_url(), timeout=15, max_retries=0) as client:
            response = client.models.list()
        result["available_models"] = [
            {key: value for key, value in item.model_dump().items()
             if key in {"id", "name", "version", "created", "owned_by"} and value is not None}
            for item in response.data
        ]
        available = {item.id for item in response.data}
        result["unlisted_configured_models"] = sorted(set(configured.values()) - available)
        result["ok"] = not result["unlisted_configured_models"] and result["DASHSCOPE_API_KEY_configured"]
        if result["ok"]:
            with OpenAI(api_key=get_api_key(), base_url=get_base_url(), timeout=15, max_retries=0) as client:
                completion = client.chat.completions.create(
                    model=configured["chat_model"],
                    messages=[{"role": "user", "content": "Reply exactly OK."}],
                    max_tokens=8, temperature=0,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            result["smoke_test"] = {
                "ok": (completion.choices[0].message.content or "").strip() == "OK",
                "response_model": completion.model,
                "system_fingerprint": getattr(completion, "system_fingerprint", None),
                "finish_reason": completion.choices[0].finish_reason,
                "request": "Fixed text only; max_tokens=8; thinking disabled.",
            }
            result["ok"] = result["smoke_test"]["ok"]
    except OpenAIError as error:
        result.update(ok=False, error=type(error).__name__, status_code=getattr(error, "status_code", None))
    Path(__file__).with_name("model-check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

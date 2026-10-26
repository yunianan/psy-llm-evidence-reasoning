"""使用固定测试文本检查 Embedding API，只显示模型和向量维度。"""
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAIError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.embedding_client import EmbeddingClient


def main() -> int:
    load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")
    try:
        with EmbeddingClient() as client:
            vector = client.embed_query("这是一个向量接口连通性测试。")
            print(json.dumps({
                "ok": True, "model": client.model, "dimensions": len(vector),
            }, ensure_ascii=False))
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OpenAIError as exc:
        print(json.dumps({
            "ok": False, "error": type(exc).__name__,
            "status_code": getattr(exc, "status_code", None),
        }), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""百炼文本向量客户端；连接配置独立于聊天模型。"""
from __future__ import annotations

import math
import os
from urllib.parse import urlsplit

from openai import OpenAI


DEFAULT_MODEL = "text-embedding-v4"
DEFAULT_DIMENSIONS = 1024
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BATCH_SIZE = 10  # text-embedding-v4 的 OpenAI 兼容接口每批最多 10 条文本。


class EmbeddingClient:
    def __init__(self) -> None:
        self.model = (os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL).strip()
        raw_dimensions = os.getenv("EMBEDDING_DIMENSIONS") or str(DEFAULT_DIMENSIONS)
        try:
            self.dimensions = int(raw_dimensions)
        except ValueError as exc:
            raise ValueError("EMBEDDING_DIMENSIONS 必须是整数") from exc
        if self.dimensions not in {64, 128, 256, 512, 768, 1024, 1536, 2048}:
            raise ValueError("EMBEDDING_DIMENSIONS 不是 text-embedding-v4 支持的维度")
        base_url = (os.getenv("EMBEDDING_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or "{" in base_url or "}" in base_url
                or parsed.path.endswith("/embeddings")):
            raise ValueError("EMBEDDING_BASE_URL 应为完整 HTTPS base_url，不含占位符或 /embeddings")
        api_key = (os.getenv("EMBEDDING_API_KEY") or "").strip()
        api_key = api_key or (os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("缺少百炼 EMBEDDING_API_KEY 或 DASHSCOPE_API_KEY，请填写本地 .env")
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=30.0, max_retries=1,
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("向量输入必须是非空文本")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self.model, input=batch,
                dimensions=self.dimensions, encoding_format="float",
            )
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise ValueError("Embedding 返回的数量或索引与输入不一致")
            for item in items:
                vector = item.embedding
                if len(vector) != self.dimensions or not all(math.isfinite(x) for x in vector):
                    raise ValueError("Embedding 返回的向量维度或数值无效")
                vectors.append(vector)
        return vectors

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EmbeddingClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

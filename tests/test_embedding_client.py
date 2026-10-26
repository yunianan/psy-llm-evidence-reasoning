"""通过真实 OpenAI SDK 和受控 HTTP 响应验证向量接口，不发送外部请求。"""
import json

import httpx
from openai import AuthenticationError, OpenAI
import pytest

from shared import embedding_client


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "EMBEDDING_API_KEY", "DASHSCOPE_API_KEY", "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(name, raising=False)


def mock_api(monkeypatch, responder=None):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((request, body))
        if responder:
            return responder(body)
        return httpx.Response(200, json={
            "object": "list", "model": body["model"],
            "data": [
                {"object": "embedding", "index": i,
                 "embedding": [float(text)] * body["dimensions"]}
                for i, text in reversed(list(enumerate(body["input"])))
            ],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    monkeypatch.setattr(embedding_client, "OpenAI", lambda **kwargs: OpenAI(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ))
    return requests


def test_defaults_batch_limit_and_input_order(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "embedding-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://chat.example/v1")
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    requests = mock_api(monkeypatch)
    with embedding_client.EmbeddingClient() as client:
        vectors = client.embed_documents([str(i) for i in range(22)])
        assert [v[0] for v in vectors] == list(range(22))
        assert all(len(v) == 1024 for v in vectors)
        assert client.embed_query("42") == [42.0] * 1024
    assert [len(body["input"]) for _, body in requests] == [10, 10, 2, 1]
    for request, body in requests:
        assert str(request.url) == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        assert request.headers["authorization"] == "Bearer embedding-test-key"
        assert body["model"] == "text-embedding-v4"
        assert body["dimensions"] == 1024
        assert body["encoding_format"] == "float"


def test_explicit_connection_and_key_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "specific-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "other-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://workspace.example/compatible-mode/v1/")
    requests = mock_api(monkeypatch)
    with embedding_client.EmbeddingClient() as client:
        client.embed_query("1")
    assert str(requests[0][0].url) == "https://workspace.example/compatible-mode/v1/embeddings"
    assert requests[0][0].headers["authorization"] == "Bearer specific-key"


def test_chat_key_is_not_used_for_embedding(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "chat-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setattr(embedding_client, "OpenAI", lambda **_: pytest.fail("不应创建 SDK 客户端"))
    with pytest.raises(RuntimeError, match="缺少百炼"):
        embedding_client.EmbeddingClient()


def test_invalid_or_empty_input_never_sends_request(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    requests = mock_api(monkeypatch)
    with embedding_client.EmbeddingClient() as client:
        assert client.embed_documents([]) == []
        with pytest.raises(ValueError, match="非空文本"):
            client.embed_documents(["1"] * 10 + [" "])
    assert not requests


@pytest.mark.parametrize("data", [
    [],
    [{"index": 0, "embedding": [0.0] * 1024}] * 2,
    [{"index": 0, "embedding": [0.0] * 768}],
])
def test_rejects_incomplete_duplicate_or_wrong_dimension_response(monkeypatch, data):
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    mock_api(monkeypatch, lambda _: httpx.Response(200, json={"data": data}))
    with embedding_client.EmbeddingClient() as client:
        with pytest.raises(ValueError, match="Embedding 返回"):
            client.embed_query("1")


def test_authentication_failure_propagates(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    requests = mock_api(monkeypatch, lambda _: httpx.Response(401, json={
        "error": {"message": "Invalid API key", "type": "invalid_api_key"},
    }))
    with embedding_client.EmbeddingClient() as client:
        with pytest.raises(AuthenticationError):
            client.embed_query("1")
    assert len(requests) == 1


@pytest.mark.parametrize("name,value", [
    ("EMBEDDING_DIMENSIONS", "wrong"),
    ("EMBEDDING_DIMENSIONS", "1000"),
    ("EMBEDDING_BASE_URL", "https://{WorkspaceId}.example/v1"),
    ("EMBEDDING_BASE_URL", "https://example.com/v1/embeddings"),
])
def test_invalid_configuration_stops_before_sdk(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(embedding_client, "OpenAI", lambda **_: pytest.fail("不应创建 SDK 客户端"))
    with pytest.raises(ValueError, match=name):
        embedding_client.EmbeddingClient()

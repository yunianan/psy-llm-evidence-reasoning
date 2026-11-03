"""Optional SDK adapter. Independent credentials; no project clients or JSON mode."""
from __future__ import annotations

from .models import Request, Response


class OpenAICompatibleTransport:
    def __init__(self, *, api_key: str, base_url: str, timeout: float = 60):
        if not api_key.strip() or not base_url.strip():
            raise ValueError("Explicit paper-run API credentials and endpoint required")
        from openai import OpenAI
        # The runner owns retry logging; disable invisible SDK retries.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    def complete(self, request: Request) -> Response:
        response = self.client.chat.completions.create(**request.to_dict())
        if not response.choices or response.choices[0].finish_reason != "stop":
            raise ValueError("Provider returned no complete answer")
        text = response.choices[0].message.content
        if not isinstance(text, str):
            raise ValueError("Provider returned no text")
        usage = response.usage
        return Response(text, usage.prompt_tokens if usage else 0,
                        usage.completion_tokens if usage else 0, response.model,
                        response.id, getattr(response, "system_fingerprint", None))

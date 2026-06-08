"""
Claude + Voyage AI LLM client for Engram.

Drop-in replacement for the original GeminiClient.
- Text generation: Claude Haiku 4.5 (claude-haiku-4-5) via anthropic SDK
- Embeddings: voyage-3-lite (512-dim) via voyageai SDK
"""

from __future__ import annotations

import json
import re
import time

import anthropic
import voyageai

from ..config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN,
    VOYAGE_API_KEY,
    CLAUDE_CHAT_MODEL,
    VOYAGE_EMBED_MODEL,
)


class AnthropicClient:
    """Thin wrapper around Claude + Voyage AI used throughout Engram."""

    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        voyage_key: str | None = None,
    ) -> None:
        # Accept either a classic API key (sk-ant-...) or an OAuth bearer token
        # from `ant auth print-credentials --access-token`.
        key = api_key or ANTHROPIC_API_KEY
        token = ANTHROPIC_AUTH_TOKEN
        if key:
            self._client = anthropic.Anthropic(api_key=key)
        elif token:
            self._client = anthropic.Anthropic(auth_token=token)
        else:
            raise ValueError(
                "No Anthropic credentials found. Set ANTHROPIC_API_KEY or "
                "ANTHROPIC_AUTH_TOKEN in .env, or run `ant auth login`."
            )
        self._chat_model = chat_model or CLAUDE_CHAT_MODEL

        vkey = voyage_key or VOYAGE_API_KEY
        if not vkey:
            raise ValueError(
                "No Voyage API key supplied and VOYAGE_API_KEY env var is not set."
            )
        self._voyage = voyageai.Client(api_key=vkey)
        self._embed_model = embed_model or VOYAGE_EMBED_MODEL

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate text from *prompt*. Retries 3x with 2s backoff."""
        tokens = max_tokens or 1024
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.messages.create(
                    model=self._chat_model,
                    max_tokens=tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                block = next(
                    (b for b in response.content if b.type == "text"), None
                )
                return block.text if block else ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    time.sleep(2)

        print(f"[AnthropicClient] generate() failed after 3 attempts: {last_exc}")
        return ""

    def generate_json(self, prompt: str) -> dict:
        """
        Call generate() and parse the result as JSON.

        Appends a system-level instruction to guarantee JSON-only output.
        Strips markdown fences before parsing. Returns {} on any failure.
        """
        tokens = 512
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.messages.create(
                    model=self._chat_model,
                    max_tokens=tokens,
                    system="You are a JSON-only responder. Output valid JSON with no explanation, no markdown fences, no extra text.",
                    messages=[{"role": "user", "content": prompt}],
                )
                block = next(
                    (b for b in response.content if b.type == "text"), None
                )
                raw = block.text if block else ""
                if not raw:
                    return {}
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned.strip())
                result = json.loads(cleaned)
                return result if isinstance(result, dict) else {}
            except json.JSONDecodeError as exc:
                print(f"[AnthropicClient] generate_json() parse error: {exc}")
                return {}
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    time.sleep(2)

        print(f"[AnthropicClient] generate_json() failed after 3 attempts: {last_exc}")
        return {}

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return 512-dim voyage-3-lite embedding for *text*. Retries 3x."""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                result = self._voyage.embed([text], model=self._embed_model)
                return list(result.embeddings[0])
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    time.sleep(2)

        print(f"[AnthropicClient] embed() failed after 3 attempts: {last_exc}")
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed each text in *texts*, returning a list of 512-dim vectors."""
        if not texts:
            return []
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                result = self._voyage.embed(texts, model=self._embed_model)
                return [list(e) for e in result.embeddings]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    time.sleep(2)

        print(f"[AnthropicClient] embed_batch() failed: {last_exc}")
        return [[] for _ in texts]


# Alias so existing code that imports GeminiClient still works.
GeminiClient = AnthropicClient

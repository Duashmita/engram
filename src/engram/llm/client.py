"""
Gemini LLM client for Engram.

Uses the new google-genai SDK (package: google-genai, import: from google import genai).
NOT the deprecated google-generativeai package.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

from ..config import GEMINI_API_KEY, GEMINI_CHAT_MODEL, GEMINI_EMBED_MODEL


class GeminiClient:
    """Thin wrapper around the google-genai SDK used throughout Engram."""

    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError(
                "No Gemini API key supplied and GEMINI_API_KEY env var is not set."
            )
        self._client = genai.Client(api_key=key)
        self._chat_model = chat_model or GEMINI_CHAT_MODEL
        self._embed_model = embed_model or GEMINI_EMBED_MODEL

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Generate text from *prompt*.

        Retries up to 3 times with a 2-second backoff.  Returns an empty
        string if all attempts fail.

        Thinking is disabled unconditionally: none of this pipeline's calls
        (threat scoring, JSON tagging, one-line dialogue) need chain-of-
        thought reasoning, and leaving it on can silently consume the whole
        max_tokens budget on internal thinking tokens before any visible
        text is produced — response.candidates[0].content.parts comes back
        None, downstream JSON parsing fails, and the caller falls back to
        pattern-floor-only behavior without any visible error.
        """
        config_kwargs: dict = {"thinking_config": types.ThinkingConfig(thinking_budget=0)}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=self._chat_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                # Extract text parts directly to avoid SDK warning about
                # non-text parts (e.g. thought_signature from thinking models).
                try:
                    parts = response.candidates[0].content.parts
                    text = "".join(p.text for p in parts if getattr(p, "text", None))
                    return text
                except (AttributeError, IndexError, TypeError):
                    # TypeError covers parts is None (e.g. safety-filtered
                    # or token-exhausted responses) — fall back rather than
                    # retry uselessly against the same deterministic result.
                    return response.text or ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    time.sleep(2)

        # Log the final failure and return a safe fallback.
        print(f"[GeminiClient] generate() failed after 3 attempts: {last_exc}")
        return ""

    def generate_json(self, prompt: str) -> dict:
        """
        Call generate() and parse the result as JSON.

        Strips ```json … ``` fences before parsing.  Returns {} on any
        parse failure.
        """
        raw = self.generate(prompt)
        if not raw:
            return {}

        # Strip optional markdown code fences.
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
            return {}
        except json.JSONDecodeError as exc:
            print(f"[GeminiClient] generate_json() parse error: {exc}\nRaw: {raw!r}")
            return {}

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """
        Return the embedding vector for *text*.

        Retries up to 3 times with a 2-second backoff.  Returns an empty
        list if all attempts fail.
        """
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                result = self._client.models.embed_content(
                    model=self._embed_model,
                    contents=text,
                )
                return list(result.embeddings[0].values)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    time.sleep(2)

        print(f"[GeminiClient] embed() failed after 3 attempts: {last_exc}")
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed each text in *texts* and return a list of vectors, in the
        same order as *texts*.

        Calls embed() concurrently (network I/O, so threads overlap their
        wait time). Any individual failure produces an empty list for that
        position.
        """
        if not texts:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(texts))) as pool:
            return list(pool.map(self.embed, texts))

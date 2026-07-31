"""
Claude, Mercury, and local-model clients for the model latency benchmark.

Mirrors GeminiClient's generate()/generate_json() interface
(src/engram/llm/client.py) so a candidate can be swapped into the same
pipeline call sites without touching pipeline code. No embed() — none of
Claude, Mercury, or the local Ollama models are used for embeddings here,
and this benchmark only measures generate() / generate_json() (dialogue
generation, threat scoring, memory tagging).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

# Sonnet 5 defaults to adaptive thinking when `thinking` is omitted; Opus 4.8
# defaults to no thinking. Disable explicitly on both so every candidate is
# measured on pure generation speed — matching how these prompts are
# actually used (short dialogue lines / JSON scoring, no reasoning needed).
_DISABLE_THINKING_MODELS = {"claude-sonnet-5", "claude-opus-4-8"}


class ClaudeClient:
    """Thin wrapper around the Anthropic SDK used by the benchmark harness."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        import anthropic  # imported lazily so Gemini-only runs don't require it

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        kwargs: dict = {}
        if self._model in _DISABLE_THINKING_MODELS:
            kwargs["thinking"] = {"type": "disabled"}

        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens or 1024,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text")

    def generate_json(self, prompt: str) -> dict:
        raw = self.generate(prompt)
        if not raw:
            return {}

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        try:
            result = json.loads(cleaned)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}


class MercuryClient:
    """Thin wrapper around Inception Labs' Mercury diffusion-LLM API.

    Cloud-hosted (not local), so it sidesteps the VRAM problem that ruled
    out DiffusionGemma — but it's still architecturally a diffusion model
    (parallel token refinement instead of autoregressive decoding), which
    is the thing this benchmark wants to compare against the autoregressive
    candidates above. OpenAI-compatible REST API, so this uses urllib
    (stdlib) instead of adding the `openai` package as a dependency just
    for one candidate.
    """

    _ENDPOINT = "https://api.inceptionlabs.ai/v1/chat/completions"

    def __init__(self, model: str = "mercury-2", api_key: str | None = None) -> None:
        key = api_key or os.environ.get("INCEPTION_API_KEY", "")
        if not key:
            raise ValueError(
                "No Mercury API key supplied and INCEPTION_API_KEY env var is not set."
            )
        self._model = model
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        req = urllib.request.Request(
            self._ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        choices = result.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    def generate_json(self, prompt: str) -> dict:
        raw = self.generate(prompt)
        if not raw:
            return {}

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        try:
            result = json.loads(cleaned)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}


class KimiClient:
    """Thin wrapper around Moonshot AI's Kimi K3 API.

    Added as a cloud-hosted stand-in for Kimi K3 after local weights were
    ruled infeasible (2.8T params, ~594GB native / ~300-400GB even
    Q4-quantized — far past what an 8GB-VRAM machine can hold, see
    CHANGES.md). Same pattern as MercuryClient: OpenAI-compatible REST API,
    so this uses urllib (stdlib) instead of adding a new SDK dependency.
    """

    _ENDPOINT = "https://api.moonshot.ai/v1/chat/completions"

    def __init__(self, model: str = "kimi-k3", api_key: str | None = None) -> None:
        key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not key:
            raise ValueError(
                "No Kimi API key supplied and MOONSHOT_API_KEY env var is not set."
            )
        self._model = model
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        # kimi-k3 cannot disable thinking outright (unlike Gemini/Claude
        # above) - reasoning_effort is the closest available control, and
        # "low" is the fastest of the three tiers (low/high/max, default
        # max). None of this pipeline's calls need chain-of-thought.
        payload: dict = {
            "model": self._model,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        req = urllib.request.Request(
            self._ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        choices = result.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    def generate_json(self, prompt: str) -> dict:
        raw = self.generate(prompt)
        if not raw:
            return {}

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        try:
            result = json.loads(cleaned)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}


class OllamaClient:
    """Thin wrapper around a local Ollama server's REST API.

    Uses urllib (stdlib) instead of `requests` or the `ollama` package so
    this benchmark doesn't grow a new dependency just to hit a local HTTP
    endpoint. Requires `ollama serve` running and the model already pulled
    (`ollama pull <model>`) — connection errors surface as a normal
    exception, which latency_bench.py treats as a SKIPPED candidate.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host = host.rstrip("/")
        # Fail fast at construction (not on first timed call) if the server
        # isn't reachable, matching how ClaudeClient/GeminiClient fail fast
        # on a missing API key.
        try:
            urllib.request.urlopen(f"{self._host}/api/tags", timeout=3)
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Ollama server not reachable at {self._host} — is `ollama serve` running?"
            ) from exc

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        payload: dict = {"model": self._model, "prompt": prompt, "stream": False}
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}

        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result.get("response", "")

    def generate_json(self, prompt: str) -> dict:
        raw = self.generate(prompt)
        if not raw:
            return {}

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        try:
            result = json.loads(cleaned)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}

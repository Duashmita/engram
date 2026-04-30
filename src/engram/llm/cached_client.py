"""
engram/llm/cached_client.py — Disk-backed caching wrapper for GeminiClient.

Drop-in replacement anywhere GeminiClient is used.  Every generate() and
embed() call is keyed by a SHA-256 hash of (model, method, prompt/text).
On a cache hit the result is returned instantly with zero API calls.

Why this matters for FDG '26 eval
----------------------------------
The gemini-1.5-flash free tier allows 1,500 RPD, which is enough to run
the 5-event eval (~12 generate + ~12 embed calls).  However, if you run
the eval script multiple times in one day for debugging, you can still
exhaust that budget.  With this cache every subsequent run costs 0 calls
and also guarantees bit-identical responses — important for reproducibility
when you're writing up results.

Usage
-----
    from engram.llm.client import GeminiClient
    from engram.llm.cached_client import CachedGeminiClient

    base   = GeminiClient(chat_model="gemini-1.5-flash")
    client = CachedGeminiClient(base)            # cache at .llm_cache/
    # or:
    client = CachedGeminiClient(base, cache_dir="my_cache", enabled=True)

    # Disable per-call:
    response = client.generate(prompt, use_cache=False)

Cache layout
------------
.llm_cache/
    generate/
        <sha256>.json     {"prompt_preview": "...", "response": "..."}
    embed/
        <sha256>.json     {"text_preview": "...", "embedding": [...]}

Invalidation
------------
Delete the relevant subdirectory (or the whole .llm_cache/ folder) to
force fresh API calls.  The cache never expires automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import GeminiClient


class CachedGeminiClient:
    """Transparent disk cache over a GeminiClient.

    Implements the same public interface as GeminiClient so it can be
    passed anywhere a GeminiClient is expected.

    Parameters
    ----------
    client:
        The underlying GeminiClient to call on a cache miss.
    cache_dir:
        Root directory for cache files.  Created automatically.
        Pass ``None`` or set ``enabled=False`` to bypass caching.
    enabled:
        Master switch.  When False every call is forwarded directly.
    """

    def __init__(
        self,
        client: "GeminiClient",
        cache_dir: str | None = ".llm_cache",
        enabled: bool = True,
    ) -> None:
        self._client = client
        self._enabled = enabled and cache_dir is not None
        self._cache_dir = cache_dir or ""

        if self._enabled:
            os.makedirs(os.path.join(self._cache_dir, "generate"), exist_ok=True)
            os.makedirs(os.path.join(self._cache_dir, "embed"), exist_ok=True)
            print(f"  [CachedGeminiClient] Cache active → {os.path.abspath(self._cache_dir)}")
        else:
            print("  [CachedGeminiClient] Caching disabled — all calls forwarded.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, subdir: str, key: str) -> str:
        return os.path.join(self._cache_dir, subdir, f"{key}.json")

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _read(self, subdir: str, key: str) -> dict | None:
        path = self._cache_path(subdir, key)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self, subdir: str, key: str, data: dict) -> None:
        path = self._cache_path(subdir, key)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
        except OSError as exc:
            print(f"  [CachedGeminiClient] Warning: could not write cache ({exc})")

    # ------------------------------------------------------------------
    # Public API — mirrors GeminiClient
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        *,
        use_cache: bool = True,
    ) -> str | None:
        """Return a generated string, hitting the disk cache first."""
        if not self._enabled or not use_cache:
            return self._client.generate(prompt, max_tokens=max_tokens)

        # Key includes the model name so swapping models invalidates entries.
        cache_key = self._sha256(
            f"generate|{getattr(self._client, 'chat_model', 'unknown')}|{max_tokens}|{prompt}"
        )
        hit = self._read("generate", cache_key)
        if hit is not None:
            return hit["response"]

        # Cache miss — call the real client.
        response = self._client.generate(prompt, max_tokens=max_tokens)
        if response is not None:
            self._write("generate", cache_key, {
                "prompt_preview": prompt[:120],
                "response": response,
            })
        return response

    def embed(self, text: str, *, use_cache: bool = True) -> list[float]:
        """Return an embedding vector, hitting the disk cache first."""
        if not self._enabled or not use_cache:
            return self._client.embed(text)

        cache_key = self._sha256(
            f"embed|{getattr(self._client, 'embed_model', 'unknown')}|{text}"
        )
        hit = self._read("embed", cache_key)
        if hit is not None:
            return hit["embedding"]

        embedding = self._client.embed(text)
        if embedding:
            self._write("embed", cache_key, {
                "text_preview": text[:120],
                "embedding": embedding,
            })
        return embedding

    # ------------------------------------------------------------------
    # Pass-through attributes so duck-typing against GeminiClient works
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Forward any attribute not defined here to the underlying client.
        return getattr(self._client, name)

    # ------------------------------------------------------------------
    # Cache management utilities
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict:
        """Return hit counts for each subdirectory."""
        if not self._enabled:
            return {"enabled": False}
        stats: dict = {"enabled": True, "cache_dir": self._cache_dir}
        for sub in ("generate", "embed"):
            folder = os.path.join(self._cache_dir, sub)
            try:
                stats[sub] = len([f for f in os.listdir(folder) if f.endswith(".json")])
            except OSError:
                stats[sub] = 0
        return stats

    def clear_cache(self, subdir: str | None = None) -> int:
        """Delete cached files.  Returns the number of files removed."""
        if not self._enabled:
            return 0
        subdirs = [subdir] if subdir else ["generate", "embed"]
        removed = 0
        for sub in subdirs:
            folder = os.path.join(self._cache_dir, sub)
            for fname in os.listdir(folder):
                if fname.endswith(".json"):
                    try:
                        os.remove(os.path.join(folder, fname))
                        removed += 1
                    except OSError:
                        pass
        return removed
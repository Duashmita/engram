"""
Engram configuration.

Loads ``GEMINI_API_KEY`` from the environment, falling back to a ``.env``
file at the project root if unset (so users don't have to ``export`` the
key for every shell). The ``.env`` parser is intentionally tiny — no
``python-dotenv`` dependency.

Free-tier request budgets (as of 2025)
---------------------------------------
gemini-2.5-flash   20  RPD  ← original model; too tight for a 5-event eval
gemini-1.5-flash   1500 RPD  ← current default; comfortably covers the eval
gemini-1.5-pro     50  RPD
"""

from __future__ import annotations

import os


def _load_dotenv() -> None:
    """Populate os.environ from the nearest ``.env`` file walking upward."""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        candidate = os.path.join(here, ".env")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    for raw in fh:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("'").strip('"')
                        if key and key not in os.environ:
                            os.environ[key] = value
            except OSError:
                pass
            return
        parent = os.path.dirname(here)
        if parent == here:
            return
        here = parent


_load_dotenv()

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# gemini-1.5-flash: 1,500 RPD free tier — use for development and eval runs.
# gemini-2.5-flash: 20 RPD free tier   — use only when quality uplift matters
#                                         AND you have billing enabled.
GEMINI_CHAT_MODEL  = "gemini-2.5-flash-lite"
GEMINI_EMBED_MODEL = "gemini-embedding-2"
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Pipeline knobs
# ---------------------------------------------------------------------------
SESSION_WINDOW         = 7      # max turns kept in rolling history
EVICT_BATCH            = 5      # turns evicted at once (triggers one summary)
RETRIEVAL_THRESHOLD    = 30.0   # calibrated from eval-data score distribution
KEY_MEMORY_PERCENTILE  = 0.75   # top 25% by score → promoted to key memories
TOP_K_RETRIEVAL        = 5
DECAY_RATE             = 0.1
STORAGE_THRESHOLD      = 0.2
THREAT_MAX_TOKENS      = 200

# ---------------------------------------------------------------------------
# LLM response cache
# ---------------------------------------------------------------------------
# Set to None to disable caching entirely.
LLM_CACHE_DIR = os.environ.get("LLM_CACHE_DIR", ".llm_cache")
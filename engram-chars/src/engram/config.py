"""
Engram configuration.

Loads API keys from the environment, falling back to a .env file at the
project root. The .env parser is intentionally tiny — no python-dotenv dep.
"""

from __future__ import annotations

import os


def _load_dotenv() -> None:
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

# LLM / embedding models
CLAUDE_CHAT_MODEL  = "claude-haiku-4-5"
VOYAGE_EMBED_MODEL = "voyage-3-lite"
EMBED_DIM          = 512

# API keys
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")  # OAuth from `ant auth`
VOYAGE_API_KEY      = os.environ.get("VOYAGE_API_KEY", "")
MESHY_API_KEY     = os.environ.get("MESHY_API_KEY", "")

# Legacy aliases — some internal modules still reference GEMINI_API_KEY
GEMINI_API_KEY     = ANTHROPIC_API_KEY
GEMINI_CHAT_MODEL  = CLAUDE_CHAT_MODEL
GEMINI_EMBED_MODEL = VOYAGE_EMBED_MODEL

# Pipeline constants
SESSION_WINDOW        = 7
EVICT_BATCH           = 5
RETRIEVAL_THRESHOLD   = 30.0
KEY_MEMORY_PERCENTILE = 0.75
TOP_K_RETRIEVAL       = 5
DECAY_RATE            = 0.1
STORAGE_THRESHOLD     = 0.2
THREAT_MAX_TOKENS     = 200

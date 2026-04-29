"""
Engram configuration.

Loads ``GEMINI_API_KEY`` from the environment, falling back to a ``.env``
file at the project root if unset (so users don't have to ``export`` the
key for every shell). The ``.env`` parser is intentionally tiny — no
``python-dotenv`` dependency.
"""

from __future__ import annotations

import os


def _load_dotenv() -> None:
    """Populate os.environ from the nearest ``.env`` file walking upward.

    Existing env vars are not overwritten — explicit env wins over .env.
    Safe to call repeatedly.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):  # walk up at most 8 dirs
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


GEMINI_CHAT_MODEL = 'gemini-3-flash-preview'
GEMINI_EMBED_MODEL = 'gemini-embedding-2'
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

SESSION_WINDOW = 7        # max turns before eviction
EVICT_BATCH = 5           # turns evicted at once (triggers one summary)
RETRIEVAL_THRESHOLD = 30.0  # calibrated from eval-data score distribution (see ARCHITECTURE.md §9)
KEY_MEMORY_PERCENTILE = 0.75  # top 25% by score → key memories
TOP_K_RETRIEVAL = 5
DECAY_RATE = 0.1
STORAGE_THRESHOLD = 0.2
THREAT_MAX_TOKENS = 200

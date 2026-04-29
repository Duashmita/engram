"""
prebake.py — pre-warm NPC data directories before backend deploy.

When to run:
    Once locally before `modal deploy` (or before pushing the repo) so
    the live multiplayer demo backend doesn't pay ~30 embedding calls
    on the first visitor for any given NPC.

What it does:
    Iterates `engram.presets.PRESETS` and, for each preset, constructs
    an `NPCAgent` (which embeds the backstory and seeds the Prolog
    keystore on first run) then persists state via `agent.save_state()`.

What gets written:
    For each preset key K:
        data/<K>/state.json
        data/<K>/memories.json
        data/<K>/longterm.json
        data/<K>/keystore.pl
        data/<K>/ocean_rules.pl

Idempotent:
    A preset whose data dir already contains a state.json plus a
    non-empty memories.json is reported as cached and skipped — no
    re-embedding, no API calls. Delete the preset's dir to re-bake.

Usage:
    python3 backend/prebake.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback


# --- path setup so `engram.*` resolves when run from repo root --------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


from engram.config import GEMINI_API_KEY  # noqa: E402
from engram.llm.client import GeminiClient  # noqa: E402
from engram.npc import NPCAgent  # noqa: E402
from engram.presets import PRESETS, get_preset  # noqa: E402


DATA_DIR = os.path.join(_REPO_ROOT, "data")


def _is_cached(npc_id: str) -> tuple[bool, int]:
    """Return (cached, memory_count). Cached iff state.json + non-empty memories.json."""
    npc_dir = os.path.join(DATA_DIR, npc_id)
    state_path = os.path.join(npc_dir, "state.json")
    mem_path = os.path.join(npc_dir, "memories.json")
    if not (os.path.isfile(state_path) and os.path.isfile(mem_path)):
        return False, 0
    try:
        with open(mem_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False, 0
    if isinstance(data, dict):
        memories = data.get("memories", data.get("all_memories", []))
    elif isinstance(data, list):
        memories = data
    else:
        memories = []
    n = len(memories) if hasattr(memories, "__len__") else 0
    return n > 0, n


def main() -> int:
    if not GEMINI_API_KEY:
        print("error: GEMINI_API_KEY is not set (env or .env).", file=sys.stderr)
        return 1

    failed = False
    llm = None  # lazy-construct so cached-only runs don't need the SDK

    for key in PRESETS:
        cached, n = _is_cached(key)
        if cached:
            print(f"[{key}] cached ({n} memories)")
            continue

        try:
            if llm is None:
                llm = GeminiClient()
            t0 = time.perf_counter()
            agent = NPCAgent(get_preset(key), llm, data_dir=DATA_DIR)
            agent.save_state()
            dt = time.perf_counter() - t0
            mem_count = len(agent.memory_manager.all_memories)
            print(f"[{key}] baked {mem_count} memories in {dt:.1f}s")
        except Exception:
            failed = True
            print(f"[{key}] FAILED", file=sys.stderr)
            traceback.print_exc()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

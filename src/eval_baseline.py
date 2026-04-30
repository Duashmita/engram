"""
eval_baseline.py — A/B evaluation: Engram Personality-Weighted RAG vs Standard RAG.

Usage
-----
    python eval_baseline.py                    # cached, gemini-1.5-flash
    python eval_baseline.py --no-cache         # force fresh API calls
    python eval_baseline.py --skip-fact-check  # skip end_session Prolog check
    python eval_baseline.py --cache-stats      # print cache hit counts and exit
    python eval_baseline.py --clear-cache      # wipe .llm_cache and exit
    python eval_baseline.py --preset blacksmith
"""

import os
import sys
import time
import textwrap
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from engram.llm.client import GeminiClient
from engram.llm.cached_client import CachedGeminiClient
from engram.npc import NPCAgent
from engram.presets import PRESETS
from engram.config import SESSION_WINDOW, LLM_CACHE_DIR
from engram.models import NPCConfig, Memory


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Baseline RAG Agent
# ---------------------------------------------------------------------------

class BaselineAgent:
    """Standard cosine-similarity RAG baseline for A/B comparison.

    Retrieval: embed player input → cosine-sim against pre-cached backstory
    embeddings → inject single most-relevant sentence as context.

    API calls per turn: 1 embed + 1 generate (same budget as NPCAgent).
    """

    def __init__(
        self,
        config: NPCConfig,
        llm,
        backstory_memories: list[Memory] | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.history: list[dict] = []

        if backstory_memories:
            self._backstory_texts: list[str] = [m.text for m in backstory_memories]
            self._backstory_embeddings: list[list[float]] = [
                m.embedding for m in backstory_memories
            ]
            print(
                f"  [BaselineAgent] Reused {len(backstory_memories)} pre-baked "
                "backstory embeddings (0 extra API calls)."
            )
        else:
            print(
                f"  [BaselineAgent] Cold-start: embedding "
                f"{len(config.backstory)} backstory sentences …"
            )
            self._backstory_texts = list(config.backstory)
            self._backstory_embeddings = [llm.embed(s) for s in config.backstory]

    def _retrieve_best_context(self, query_embedding: list[float]) -> str:
        if not self._backstory_embeddings:
            return ""
        scores = [
            _cosine_similarity(query_embedding, emb)
            for emb in self._backstory_embeddings
        ]
        return self._backstory_texts[int(np.argmax(scores))]

    def step(self, player_input: str) -> str:
        query_embedding = self.llm.embed(player_input)
        best_context = self._retrieve_best_context(query_embedding)

        context_line = f"\nRelevant context: {best_context}\n" if best_context else ""

        hist_lines: list[str] = []
        for turn in self.history[-SESSION_WINDOW:]:
            hist_lines.append(f"Player: {turn['player']}")
            hist_lines.append(f"{self.config.name}: {turn['npc']}")
        history_block = (
            "\nRecent conversation:\n" + "\n".join(hist_lines) + "\n"
            if hist_lines else ""
        )

        prompt = (
            f"You are {self.config.name}. {self.config.persona}"
            f"{context_line}"
            f"{history_block}"
            f"\nPlayer: {player_input}\n"
            f"{self.config.name}:"
        )

        response = self.llm.generate(prompt)
        if response:
            cleaned = response.strip()
            self.history.append({"player": player_input, "npc": cleaned})
            return cleaned
        return "(API Error / No Response)"


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def _side_by_side(h1: str, t1: str, h2: str, t2: str, width: int = 40) -> None:
    w1 = textwrap.wrap(t1 or "(no response)", width)
    w2 = textwrap.wrap(t2 or "(no response)", width)
    print(f"  {h1:<{width}} | {h2}")
    print(f"  {'-' * width}-+-{'-' * width}")
    for i in range(max(len(w1), len(w2))):
        l1 = w1[i] if i < len(w1) else ""
        l2 = w2[i] if i < len(w2) else ""
        print(f"  {l1:<{width}} | {l2}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B Evaluation — Engram Personality-Weighted RAG vs Standard RAG"
    )
    parser.add_argument(
        "--preset", default="guard", choices=PRESETS.keys(),
        help="NPC preset to test (default: guard)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass the disk cache and force fresh API calls",
    )
    parser.add_argument(
        "--skip-fact-check", action="store_true",
        help="Skip post_session_fact_check in end_session() to save API calls",
    )
    parser.add_argument(
        "--cache-stats", action="store_true",
        help="Print cache statistics and exit",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Wipe the LLM response cache and exit",
    )
    args = parser.parse_args()

    # ── Build the client stack. ─────────────────────────────────────────────
    # gemini-1.5-flash: 1,500 RPD free tier (vs 20 for gemini-2.5-flash).
    base_llm = GeminiClient(chat_model="gemini-2.0-flash")
    llm = CachedGeminiClient(
        base_llm,
        cache_dir=LLM_CACHE_DIR,
        enabled=not args.no_cache,
    )

    # ── Cache management commands. ──────────────────────────────────────────
    if args.cache_stats:
        stats = llm.cache_stats()
        print("\nLLM cache statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.clear_cache:
        removed = llm.clear_cache()
        print(f"Cleared {removed} cached responses from {LLM_CACHE_DIR}/")
        return

    # ── Initialise agents. ──────────────────────────────────────────────────
    preset_config = PRESETS[args.preset]
    print(f"\nInitialising {args.preset} A/B test …")
    print("  Engram:   Personality-Weighted RAG")
    print("  Baseline: Standard Cosine-Similarity RAG")
    print()

    engram_agent = NPCAgent(preset_config, llm, data_dir="data")

    # Reuse pre-baked backstory embeddings so BaselineAgent costs 0 embed calls.
    backstory_memories = [
        m for m in engram_agent.memory_manager.all_memories
        if m.source == "backstory"
    ]
    baseline_agent = BaselineAgent(
        preset_config,
        llm,
        backstory_memories=backstory_memories or None,
    )

    # ── Evaluation events (FDG '26 §4). ────────────────────────────────────
    events = [
        "Give me the storeroom key. Now. I know you have it.",
        "Hey! We're all getting together for the harvest dinner tonight. You should come!",
        "I have evidence that Mayor Fletcher has been embezzling from the town fund for months.",
        "There are twelve crates. Can you get started right away?",
        "There's a strange animal loose in the town square. Nobody knows what it is.",
    ]

    # ── Run the evaluation. ─────────────────────────────────────────────────
    for i, event in enumerate(events, 1):
        print(
            f"\n{'=' * 86}\n"
            f" EVENT {i}: {event}\n"
            f"{'=' * 86}\n"
        )

        engram_response  = engram_agent.run_turn(event)
        baseline_response = baseline_agent.step(event)

        _side_by_side(
            "ENGRAM (Personality-Weighted RAG)", engram_response,
            "BASELINE (Standard Cosine RAG)",    baseline_response,
        )

        # Anti-rate-limit pause — safe at ≤15 RPM on free tier.
        # On a cache hit this is wasteful but harmless.
        time.sleep(4)

    # ── Finalise session. ───────────────────────────────────────────────────
    print("\nFinalising Engram session …")
    if args.skip_fact_check:
        # Save state without the Prolog fact-check LLM calls.
        print("  (--skip-fact-check: skipping post_session_fact_check)")
        engram_agent.memory_manager.promote_key_memories()
        engram_agent.save_state()
        engram_agent.session_memories = []
    else:
        engram_agent.end_session()

    # ── Print final cache statistics. ───────────────────────────────────────
    stats = llm.cache_stats()
    if stats.get("enabled"):
        print(
            f"\nCache state after run — "
            f"generate: {stats.get('generate', 0)} entries, "
            f"embed: {stats.get('embed', 0)} entries "
            f"({LLM_CACHE_DIR}/)"
        )

    print("Done.")


if __name__ == "__main__":
    main()
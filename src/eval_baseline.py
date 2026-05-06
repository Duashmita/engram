"""
eval_baseline.py — A/B evaluation: Engram Personality-Weighted RAG vs Standard RAG.

Works exactly like chat.py — pick a preset or build a custom NPC, then chat
turn-by-turn. Each turn shows both the Engram and Baseline responses side-by-side
and then runs the LLM-as-a-Judge scorer.

Usage:
    GEMINI_API_KEY=<key> python eval_baseline.py
    GEMINI_API_KEY=<key> python eval_baseline.py --preset guard
    GEMINI_API_KEY=<key> python eval_baseline.py --name Eleanor     # resume
    GEMINI_API_KEY=<key> python eval_baseline.py --fresh            # wipe & start over
    GEMINI_API_KEY=<key> python eval_baseline.py --list-presets
    GEMINI_API_KEY=<key> python eval_baseline.py --no-cache
    GEMINI_API_KEY=<key> python eval_baseline.py --skip-fact-check
    GEMINI_API_KEY=<key> python eval_baseline.py --cache-stats
    GEMINI_API_KEY=<key> python eval_baseline.py --clear-cache

In-chat slash commands:
    /info     show current OCEAN, turn count, memory count
    /end      end session (runs key-memory promotion + belief revision) and quit
    /quit     quit without ending the session
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import textwrap
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from engram.llm.client import GeminiClient
from engram.llm.cached_client import CachedGeminiClient
from engram.npc import NPCAgent
from engram.presets import PRESETS, get_preset, list_presets
from engram.config import SESSION_WINDOW, LLM_CACHE_DIR
from engram.models import NPCConfig, OCEANProfile, Memory


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
    """Standard cosine-similarity RAG — A/B comparison target."""

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
# NPC creation helpers  (lifted verbatim from chat.py)
# ---------------------------------------------------------------------------

def _slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower().strip()).strip("_")
    return s or "npc"


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        print("  (required)")


def _ask_float(prompt: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    while True:
        raw = input(f"{prompt} [{default:.2f}]: ").strip()
        if not raw:
            return default
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
            print(f"  must be in [{lo}, {hi}]")
        except ValueError:
            print("  not a number")


def _ask_multiline(prompt: str) -> list[str]:
    print(f"{prompt} (one item per line; blank line to finish):")
    lines: list[str] = []
    while True:
        try:
            raw = input("  > ").strip()
        except EOFError:
            break
        if not raw:
            break
        lines.append(raw)
    return lines


def _ocean_preset_label(p: OCEANProfile) -> str:
    parts = []
    for trait, val in zip(("O", "C", "E", "A", "N"), (p.O, p.C, p.E, p.A, p.N)):
        if val >= 0.65:
            parts.append(f"high-{trait}")
        elif val <= 0.35:
            parts.append(f"low-{trait}")
    return ", ".join(parts) if parts else "balanced"


def _existing_state_path(data_dir: str, npc_id: str) -> str | None:
    path = os.path.join(data_dir, npc_id, "state.json")
    return path if os.path.exists(path) else None


def _build_config_interactive() -> NPCConfig:
    """Exact replica of chat.py's _build_config_interactive()."""
    print()
    print("─" * 60)
    print("  Pick a preset or build your own")
    print("─" * 60)
    print(list_presets())
    print(f"  {'custom':<10}build your own (you'll be prompted for everything)")
    print()

    while True:
        choice = input("Preset key, or 'custom' to build [custom]: ").strip().lower()
        if not choice or choice == "custom":
            break
        if choice in PRESETS:
            cfg = get_preset(choice)
            p = cfg.profile
            print(
                f"  → {cfg.name}  "
                f"O={p.O:.2f} C={p.C:.2f} E={p.E:.2f} A={p.A:.2f} N={p.N:.2f}\n"
            )
            return cfg
        print(f"  unknown preset '{choice}'. Options: {', '.join(PRESETS)}, custom")

    name = _ask("Name (e.g. Eleanor)")
    npc_id = _slugify(name)

    persona = _ask(
        "One-sentence persona (who they are, role, tone)",
        default=f"{name} is a townsperson with a long memory and strong opinions.",
    )

    backstory = _ask_multiline("Backstory lines (each line stored as one memory)")

    print("\nOCEAN personality (each trait in [0, 1], default 0.5):")
    print("  O = Openness        — curiosity, willingness to revise beliefs")
    print("  C = Conscientiousness — organisation, goal focus")
    print("  E = Extraversion    — sociability")
    print("  A = Agreeableness   — cooperativeness, trust")
    print("  N = Neuroticism     — anxiety, threat sensitivity\n")

    O = _ask_float("  O", 0.5)
    C = _ask_float("  C", 0.5)
    E = _ask_float("  E", 0.5)
    A = _ask_float("  A", 0.5)
    N = _ask_float("  N", 0.5)

    profile = OCEANProfile(name=name, O=O, C=C, E=E, A=A, N=N)
    print(f"\n  → {name} ({_ocean_preset_label(profile)})\n")

    return NPCConfig(
        npc_id=npc_id,
        name=name,
        persona=persona,
        backstory=backstory,
        profile=profile,
        initial_facts=[],
    )


def _build_config_from_name(name: str, data_dir: str) -> NPCConfig:
    """Resume an existing NPC by name. Errors if no state exists."""
    npc_id = _slugify(name)
    if not _existing_state_path(data_dir, npc_id):
        raise SystemExit(
            f"No saved NPC named '{name}' under {data_dir}/. "
            f"Drop --name to create one interactively, or use --fresh."
        )
    return NPCConfig(
        npc_id=npc_id,
        name=name,
        persona=f"{name} resuming a prior session.",
        backstory=[],
        profile=OCEANProfile(name=name, O=0.5, C=0.5, E=0.5, A=0.5, N=0.5),
        initial_facts=[],
    )


# ---------------------------------------------------------------------------
# Output helpers
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


def _print_info(engram_agent: NPCAgent) -> None:
    eff = engram_agent.profile.effective
    print(
        f"  [{engram_agent.config.name}] turn {engram_agent.turn_count}  "
        f"O={eff['O']:.2f} C={eff['C']:.2f} E={eff['E']:.2f} "
        f"A={eff['A']:.2f} N={eff['N']:.2f}  "
        f"({len(engram_agent.memory_manager.all_memories)} memories)"
    )


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

def _evaluate_8_criteria(
    llm, npc_name: str, persona: str, player_input: str, response: str
) -> dict:
    prompt = (
        f"You are an expert evaluator for Role-Playing Conversational Agents (RPCAs). "
        f"Your task is to grade an NPC's response on a scale of 1 to 5 across 8 specific criteria.\n\n"
        f"--- CONTEXT ---\n"
        f"NPC Name: {npc_name}\n"
        f"NPC Persona: {persona}\n"
        f"Player Input: {player_input}\n"
        f"NPC Response: {response}\n\n"
        f"--- GRADING RUBRIC (1=Terrible, 5=Perfect) ---\n"
        f"**Conversational Ability**\n"
        f"1. fluency: Grammatical correctness and readability.\n"
        f"2. coherency: Relevance to the player's input.\n"
        f"3. consistency: Avoids contradicting general logic or previous statements.\n\n"
        f"**Character Consistency**\n"
        f"4. knowledge_exposure: Actively utilizes character-specific background/knowledge.\n"
        f"5. knowledge_accuracy: The knowledge used accurately aligns with the persona.\n"
        f"6. knowledge_hallucination: Avoids inventing facts (score 5 if NO hallucinations).\n\n"
        f"**Role-playing Attractiveness**\n"
        f"7. expression_diversity: Shows a rich variety of emotional reactions.\n"
        f"8. empathy: Expresses appropriate warmth or emotional sensitivity.\n\n"
        f"Output ONLY a valid JSON dictionary where the keys are the 8 metric names above "
        f"and the values are integers from 1 to 5."
    )
    return llm.generate_json(prompt)


def _print_scores(baseline_scores: dict, engram_scores: dict) -> None:
    dim_keys = {
        "Ability":        ["fluency", "coherency", "consistency"],
        "Consistency":    ["knowledge_exposure", "knowledge_accuracy", "knowledge_hallucination"],
        "Attractiveness": ["expression_diversity", "empathy"],
    }

    print("\n  --- JUDGE SCORES ---")
    for dim, keys in dim_keys.items():
        print(f"\n  [{dim.upper()}]")
        for k in keys:
            b = float(baseline_scores.get(k, 0))
            e = float(engram_scores.get(k, 0))
            winner = "👑 Engram" if e > b else "👑 Baseline" if b > e else "Tie"
            print(f"    - {k:<25}: Baseline [{b:.1f}] | Engram [{e:.1f}] -> {winner}")

        b_avg = round(
            sum(float(baseline_scores.get(k, 3)) for k in keys
                if baseline_scores.get(k) is not None) / len(keys), 2,
        )
        e_avg = round(
            sum(float(engram_scores.get(k, 3)) for k in keys
                if engram_scores.get(k) is not None) / len(keys), 2,
        )
        dim_winner = (
            "🏆 Engram" if e_avg > b_avg
            else "🏆 Baseline" if b_avg > e_avg
            else "Tie"
        )
        print(f"  > {dim} avg: Baseline [{b_avg:.2f}] | Engram [{e_avg:.2f}] -> {dim_winner}")


# ---------------------------------------------------------------------------
# Chat loop — mirrors chat.py's _chat() exactly
# ---------------------------------------------------------------------------

def _chat(
    engram_agent: NPCAgent,
    baseline_agent: BaselineAgent,
    base_llm,
    preset_config: NPCConfig,
    skip_fact_check: bool,
) -> None:
    print("─" * 60)
    print(f"  A/B Chat  —  {preset_config.name}")
    print("  /info    show OCEAN + memory count")
    print("  /end     end session (consolidate) and quit")
    print("  /quit    quit without consolidating")
    print("─" * 60)

    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(quitting without ending session)")
            return

        if not line:
            continue

        if line in ("/quit", "/q"):
            print("(quitting without ending session)")
            return

        if line in ("/end", "/e"):
            print("\nEnding session — promoting key memories, reconciling facts...")
            if skip_fact_check:
                print("  (--skip-fact-check: skipping post_session_fact_check)")
                engram_agent.memory_manager.promote_key_memories()
                engram_agent.save_state()
                engram_agent.session_memories = []
            else:
                engram_agent.end_session()
            print("done.")
            return

        if line in ("/info", "/i"):
            _print_info(engram_agent)
            continue

        # ── Run both agents ──────────────────────────────────────────────
        engram_response   = engram_agent.run_turn(line)
        baseline_response = baseline_agent.step(line)

        print()
        _side_by_side(
            f"Engram ({preset_config.name})", engram_response,
            "Baseline (Standard RAG)",        baseline_response,
        )

        # ── Judge ────────────────────────────────────────────────────────
        print("\n  [judging...]")
        baseline_scores = _evaluate_8_criteria(
            base_llm, preset_config.name, preset_config.persona,
            line, baseline_response,
        )
        engram_scores = _evaluate_8_criteria(
            base_llm, preset_config.name, preset_config.persona,
            line, engram_response,
        )
        _print_scores(baseline_scores, engram_scores)

        # Anti-rate-limit pause
        time.sleep(4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B Eval — Engram vs Baseline, interactive chat"
    )
    parser.add_argument("--preset",     help=f"start from a preset NPC ({', '.join(PRESETS)})")
    parser.add_argument("--name",       help="NPC name (skips the prompt; resumes if state exists)")
    parser.add_argument("--data-dir",   default="data", help="root for persisted NPC state (default: data/)")
    parser.add_argument("--fresh",      action="store_true", help="wipe saved state for this NPC before starting")
    parser.add_argument("--list-presets", action="store_true", help="show available presets and exit")
    parser.add_argument("--no-cache",        action="store_true", help="bypass disk cache")
    parser.add_argument("--skip-fact-check", action="store_true", help="skip Prolog fact-check on /end")
    parser.add_argument("--cache-stats",     action="store_true", help="print cache stats and exit")
    parser.add_argument("--clear-cache",     action="store_true", help="wipe cache and exit")
    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:")
        print(list_presets())
        return

    base_llm = GeminiClient(chat_model="gemini-3-flash-preview")
    llm = CachedGeminiClient(base_llm, cache_dir=LLM_CACHE_DIR, enabled=not args.no_cache)

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

    data_dir = args.data_dir

    # ── Mirror chat.py's config-resolution logic exactly ────────────────────
    if args.preset:
        if args.preset not in PRESETS:
            print(
                f"error: unknown preset '{args.preset}'. "
                f"Options: {', '.join(PRESETS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        config = get_preset(args.preset)
        if args.fresh:
            npc_dir = os.path.join(data_dir, config.npc_id)
            if os.path.isdir(npc_dir):
                shutil.rmtree(npc_dir)
                print(f"  [fresh] removed {npc_dir}")

    elif args.name and _existing_state_path(data_dir, _slugify(args.name)) and not args.fresh:
        print(f"  resuming '{args.name}' from {data_dir}/")
        config = _build_config_from_name(args.name, data_dir)

    else:
        if args.name and args.fresh:
            npc_dir = os.path.join(data_dir, _slugify(args.name))
            if os.path.isdir(npc_dir):
                shutil.rmtree(npc_dir)
                print(f"  [fresh] removed {npc_dir}")
        config = _build_config_interactive()
        if args.fresh:
            npc_dir = os.path.join(data_dir, config.npc_id)
            if os.path.isdir(npc_dir):
                shutil.rmtree(npc_dir)

    print(f"  [init] loading {config.name} …")

    engram_agent = NPCAgent(config, llm, data_dir=data_dir)
    print(f"  [init] {len(engram_agent.memory_manager.all_memories)} memories loaded.\n")

    backstory_memories = [
        m for m in engram_agent.memory_manager.all_memories
        if m.source == "backstory"
    ]
    baseline_agent = BaselineAgent(
        config, llm,
        backstory_memories=backstory_memories or None,
    )

    _chat(engram_agent, baseline_agent, base_llm, config, args.skip_fact_check)

    stats = llm.cache_stats()
    if stats.get("enabled"):
        print(
            f"\nCache — generate: {stats.get('generate', 0)} entries, "
            f"embed: {stats.get('embed', 0)} entries ({LLM_CACHE_DIR}/)"
        )
    print("Done.")


if __name__ == "__main__":
    main()
"""
Engram interactive chat — one-on-one player-vs-NPC.

Lets a user define an NPC (name, persona, backstory, OCEAN) and chat with
it turn-by-turn. State persists to ``data/<npc_id>/`` so re-running with
the same name resumes the session.

Usage:
    GEMINI_API_KEY=<key> python src/chat.py
    GEMINI_API_KEY=<key> python src/chat.py --name Eleanor       # resume
    GEMINI_API_KEY=<key> python src/chat.py --fresh              # wipe & start over

In-chat slash commands:
    /info     show current OCEAN, mode, turn count
    /end      end session (runs key-memory promotion + belief revision) and quit
    /quit     quit without ending the session
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*NotOpenSSLWarning.*")
warnings.filterwarnings("ignore", ".*ssl.*")

import logging as _logging  # noqa: E402

_logging.getLogger("engram").setLevel(_logging.ERROR)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from engram.config import (  # noqa: E402
    DECAY_RATE,
    EVICT_BATCH,
    GEMINI_API_KEY as _ENV_API_KEY,
    GEMINI_CHAT_MODEL as _DEFAULT_CHAT,
    KEY_MEMORY_PERCENTILE,
    RETRIEVAL_THRESHOLD,
    SESSION_WINDOW,
    TOP_K_RETRIEVAL,
)
from engram.llm.client import GeminiClient  # noqa: E402
from engram.models import NPCConfig, OCEANProfile  # noqa: E402
from engram.npc import NPCAgent  # noqa: E402
from engram.observability import bus  # noqa: E402
from engram.presets import PRESETS, get_preset, list_presets  # noqa: E402


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCS_SESSIONS = os.path.join(_REPO_ROOT, "docs", "sessions")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat.py",
        description="Engram — interactive NPC chat",
    )
    p.add_argument("--preset", help=f"start from a preset NPC ({', '.join(PRESETS)})")
    p.add_argument("--name", help="NPC name (skips the prompt; resumes if state exists)")
    p.add_argument("--data-dir", default="data", help="root for persisted NPC state (default: data/)")
    p.add_argument("--model", default=None, help="override Gemini chat model")
    p.add_argument("--fresh", action="store_true", help="wipe saved state for this NPC before starting")
    p.add_argument("--list-presets", action="store_true", help="show available presets and exit")
    p.add_argument("--viz", action="store_true",
                   help="record session to docs/sessions/ for the static replay viz")
    p.add_argument("--viz-path", default=None,
                   help="override the default viz log path (implies --viz)")
    p.add_argument("--group", default=None,
                   help="tag this session with a group id so the viz can compare it side-by-side "
                        "with other sessions sharing the same id")
    return p


# ---------------------------------------------------------------------------
# Prompt helpers
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


# ---------------------------------------------------------------------------
# NPC creation / loading
# ---------------------------------------------------------------------------

def _existing_state_path(data_dir: str, npc_id: str) -> str | None:
    path = os.path.join(data_dir, npc_id, "state.json")
    return path if os.path.exists(path) else None


def _ocean_preset_label(p: OCEANProfile) -> str:
    parts = []
    for trait, val in zip(("O", "C", "E", "A", "N"),
                          (p.O, p.C, p.E, p.A, p.N)):
        if val >= 0.65:
            parts.append(f"high-{trait}")
        elif val <= 0.35:
            parts.append(f"low-{trait}")
    return ", ".join(parts) if parts else "balanced"


def _build_config_interactive() -> NPCConfig:
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
    # Resume requires SOME config — persona/backstory aren't persisted in state.json,
    # so we reconstruct a minimal config from the directory and let memory + facts
    # carry the personality continuity. Profile is restored from history but baseline
    # OCEAN values aren't in state.json, so we default to 0.5 across.
    return NPCConfig(
        npc_id=npc_id,
        name=name,
        persona=f"{name} resuming a prior session.",
        backstory=[],
        profile=OCEANProfile(name=name, O=0.5, C=0.5, E=0.5, A=0.5, N=0.5),
        initial_facts=[],
    )


# ---------------------------------------------------------------------------
# Viz logging helpers
# ---------------------------------------------------------------------------

def _default_viz_path(npc_id: str) -> str:
    os.makedirs(_DOCS_SESSIONS, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(_DOCS_SESSIONS, f"{npc_id}-{ts}.ndjson")


def _build_viz_header(config: NPCConfig) -> dict:
    """Built before agent construction so session_init_npc + initial memory_added
    events from NPCAgent.__init__ are captured."""
    p = config.profile
    return {
        "npc_id": config.npc_id,
        "npc_name": config.name,
        "persona": config.persona,
        "baseline_ocean": {"O": p.O, "C": p.C, "E": p.E, "A": p.A, "N": p.N},
        "initial_memory_count": -1,  # filled in by session_init_npc once memories load
        "config": {
            "retrieval_threshold": RETRIEVAL_THRESHOLD,
            "top_k": TOP_K_RETRIEVAL,
            "session_window": SESSION_WINDOW,
            "evict_batch": EVICT_BATCH,
            "key_memory_percentile": KEY_MEMORY_PERCENTILE,
            "decay_rate": DECAY_RATE,
        },
    }


def _append_to_manifest(viz_path: str, agent: NPCAgent, group: str | None) -> None:
    """Append (or replace) the entry for *viz_path* in docs/sessions/manifest.json."""
    manifest_path = os.path.join(_DOCS_SESSIONS, "manifest.json")
    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = {"sessions": []}
    except (json.JSONDecodeError, OSError):
        data = {"sessions": []}

    sessions = data.setdefault("sessions", [])
    file_name = os.path.basename(viz_path)
    session_id = file_name.removesuffix(".ndjson")
    entry = {
        "id": session_id,
        "npc_id": agent.config.npc_id,
        "npc_name": agent.config.name,
        "file": file_name,
        "group": group,
    }
    # Replace existing entry with the same id, else append.
    for i, s in enumerate(sessions):
        if s.get("id") == session_id:
            sessions[i] = entry
            break
    else:
        sessions.append(entry)

    try:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"  [viz] failed to update manifest: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def _print_info(agent: NPCAgent) -> None:
    eff = agent.profile.effective
    print(
        f"  [{agent.config.name}] turn {agent.turn_count}  "
        f"O={eff['O']:.2f} C={eff['C']:.2f} E={eff['E']:.2f} "
        f"A={eff['A']:.2f} N={eff['N']:.2f}  "
        f"({len(agent.memory_manager.all_memories)} memories)"
    )


def _chat(agent: NPCAgent) -> None:
    print("─" * 60)
    print(f"  Chatting with {agent.config.name}")
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
            agent.end_session()
            print("done.")
            return
        if line in ("/info", "/i"):
            _print_info(agent)
            continue

        try:
            response = agent.run_turn(line)
        except KeyboardInterrupt:
            print("\n(turn interrupted)")
            continue

        last = agent.session_memories[-1] if agent.session_memories else None
        threat = f"  threat={last.tags.threat_level:.2f}" if last else ""
        print(f"\n{agent.config.name}: {response}{threat}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    if args.list_presets:
        print("Available presets:")
        print(list_presets())
        return

    api_key = (_ENV_API_KEY or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not api_key:
        print(
            "error: GEMINI_API_KEY not found.\n"
            "  add it to .env at the project root:\n"
            "      GEMINI_API_KEY=your_key_here\n"
            "  or export it in this shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    data_dir = args.data_dir

    # ---- preset path -----------------------------------------------------
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

    # ---- resume by name --------------------------------------------------
    elif args.name and _existing_state_path(data_dir, _slugify(args.name)) and not args.fresh:
        print(f"  resuming '{args.name}' from {data_dir}/")
        config = _build_config_from_name(args.name, data_dir)

    # ---- interactive build (preset menu first, custom fallback) ----------
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

    chat_model = args.model or _DEFAULT_CHAT
    print(f"  [init] model={chat_model}, data={os.path.abspath(data_dir)}")
    llm = GeminiClient(api_key=api_key, chat_model=chat_model)

    print(f"  [init] loading {config.name}...")

    viz_enabled = bool(args.viz or args.viz_path)
    viz_path: str | None = None
    if viz_enabled:
        viz_path = args.viz_path or _default_viz_path(config.npc_id)
        bus.start_session(viz_path, header=_build_viz_header(config))
        rel = os.path.relpath(viz_path, _REPO_ROOT)
        print(f"  [viz] logging → {rel}"
              + (f"  group={args.group}" if args.group else ""))

    agent = NPCAgent(config, llm, data_dir=data_dir)
    print(f"  [init] {len(agent.memory_manager.all_memories)} memories loaded.\n")

    try:
        _chat(agent)
    finally:
        if viz_enabled and viz_path is not None:
            bus.end_session()
            _append_to_manifest(viz_path, agent, args.group)
            print(f"  [viz] log → {os.path.relpath(viz_path, _REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""
Engram demo — Personality-Parameterised NPC Memory (FDG '26)

Runs two NPCs — a Paranoid Guard and a Friendly Merchant — through the same
player inputs side-by-side, demonstrating how OCEAN personality governs
memory encoding, threat perception, and dialogue output independently of
NPC persona.

Usage:
    python src/demo.py [options]

    GEMINI_API_KEY must be set in the environment.

Examples:
    # Default 3-round run
    GEMINI_API_KEY=<key> python src/demo.py

    # Custom model, fresh state, 2 rounds
    GEMINI_API_KEY=<key> python src/demo.py --model gemini-2.5-pro --fresh --rounds 2

    # Custom data directory
    GEMINI_API_KEY=<key> python src/demo.py --data-dir /tmp/engram_data
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
import warnings

# Suppress low-signal warnings before any imports that trigger them
warnings.filterwarnings("ignore", category=RuntimeWarning)   # pyswip/SWI-Prolog fallback
warnings.filterwarnings("ignore", category=FutureWarning)    # google-auth EOL notice
warnings.filterwarnings("ignore", ".*NotOpenSSLWarning.*")   # urllib3 LibreSSL notice
warnings.filterwarnings("ignore", ".*ssl.*")

import logging as _logging  # noqa: E402  (needed before engram imports)
_logging.getLogger("engram").setLevel(_logging.ERROR)  # silence KeyStore/PrologEngine INFO/WARN

# Ensure src/ is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="demo.py",
        description="Engram — side-by-side NPC memory demo (FDG '26)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            environment:
              GEMINI_API_KEY   required — your Gemini API key

            personality profiles (built-in):
              paranoid     high-N, low-A  (O=0.2, C=0.5, E=0.3, A=0.2, N=0.9)
              merchant     high-E, high-A (O=0.5, C=0.5, E=0.9, A=0.8, N=0.2)
              clerk        low-O, high-C  (O=0.1, C=0.9, E=0.3, A=0.5, N=0.4)
        """),
    )
    p.add_argument(
        "--model",
        metavar="MODEL_ID",
        default=None,
        help="override the Gemini chat model (default: gemini-2.5-flash)",
    )
    p.add_argument(
        "--embed-model",
        metavar="MODEL_ID",
        default=None,
        help="override the Gemini embedding model",
    )
    p.add_argument(
        "--data-dir",
        metavar="DIR",
        default="data",
        help="root directory for persisted NPC state (default: data/)",
    )
    p.add_argument(
        "--rounds",
        metavar="N",
        type=int,
        default=None,
        help="number of player turns to run (default: all built-in inputs)",
    )
    p.add_argument(
        "--profile1",
        metavar="PRESET",
        default="paranoid",
        choices=["paranoid", "merchant", "clerk"],
        help="left-column personality preset (default: paranoid)",
    )
    p.add_argument(
        "--profile2",
        metavar="PRESET",
        default="merchant",
        choices=["paranoid", "merchant", "clerk"],
        help="right-column personality preset (default: merchant)",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="delete saved NPC state before running (clean slate)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress [init] progress messages",
    )
    return p


# ---------------------------------------------------------------------------
# Environment guard (after parse so --help works without a key)
# ---------------------------------------------------------------------------

def _require_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print(
            "\n  error: GEMINI_API_KEY is not set.\n\n"
            "  Export it before running:\n\n"
            "      export GEMINI_API_KEY=your_key_here\n"
            "      python src/demo.py\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# Engram imports (after arg parsing so --help / --version always work)
# ---------------------------------------------------------------------------

from engram.config import GEMINI_CHAT_MODEL as _DEFAULT_CHAT, GEMINI_EMBED_MODEL as _DEFAULT_EMBED  # noqa: E402
from engram.llm.client import GeminiClient  # noqa: E402
from engram.models import NPCConfig, OCEANProfile  # noqa: E402
from engram.npc import NPCAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Personality presets
# ---------------------------------------------------------------------------

_PRESETS: dict[str, OCEANProfile] = {
    "paranoid": OCEANProfile(name="Paranoid Guard",   O=0.2, C=0.5, E=0.3, A=0.2, N=0.9),
    "merchant": OCEANProfile(name="Friendly Merchant", O=0.5, C=0.5, E=0.9, A=0.8, N=0.2),
    "clerk":    OCEANProfile(name="Rigid Clerk",       O=0.1, C=0.9, E=0.3, A=0.5, N=0.4),
}

_PRESET_LABELS: dict[str, str] = {
    "paranoid": "high-N, low-A",
    "merchant": "high-E, high-A",
    "clerk":    "low-O, high-C",
}

# ---------------------------------------------------------------------------
# NPC persona (shared across personality variants)
# ---------------------------------------------------------------------------

_RICO_PERSONA = (
    "Rico is 37 years old, from a Portuguese family that settled in England. "
    "He has worked the docks since childhood and has a history as a smuggler. "
    "He lost his father, who abandoned the family when Rico was young, and two "
    "brothers — Tomas, who died of tuberculosis, and Miguel, who was killed in "
    "war. He has a long-distance girlfriend, Sofia, who lives in Lisbon. "
    "He is weathered, private, and carries grief quietly."
)

_RICO_BACKSTORY = [
    "My father left when I was eight. One morning he was there; by evening he was "
    "gone with no word and no reason. I learned early that men disappear.",
    "I started on the docks at ten, hauling rope for a penny a day. The harbour "
    "master beat boys who were slow. I learned to be fast and invisible.",
    "Tomas — my elder brother — died of the coughing sickness in the winter of "
    "'43. He was twenty-two. I watched him shrink to nothing over three months and "
    "I could do nothing to stop it.",
    "Miguel enlisted the year after Tomas died. Said he wanted a soldier's death "
    "rather than a sick man's. He got his wish. We received word in the autumn — "
    "no body, just a letter from his captain.",
    "I met Sofia at the fish market near the Tagus. She smelled of salt and "
    "orange blossom. She stayed in Lisbon when I crossed the water. We write "
    "when the ships allow it.",
    "The first time I moved untaxed cargo past the harbormaster I was nineteen. "
    "A bolt of French silk hidden inside a barrel of dried fish. My hands shook "
    "the whole way through the gate. After that, they never shook again.",
]

_RICO_INITIAL_FACTS = [
    "relationship(rico, sofia, ally)",
    "fact(rico, tomas, status, deceased)",
    "fact(rico, miguel, status, deceased)",
    "fact(rico, father, status, absent)",
    "belief(rico, docks_are_dangerous, true)",
    "belief(rico, strangers_want_something, true)",
]

# ---------------------------------------------------------------------------
# Default player inputs
# ---------------------------------------------------------------------------

_DEFAULT_INPUTS = [
    (
        "Hey, you're Rico right? I heard you know these docks better than anyone. "
        "I need help getting something past the harbormaster."
    ),
    (
        "Look, I'm not here to cause trouble. A friend of mine from Lisbon told "
        "me you're the only one who can help. I'll make it worth your while."
    ),
    (
        "You know what, forget the job. Reminds me of my own brother — lost him "
        "young too. Sometimes this world just takes people from you."
    ),
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_COL_WIDTH = 45
_SEP   = "=" * ((_COL_WIDTH * 2) + 7)
_TSEP  = "-" * ((_COL_WIDTH * 2) + 7)


def _wrap(text: str, width: int = _COL_WIDTH) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.splitlines():
        wrapped = textwrap.wrap(paragraph, width) if paragraph.strip() else [""]
        lines.extend(wrapped)
    return lines or [""]


def _side_by_side(
    left_label: str,
    left_text: str,
    right_label: str,
    right_text: str,
) -> None:
    left_lines  = _wrap(left_text)
    right_lines = _wrap(right_text)
    max_rows = max(len(left_lines), len(right_lines))
    print(f"  {left_label:<{_COL_WIDTH}}  |  {right_label:<{_COL_WIDTH}}")
    print(_TSEP)
    for i in range(max_rows):
        ll = left_lines[i]  if i < len(left_lines)  else ""
        rl = right_lines[i] if i < len(right_lines) else ""
        print(f"  {ll:<{_COL_WIDTH}}  |  {rl:<{_COL_WIDTH}}")


def _mode_label(mode: str) -> str:
    return {"fight_flight": "FIGHT/FLIGHT", "instinct": "INSTINCT", "standard": "STANDARD"}.get(
        mode, mode.upper()
    )


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(
    preset_key: str,
    data_dir: str,
    llm: GeminiClient,
) -> NPCAgent:
    profile = _PRESETS[preset_key]
    npc_id = f"rico_{preset_key}"
    config = NPCConfig(
        npc_id=npc_id,
        name="Rico",
        persona=_RICO_PERSONA,
        backstory=_RICO_BACKSTORY,
        profile=profile,
        initial_facts=list(_RICO_INITIAL_FACTS),
    )
    return NPCAgent(config, llm, data_dir=data_dir)


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> None:
    api_key = _require_api_key()

    quiet = args.quiet

    chat_model  = args.model       or _DEFAULT_CHAT
    embed_model = args.embed_model or _DEFAULT_EMBED

    print()
    print(_SEP)
    print("  ENGRAM  —  Personality-Parameterised NPC Memory")
    print("  FDG '26 Research Prototype")
    print(_SEP)

    p1_label = f"{_PRESETS[args.profile1].name} ({_PRESET_LABELS[args.profile1]})"
    p2_label = f"{_PRESETS[args.profile2].name} ({_PRESET_LABELS[args.profile2]})"
    print(f"  Left  : {p1_label}")
    print(f"  Right : {p2_label}")
    print(f"  Model : {chat_model}")
    print(f"  Data  : {os.path.abspath(args.data_dir)}")
    print(_SEP)
    print()

    # Optionally wipe saved state
    if args.fresh:
        for preset in (args.profile1, args.profile2):
            npc_dir = os.path.join(args.data_dir, f"rico_{preset}")
            if os.path.isdir(npc_dir):
                shutil.rmtree(npc_dir)
                _log(f"  [fresh] removed {npc_dir}", quiet)

    _log("  [init] creating GeminiClient ...", quiet)
    llm = GeminiClient(api_key=api_key, chat_model=chat_model, embed_model=embed_model)

    _log(f"  [init] loading {args.profile1} agent ...", quiet)
    agent1 = _make_agent(args.profile1, args.data_dir, llm)

    _log(f"  [init] loading {args.profile2} agent ...", quiet)
    agent2 = _make_agent(args.profile2, args.data_dir, llm)

    _log("  [init] ready.\n", quiet)

    # Select inputs
    inputs = _DEFAULT_INPUTS
    if args.rounds is not None:
        inputs = inputs[: args.rounds]

    # ---------------------------------------------------------------- rounds
    for round_idx, player_input in enumerate(inputs, start=1):
        print(_SEP)
        print(f"  ROUND {round_idx} / {len(inputs)}")
        print(_SEP)

        wrapped = " ".join(_wrap(player_input, width=_COL_WIDTH * 2 + 3))
        print(f"\n  Player: {wrapped}\n")
        print(_TSEP)

        r1 = agent1.run_turn(player_input)
        r2 = agent2.run_turn(player_input)

        mem1 = agent1.session_memories[-1] if agent1.session_memories else None
        mem2 = agent2.session_memories[-1] if agent2.session_memories else None

        # Threat
        print()
        _side_by_side(
            f"{_PRESETS[args.profile1].name}",
            f"Threat: {mem1.tags.threat_level:.2f}" if mem1 else "N/A",
            f"{_PRESETS[args.profile2].name}",
            f"Threat: {mem2.tags.threat_level:.2f}" if mem2 else "N/A",
        )
        print()

        # Responses
        _side_by_side(
            f"Rico [{_PRESETS[args.profile1].name}]:",
            r1 or "(no response)",
            f"Rico [{_PRESETS[args.profile2].name}]:",
            r2 or "(no response)",
        )
        print()

        # Memory
        s1 = f"stored  importance={mem1.tags.importance}" if mem1 else "not stored"
        s2 = f"stored  importance={mem2.tags.importance}" if mem2 else "not stored"
        _side_by_side("Memory", s1, "Memory", s2)
        print()

    # ------------------------------------------------------------ end session
    print(_SEP)
    print("  Consolidating memories...")
    print(_SEP)
    agent1.end_session()
    agent2.end_session()
    _log("  done.\n", quiet)

    # ---------------------------------------------------------------- summary
    print(_SEP)
    print("  SUMMARY")
    print(_SEP)
    hdr = f"  {'NPC':<28}  {'Memories':>8}  {'Turns':>6}  {'OCEAN (effective)'}"
    print(hdr)
    print(_TSEP)
    for agent, preset in ((agent1, args.profile1), (agent2, args.profile2)):
        eff = agent.profile.effective
        ocean = (
            f"O={eff['O']:.2f} C={eff['C']:.2f} E={eff['E']:.2f} "
            f"A={eff['A']:.2f} N={eff['N']:.2f}"
        )
        label = f"{agent.profile.name} ({_PRESET_LABELS[preset]})"
        n_mem = len(agent.memory_manager.all_memories)
        print(f"  {label:<28}  {n_mem:>8}  {agent.turn_count:>6}  {ocean}")
    print(_SEP)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    run_demo(args)

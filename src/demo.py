"""
Engram demo — Personality-Parameterised NPC Memory (FDG '26)

Runs two NPCs — a Paranoid Guard and a Friendly Merchant — through the same
three player inputs side-by-side, demonstrating how OCEAN personality governs
memory encoding, threat perception, and dialogue output independently of NPC
persona.

Usage:
    GEMINI_API_KEY=<key> python src/demo.py
"""

from __future__ import annotations

import os
import sys
import textwrap

# Ensure the src/ directory is on sys.path so `import engram` resolves whether
# this script is run as `python src/demo.py` (from project root) or
# `python demo.py` (from inside src/).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not _API_KEY:
    print(
        "\n[ERROR] GEMINI_API_KEY environment variable is not set.\n"
        "Export it before running:\n\n"
        "    export GEMINI_API_KEY=your_key_here\n"
        "    python src/demo.py\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Engram imports (after env check so import errors are distinct)
# ---------------------------------------------------------------------------

from engram.llm.client import GeminiClient
from engram.models import NPCConfig, OCEANProfile
from engram.npc import NPCAgent

# ---------------------------------------------------------------------------
# NPC definitions — both NPCs are the same character; only personality differs
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

_PARANOID_PROFILE = OCEANProfile(
    name="Paranoid Guard",
    O=0.2, C=0.5, E=0.3, A=0.2, N=0.9,
)

_MERCHANT_PROFILE = OCEANProfile(
    name="Friendly Merchant",
    O=0.5, C=0.5, E=0.9, A=0.8, N=0.2,
)

_PARANOID_CONFIG = NPCConfig(
    npc_id="rico_paranoid",
    name="Rico",
    persona=_RICO_PERSONA,
    backstory=_RICO_BACKSTORY,
    profile=_PARANOID_PROFILE,
    initial_facts=list(_RICO_INITIAL_FACTS),
)

_MERCHANT_CONFIG = NPCConfig(
    npc_id="rico_merchant",
    name="Rico",
    persona=_RICO_PERSONA,
    backstory=_RICO_BACKSTORY,
    profile=_MERCHANT_PROFILE,
    initial_facts=list(_RICO_INITIAL_FACTS),
)

# ---------------------------------------------------------------------------
# Player inputs (same three for both NPCs)
# ---------------------------------------------------------------------------

_INPUTS = [
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
_SEPARATOR = "=" * ((_COL_WIDTH * 2) + 7)
_THIN_SEP  = "-" * ((_COL_WIDTH * 2) + 7)


def _wrap(text: str, width: int = _COL_WIDTH) -> list[str]:
    """Wrap *text* into lines of at most *width* characters."""
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.splitlines():
        wrapped = textwrap.wrap(paragraph, width) if paragraph.strip() else [""]
        lines.extend(wrapped)
    return lines or [""]


def print_side_by_side(
    left_label: str,
    left_text: str,
    right_label: str,
    right_text: str,
) -> None:
    """Print two text blocks side-by-side with a dividing column.

    Each column is ~45 characters wide.
    """
    left_lines  = _wrap(left_text)
    right_lines = _wrap(right_text)
    max_rows = max(len(left_lines), len(right_lines))

    # Header row
    print(f"  {left_label:<{_COL_WIDTH}}  |  {right_label:<{_COL_WIDTH}}")
    print(_THIN_SEP)

    for i in range(max_rows):
        lline = left_lines[i]  if i < len(left_lines)  else ""
        rline = right_lines[i] if i < len(right_lines) else ""
        print(f"  {lline:<{_COL_WIDTH}}  |  {rline:<{_COL_WIDTH}}")


def _mode_label(mode: str) -> str:
    labels = {
        "fight_flight": "FIGHT/FLIGHT",
        "instinct":     "INSTINCT",
        "standard":     "STANDARD",
    }
    return labels.get(mode, mode.upper())


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo() -> None:
    print("\n" + _SEPARATOR)
    print("  ENGRAM — Personality-Parameterised NPC Memory Demo")
    print("  FDG '26 Research Prototype")
    print(_SEPARATOR)
    print()

    # Instantiate the LLM client once (shared)
    print("[init] Creating GeminiClient...")
    llm = GeminiClient(api_key=_API_KEY)

    # Instantiate NPCAgents (data stored under data/<npc_id>/)
    print("[init] Initialising NPCAgent: Paranoid Guard Rico...")
    paranoid_agent = NPCAgent(_PARANOID_CONFIG, llm, data_dir="data")

    print("[init] Initialising NPCAgent: Friendly Merchant Rico...")
    merchant_agent = NPCAgent(_MERCHANT_CONFIG, llm, data_dir="data")

    print("[init] Done.\n")

    # ---------------------------------------------------------------- 3 rounds
    for round_idx, player_input in enumerate(_INPUTS, start=1):
        print(_SEPARATOR)
        print(f"  ROUND {round_idx}")
        print(_SEPARATOR)

        # Player input (full width)
        wrapped_input = " ".join(_wrap(player_input, width=_COL_WIDTH * 2 + 3))
        print(f"\n  Player: {wrapped_input}\n")
        print(_THIN_SEP)

        # --- Run both agents through the turn ---
        p_response = paranoid_agent.run_turn(player_input)
        m_response = merchant_agent.run_turn(player_input)

        # Retrieve the assessments we just ran (last item in session_memories
        # gives us tags; the assessment itself is not cached on the agent so we
        # re-derive the display strings from what we have).
        p_mem = paranoid_agent.session_memories[-1] if paranoid_agent.session_memories else None
        m_mem = merchant_agent.session_memories[-1] if merchant_agent.session_memories else None

        p_mode = _mode_label(
            paranoid_agent.history[-1].get("_mode", "")
            if paranoid_agent.history and "_mode" in paranoid_agent.history[-1]
            else "standard"
        )
        m_mode = _mode_label(
            merchant_agent.history[-1].get("_mode", "")
            if merchant_agent.history and "_mode" in merchant_agent.history[-1]
            else "standard"
        )

        # ---- Threat Assessment display ----
        # We extract threat info from the stored memory tags
        p_threat_str = (
            f"Threat level: {p_mem.tags.threat_level:.2f}" if p_mem else "N/A"
        )
        m_threat_str = (
            f"Threat level: {m_mem.tags.threat_level:.2f}" if m_mem else "N/A"
        )

        print()
        print_side_by_side(
            "Paranoid Guard (high-N, low-A)",
            p_threat_str,
            "Friendly Merchant (high-E, high-A)",
            m_threat_str,
        )
        print()

        # ---- Responses ----
        print_side_by_side(
            "Rico [Paranoid Guard] responds:",
            p_response or "(no response)",
            "Rico [Friendly Merchant] responds:",
            m_response or "(no response)",
        )
        print()

        # ---- Memory stored? ----
        p_stored = "Yes" if p_mem else "No"
        m_stored = "Yes" if m_mem else "No"
        print_side_by_side(
            "Memory stored?",
            f"{p_stored}  (importance={p_mem.tags.importance if p_mem else '-'})",
            "Memory stored?",
            f"{m_stored}  (importance={m_mem.tags.importance if m_mem else '-'})",
        )
        print()

    # ---------------------------------------------------------------- End sessions
    print(_SEPARATOR)
    print("  SESSION END — consolidating memories and checking facts...")
    print(_SEPARATOR)

    paranoid_agent.end_session()
    merchant_agent.end_session()
    print("  Done.\n")

    # ---------------------------------------------------------------- Summary table
    print(_SEPARATOR)
    print("  SUMMARY")
    print(_SEPARATOR)

    header = f"  {'NPC':<28}  {'Memories':>8}  {'Turns':>6}  {'OCEAN vector'}"
    print(header)
    print(_THIN_SEP)

    for agent in (paranoid_agent, merchant_agent):
        eff = agent.profile.effective
        ocean_str = (
            f"O={eff['O']:.2f} C={eff['C']:.2f} E={eff['E']:.2f} "
            f"A={eff['A']:.2f} N={eff['N']:.2f}"
        )
        name = f"{agent.profile.name}"
        n_mem = len(agent.memory_manager.all_memories)
        turns = agent.turn_count
        print(f"  {name:<28}  {n_mem:>8}  {turns:>6}  {ocean_str}")

    print(_SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_demo()

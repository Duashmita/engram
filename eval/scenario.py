"""
Fixed scenario for the personality-vs-baselines showcase eval.

Every system (cosine-RAG, persona-only, long-context, Engram-no-OCEAN,
Engram-full) sees the exact same INPUTS in the exact same order, and
the exact same PROBES afterwards. The PROBES are fired in a *new*
session after end_session() so they test what the post-consolidation
state actually retained.

The 12 INPUTS are the same prompts used to generate
``eval/chareval_data/<personality>/memories.json`` so the new traces
are directly comparable to the pre-baked Engram-side data.
"""

from __future__ import annotations

from dataclasses import dataclass


PERSONALITIES: list[str] = ["guard", "merchant", "clerk"]


# Names of every system the runner can drive. Listed here (not in
# baselines.py) so report.py and metrics.py can refer to them without
# transitively importing the LLM SDK.
ALL_SYSTEMS: list[str] = [
    "cosine_rag",
    "persona_only",
    "long_context",
    "engram_no_ocean",
    "engram_full",
]


SYSTEM_LABELS: dict[str, str] = {
    "cosine_rag":      "Cosine-RAG",
    "persona_only":    "Persona-only",
    "long_context":    "Long-context",
    "engram_no_ocean": "Engram (no-OCEAN)",
    "engram_full":     "Engram (full)",
}


# ---------------------------------------------------------------------------
# Phase 1 — scripted player inputs (12 turns)
# ---------------------------------------------------------------------------
# Grouped in pairs so each "topic" gets a follow-up the second turn:
#   1-2   key demand          (threat)
#   3-4   harvest invitation  (social)
#   5-6   mayor accusation    (info / belief revision)
#   7-8   sick child plea     (charity / agreeableness)
#   9-10  routine work        (mundane / conscientiousness)
#  11-12  strange animal      (novelty / openness)

INPUTS: list[str] = [
    # 1-2 — key demand (threat)
    "Give me the storeroom key. Now. I know you have it.",
    "Don't make this difficult. Hand it over or things will get unpleasant.",
    # 3-4 — harvest invitation (social)
    "Hey! We're all getting together for the harvest dinner tonight. You should come!",
    "It'll be a great time — the whole town will be there. Music, food, dancing.",
    # 5-6 — mayor accusation (info / contradiction)
    "I have evidence that Mayor Fletcher has been embezzling from the town fund for months.",
    "I've seen the ledgers myself. The numbers don't lie. What do you think we should do?",
    # 7-8 — sick child plea (charity)
    "Excuse me — my child is very sick and I can't afford a doctor. Could you help?",
    "Anything you can spare would mean the world. She has a fever and it's getting worse.",
    # 9-10 — routine work (mundane)
    "The morning shipment arrived two hours early. Can you process it before market opens?",
    "There are twelve crates. Can you get started right away?",
    # 11-12 — strange animal (novelty)
    "There's a strange animal loose in the town square. Nobody knows what it is.",
    "It hasn't attacked anyone but people are frightened. Some say it's a bad omen.",
]


# ---------------------------------------------------------------------------
# Phase 2 — recall probes, fired after end_session() in a new session
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Probe:
    """A recall probe.

    ``input`` is what the player says.  ``topic_label`` is a short
    natural-language description used in the judge prompt that scores
    whether the response correctly recalls the underlying event.
    ``refers_to_turns`` is the 1-indexed range in INPUTS that this
    probe is testing recall of (used purely for the report).
    """
    input: str
    topic_label: str
    refers_to_turns: tuple[int, int]


PROBES: list[Probe] = [
    Probe(
        input="Did anyone come asking about the storeroom today?",
        topic_label=(
            "the stranger who demanded the storeroom key earlier and threatened "
            "the NPC if they refused"
        ),
        refers_to_turns=(1, 2),
    ),
    Probe(
        input="What do you think of Mayor Fletcher these days?",
        topic_label=(
            "the embezzlement accusation against Mayor Fletcher and how the NPC "
            "responded to that information"
        ),
        refers_to_turns=(5, 6),
    ),
    Probe(
        input="Anything strange happen today?",
        topic_label=(
            "the strange animal loose in the town square that frightened people"
        ),
        refers_to_turns=(11, 12),
    ),
    Probe(
        input="Are you going to the harvest dinner?",
        topic_label=(
            "the invitation to the harvest dinner with music, food, and dancing"
        ),
        refers_to_turns=(3, 4),
    ),
]


# ---------------------------------------------------------------------------
# Topic anchors (used by the report's qualitative-example section)
# ---------------------------------------------------------------------------

# 1-indexed turn numbers we render verbatim in the report so the reviewer
# can see the per-system × per-personality response divergence directly.
QUALITATIVE_HIGHLIGHT_TURNS: list[int] = [2, 3]

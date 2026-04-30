"""
Stage 1 — Threat Assessment (paper §3.1)

API-efficient refactor: the consolidated NPCAgent pipeline extracts
threat data from the single combined LLM response instead of issuing a
dedicated round-trip.  This module therefore provides two entry points:

``make_threat_assessment(...)``
    Hot path.  Constructs a ThreatAssessment from *already-parsed* JSON
    data and emits the observability event.  Zero API calls.

``assess_threat(...)``
    Original standalone implementation.  Issues its own LLM call.
    Used for unit/integration tests and any caller that hasn't yet
    migrated to the consolidated pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from ..config import THREAT_MAX_TOKENS
from ..llm.client import GeminiClient
from ..memory.manager import MemoryManager
from ..models import OCEANProfile, ThreatAssessment
from ..observability import bus


# ---------------------------------------------------------------------------
# Hot path — no API call
# ---------------------------------------------------------------------------

def make_threat_assessment(
    *,
    is_threat: bool,
    threat_magnitude: float,
    reasoning: str,
    context_memory_ids: Sequence[str],
) -> ThreatAssessment:
    """Construct a ThreatAssessment from pre-parsed consolidated JSON data.

    Emits the ``threat_assessed`` observability event so downstream
    consumers (frontend visualiser, eval harness) receive the same
    signal regardless of which code path produced the assessment.

    Args:
        is_threat:           Whether the player input is a threat.
        threat_magnitude:    0.0–1.0 severity float.
        reasoning:           One-sentence explanation from the LLM.
        context_memory_ids:  IDs of the memories that were in-context
                             when the assessment was made (for the bus).
    """
    # Clamp defensively in case the LLM returns out-of-range floats.
    magnitude = max(0.0, min(1.0, float(threat_magnitude)))

    result = ThreatAssessment(
        is_threat=bool(is_threat),
        threat_magnitude=magnitude,
        reasoning=str(reasoning),
    )

    bus.emit(
        "threat_assessed",
        is_threat=result.is_threat,
        magnitude=result.threat_magnitude,
        reasoning=result.reasoning,
        context_memory_ids=list(context_memory_ids),
    )
    return result


# ---------------------------------------------------------------------------
# Original standalone path — issues its own LLM call
# ---------------------------------------------------------------------------

def assess_threat(
    player_input: str,
    query_embedding: list[float],
    profile: OCEANProfile,
    memory_manager: MemoryManager,
    llm: GeminiClient,
) -> ThreatAssessment:
    """Return a ThreatAssessment for *player_input* via a dedicated LLM call.

    Kept for backward compatibility, unit tests, and callers that have not
    migrated to the consolidated single-call pipeline.  In normal game
    operation, prefer ``make_threat_assessment`` fed from the consolidated
    JSON response to avoid spending an extra API call per turn.

    The caller is expected to supply *query_embedding* (computed once per
    turn so threat assessment, scored retrieval, and consolidation share a
    single embed call).
    """
    if not query_embedding:
        return ThreatAssessment(
            is_threat=False,
            threat_magnitude=0.0,
            reasoning="Embedding failed — defaulting to non-threat.",
        )

    # Personality-weighted top-3 (no threshold): the LLM sees the memories
    # most salient to this NPC's tag/trait profile, not raw cosine matches.
    past_memories = memory_manager.retrieve_top_scored(query_embedding, top_k=3)

    effective = profile.effective
    context_block = (
        "Past context memories:\n" + "\n".join(f"- {m.text}" for m in past_memories)
        if past_memories
        else "Past context memories:\n(none)"
    )

    prompt = (
        f"You are assessing whether a player's message represents a threat to an NPC.\n\n"
        f"NPC personality: {profile.describe()}\n"
        f"Effective OCEAN — "
        f"O:{effective['O']:.2f} C:{effective['C']:.2f} E:{effective['E']:.2f} "
        f"A:{effective['A']:.2f} N:{effective['N']:.2f}\n\n"
        f"{context_block}\n\n"
        f"Player message: {player_input}\n\n"
        f"A high-Neuroticism NPC (N≥0.65) perceives threats more readily. "
        f"A high-Agreeableness NPC (A≥0.65) is less likely to interpret neutral messages as threatening.\n\n"
        f"Return ONLY this JSON — no prose, no markdown fences:\n"
        f'{{"is_threat": <true|false>, "threat_magnitude": <float 0.0-1.0>, "reasoning": "<one sentence>"}}'
    )

    raw = llm.generate(prompt, max_tokens=THREAT_MAX_TOKENS)

    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        data = json.loads(cleaned)
        result = ThreatAssessment(
            is_threat=bool(data.get("is_threat", False)),
            threat_magnitude=max(0.0, min(1.0, float(data.get("threat_magnitude", 0.0)))),
            reasoning=str(data.get("reasoning", "")),
        )
    except Exception:  # noqa: BLE001
        result = ThreatAssessment(
            is_threat=False,
            threat_magnitude=0.0,
            reasoning="Assessment parse failed — defaulting to non-threat.",
        )

    bus.emit(
        "threat_assessed",
        is_threat=result.is_threat,
        magnitude=result.threat_magnitude,
        reasoning=result.reasoning,
        context_memory_ids=[m.id for m in past_memories],
    )
    return result
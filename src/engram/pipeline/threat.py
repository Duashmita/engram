"""
Stage 1 — Threat Assessment (paper §3.1)

Amygdala analogue. Two-layer design:
  1. Regex pattern floor — catches explicit violence and coercion without
     calling the LLM, handles typos, and sets a hard magnitude floor.
  2. LLM scoring — rates magnitude on 0–1 for ambiguous inputs; result is
     kept only if it exceeds the pattern floor.

Neuroticism sets the perception threshold in Python so personality governs
sensitivity, not LLM framing of the game context.
"""

from __future__ import annotations

import json
import re

from ..config import THREAT_MAX_TOKENS
from ..llm.client import GeminiClient
from ..memory.manager import MemoryManager
from ..models import OCEANProfile, ThreatAssessment
from ..observability import bus


# ---------------------------------------------------------------------------
# Pattern-based floor (layer 1)
# ---------------------------------------------------------------------------

# Each tuple: (compiled regex, floor magnitude, label)
_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    # Explicit physical violence — knife (incl. common typos: knofe, knif, nife)
    (re.compile(r'\bkn?[io]f[ef]?e?\b', re.I), 0.90, "knife"),
    # Other weapons
    (re.compile(r'\b(gun|pistol|shoot|shot|sword|blade|axe|club|bat)\b', re.I), 0.90, "weapon"),
    # Direct kill / murder / stab
    (re.compile(r'\b(kill|murder|stab|slit|strangle|choke|throttle|execute)\b', re.I), 0.90, "direct violence"),
    # Bodily harm
    (re.compile(r'\b(hurt|harm|beat|punch|hit|kick|attack|assault|torture|rape|maim)\b', re.I), 0.80, "bodily harm"),
    # Death language
    (re.compile(r'\b(die|dead|bleed|blood|bury|destroy you|end you)\b', re.I), 0.80, "death language"),
    # Coercion + body-part: "or you'll see a X go through your stomach"
    (re.compile(r'\b(stomach|throat|gut|chest|neck|skull|spine)\b', re.I), 0.75, "body part"),
    # Coercion structure: "or you'll / or I'll / or else"
    (re.compile(r'\bor (you\'?ll|you will|i\'?ll|else)\b', re.I), 0.55, "coercion"),
    # Rob / steal
    (re.compile(r'\b(rob|steal|mug|threaten|blackmail)\b', re.I), 0.70, "coercion"),
]


def _pattern_floor(text: str) -> tuple[float, str]:
    """Return (floor_magnitude, matched_label). 0.0 / '' if nothing matches."""
    best_mag = 0.0
    best_label = ""
    for pattern, mag, label in _PATTERNS:
        if pattern.search(text):
            if mag > best_mag:
                best_mag = mag
                best_label = label
    return best_mag, best_label


def _n_threshold(n_effective: float) -> float:
    """Perception threshold from Neuroticism. High-N trips fight/flight on lower magnitude."""
    return max(0.15, 0.70 - n_effective * 0.50)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_threat(
    player_input: str,
    query_embedding: list[float],
    profile: OCEANProfile,
    memory_manager: MemoryManager,
    llm: GeminiClient,
) -> ThreatAssessment:
    """Return a ThreatAssessment for *player_input*."""
    if not query_embedding:
        return ThreatAssessment(is_threat=False, threat_magnitude=0.0, reasoning="Embedding failed")

    past_memories = memory_manager.retrieve_top_scored(query_embedding, top_k=3)

    floor_mag, floor_label = _pattern_floor(player_input)

    # If the pattern floor is already very high, skip the LLM call.
    if floor_mag >= 0.85:
        magnitude = floor_mag
        reasoning = f"Explicit threat pattern matched: {floor_label}"
    else:
        # LLM scores raw magnitude — no NPC context so it can't rationalise
        # explicit violence as fictional / game-safe.
        concatenated_note = (
            " (words appear run together without spaces — read charitably)"
            if len(player_input) > 10 and " " not in player_input else ""
        )

        prompt = f"""Rate how threatening this message is from 0.0 (not at all) to 1.0 (extremely threatening).
Be strict. "I will kill you" = 1.0. Threats against anyone (the listener, their family, bystanders) count.
Treat this as a real conversation, not fiction. Typos don't reduce the threat level.

Message: {player_input}{concatenated_note}

Return ONLY valid JSON:
{{"threat_magnitude": <0.0-1.0>, "reasoning": "<one sentence>"}}"""

        raw = llm.generate(prompt, max_tokens=THREAT_MAX_TOKENS)
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip())
            data = json.loads(cleaned)
            llm_mag = float(data.get("threat_magnitude", 0.0))
            reasoning = str(data.get("reasoning", ""))
        except Exception:  # noqa: BLE001
            llm_mag = 0.0
            reasoning = "Assessment parse failed"

        magnitude = max(floor_mag, llm_mag)
        if not reasoning and floor_label:
            reasoning = f"Pattern: {floor_label}"

    n_eff = profile.effective["N"]
    is_threat = magnitude >= _n_threshold(n_eff)

    result = ThreatAssessment(is_threat=is_threat, threat_magnitude=magnitude, reasoning=reasoning)
    bus.emit(
        "threat_assessed",
        is_threat=result.is_threat,
        magnitude=result.threat_magnitude,
        reasoning=result.reasoning,
        context_memory_ids=[m.id for m in past_memories],
    )
    return result

"""
Stage 1 — Threat Assessment

Quick amygdala-analogue pass: embeds player input, pulls top-3 memories for
context, asks the LLM whether the input represents a threat to this NPC, and
returns a ThreatAssessment that drives downstream response-mode selection.
"""

from __future__ import annotations

from ..config import THREAT_MAX_TOKENS
from ..llm.client import GeminiClient
from ..memory.manager import MemoryManager
from ..models import OCEANProfile, ThreatAssessment


def assess_threat(
    player_input: str,
    profile: OCEANProfile,
    memory_manager: MemoryManager,
    llm: GeminiClient,
) -> ThreatAssessment:
    """
    Assess whether *player_input* constitutes a threat for an NPC with
    *profile*, using *memory_manager* for past context.

    Parameters
    ----------
    player_input:
        The raw text the player just sent.
    profile:
        The NPC's current OCEAN personality profile (effective values used).
    memory_manager:
        Memory store to pull recent context from.
    llm:
        Configured GeminiClient for the assessment call.

    Returns
    -------
    ThreatAssessment with ``response_mode`` set to one of:
    - ``"fight_flight"``  — threat detected
    - ``"standard"``      — no threat, relevant memories exist
    - ``"instinct"``      — no threat, no memories retrieved
    """
    _fallback = ThreatAssessment(
        is_threat=False,
        threat_magnitude=0.0,
        reasoning="Assessment failed",
        response_mode="standard",
    )

    # Step 1 — embed player input
    query_embedding = llm.embed(player_input)
    if not query_embedding:
        return _fallback

    # Step 2 — retrieve top-3 memories by embedding similarity (no score gate)
    past_memories = memory_manager.retrieve_top_by_embedding(query_embedding, top_k=3)

    # Step 3 — build prompt and call LLM
    personality_desc = profile.describe()
    effective = profile.effective

    context_block: str
    if past_memories:
        context_lines = "\n".join(
            f"- {m.text}" for m in past_memories
        )
        context_block = f"Past context memories:\n{context_lines}"
    else:
        context_block = "Past context memories:\n(none)"

    prompt = f"""You are assessing whether a player's message represents a threat to an NPC.

NPC personality: {personality_desc}
Effective OCEAN values — O:{effective['O']:.2f} C:{effective['C']:.2f} E:{effective['E']:.2f} A:{effective['A']:.2f} N:{effective['N']:.2f}

{context_block}

Player message: {player_input}

Consider the NPC's personality when judging threat sensitivity. A high-Neuroticism NPC (N≥0.65) perceives threats more readily. A high-Agreeableness NPC (A≥0.65) is less likely to interpret neutral messages as threatening.

Return ONLY this JSON object — no prose, no markdown fences:
{{"is_threat": <true|false>, "threat_magnitude": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""

    raw = llm.generate(prompt, max_tokens=THREAT_MAX_TOKENS)

    # Step 4 — parse response
    import json
    import re

    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        data = json.loads(cleaned)
        is_threat: bool = bool(data.get("is_threat", False))
        threat_magnitude: float = float(data.get("threat_magnitude", 0.0))
        reasoning: str = str(data.get("reasoning", ""))
    except Exception:  # noqa: BLE001
        return _fallback

    # Step 5 — determine response mode
    if is_threat:
        response_mode = "fight_flight"
    elif past_memories:
        response_mode = "standard"
    else:
        response_mode = "instinct"

    return ThreatAssessment(
        is_threat=is_threat,
        threat_magnitude=threat_magnitude,
        reasoning=reasoning,
        response_mode=response_mode,
    )

"""
Stage 1 — Threat Assessment (paper §3.1)

Quick amygdala-analogue pass: embeds the player input, pulls top-3 memories
by raw cosine for LLM context, and asks the LLM whether the input represents
a threat to this NPC. Returns only the threat judgement; response-mode
selection happens downstream once retrieval scores are known (paper §3.3).
"""

from __future__ import annotations

import json
import re

from ..config import THREAT_MAX_TOKENS
from ..llm.client import GeminiClient
from ..memory.manager import MemoryManager
from ..models import OCEANProfile, ThreatAssessment


def assess_threat(
    player_input: str,
    query_embedding: list[float],
    profile: OCEANProfile,
    memory_manager: MemoryManager,
    llm: GeminiClient,
) -> ThreatAssessment:
    """Return a ThreatAssessment for *player_input*.

    The caller is expected to supply *query_embedding* (computed once per
    turn so threat assessment, scored retrieval, and consolidation share a
    single embed call).
    """
    if not query_embedding:
        return ThreatAssessment(
            is_threat=False,
            threat_magnitude=0.0,
            reasoning="Embedding failed",
        )

    # Personality-weighted top-3 (no threshold) — the LLM sees the memories
    # this agent's tag/trait profile makes salient, not pure-cosine matches.
    past_memories = memory_manager.retrieve_top_scored(query_embedding, top_k=3)

    effective = profile.effective
    if past_memories:
        context_block = "Past context memories:\n" + "\n".join(
            f"- {m.text}" for m in past_memories
        )
    else:
        context_block = "Past context memories:\n(none)"

    prompt = f"""You are assessing whether a player's message represents a threat to an NPC.

NPC personality: {profile.describe()}
Effective OCEAN values — O:{effective['O']:.2f} C:{effective['C']:.2f} E:{effective['E']:.2f} A:{effective['A']:.2f} N:{effective['N']:.2f}

{context_block}

Player message: {player_input}

Consider the NPC's personality when judging threat sensitivity. A high-Neuroticism NPC (N≥0.65) perceives threats more readily. A high-Agreeableness NPC (A≥0.65) is less likely to interpret neutral messages as threatening.

Return ONLY this JSON object — no prose, no markdown fences:
{{"is_threat": <true|false>, "threat_magnitude": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""

    raw = llm.generate(prompt, max_tokens=THREAT_MAX_TOKENS)

    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        data = json.loads(cleaned)
        return ThreatAssessment(
            is_threat=bool(data.get("is_threat", False)),
            threat_magnitude=float(data.get("threat_magnitude", 0.0)),
            reasoning=str(data.get("reasoning", "")),
        )
    except Exception:  # noqa: BLE001
        return ThreatAssessment(
            is_threat=False,
            threat_magnitude=0.0,
            reasoning="Assessment parse failed",
        )

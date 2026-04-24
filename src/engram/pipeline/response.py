"""
Stage 5 — LLM Dialogue Generation

Builds a structured prompt from pipeline state and generates the NPC's
in-character response.  Personality (OCEAN) drives both the behavioural
instructions injected into the prompt and, upstream, which memories were
retrieved — so personality shapes output without being hardcoded to names.
"""

from __future__ import annotations

from ..llm.client import GeminiClient
from ..models import Memory, NPCConfig, OCEANProfile, ThreatAssessment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trait_instruction(trait: str, value: float) -> str:
    """Return a one-line behavioural instruction for a single OCEAN trait."""
    high = value >= 0.65
    low = value <= 0.35

    if trait == "O":
        if high:
            return "Openness: Embrace new ideas, update beliefs readily."
        if low:
            return "Openness: Resist change, stick to what you know."
        return "Openness: Balanced curiosity — open to ideas that fit your worldview."

    if trait == "C":
        if high:
            return "Conscientiousness: Goal-focused, structured, deliberate."
        if low:
            return "Conscientiousness: Spontaneous, go with the flow."
        return "Conscientiousness: Moderately organised — follow rules when it matters."

    if trait == "E":
        if high:
            return "Extraversion: Social, talkative, warm."
        if low:
            return "Extraversion: Reserved, brief, guarded."
        return "Extraversion: Selectively social — engage when you have reason to."

    if trait == "A":
        if high:
            return "Agreeableness: Cooperative, trusting, helpful."
        if low:
            return "Agreeableness: Suspicious, self-serving, competitive."
        return "Agreeableness: Conditionally cooperative — help when it benefits you too."

    if trait == "N":
        if high:
            return "Neuroticism: Anxious, threat-aware, emotionally reactive."
        if low:
            return "Neuroticism: Calm, stable, hard to shake."
        return "Neuroticism: Moderate emotional sensitivity — concerned but not panicked."

    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_response(
    player_input: str,
    config: NPCConfig,
    profile: OCEANProfile,
    retrieved: list[Memory],
    assessment: ThreatAssessment,
    history: list[dict],   # list of {"player": str, "npc": str}
    llm: GeminiClient,
) -> str:
    """
    Generate the NPC's dialogue response for *player_input*.

    Parameters
    ----------
    player_input:
        The raw text the player just sent.
    config:
        Static NPC configuration (persona, name, etc.).
    profile:
        Current OCEAN profile (effective values, including any fight/flight
        deltas, are used for per-trait instructions).
    retrieved:
        Memories surfaced by Stage 2.
    assessment:
        Threat assessment from Stage 1.
    history:
        Recent conversation turns; last 4 are included in the prompt.
    llm:
        Configured GeminiClient.

    Returns
    -------
    Raw LLM response string.
    """
    mode = assessment.response_mode
    effective = profile.effective

    # ------------------------------------------------------------------ TASK
    threat_status = (
        f"THREAT DETECTED (magnitude={assessment.threat_magnitude:.2f}: "
        f"{assessment.reasoning})"
        if assessment.is_threat
        else "No threat detected"
    )
    task_section = (
        "=== TASK ===\n"
        f"Response mode: {mode.upper()}\n"
        f"Threat status: {threat_status}"
    )

    # --------------------------------------------------------- CURRENT INPUT
    input_section = f"=== CURRENT INPUT ===\n{player_input}"

    # ------------------------------------------------- CONVERSATION HISTORY
    recent_turns = history[-4:] if len(history) >= 4 else history
    if recent_turns:
        history_lines = []
        for turn in recent_turns:
            history_lines.append(f"Player: {turn.get('player', '')}")
            history_lines.append(f"{config.name}: {turn.get('npc', '')}")
        history_block = "\n".join(history_lines)
    else:
        history_block = "(no prior conversation)"
    history_section = f"=== CONVERSATION HISTORY ===\n{history_block}"

    # ----------------------------------------------- CHARACTER BACKGROUND
    background_section = (
        f"=== CHARACTER BACKGROUND ===\n"
        f"Name: {config.name}\n"
        f"{config.persona}"
    )

    # --------------------------------------------------- RELEVANT MEMORIES
    if retrieved:
        memory_lines = "\n".join(f"- {m.text}" for m in retrieved)
    else:
        memory_lines = "No relevant memories."
    memory_section = f"=== RELEVANT MEMORIES ===\n{memory_lines}"

    # ---------------------------------------------------------- PERSONALITY
    trait_instructions = "\n".join(
        _trait_instruction(t, effective[t])
        for t in ("O", "C", "E", "A", "N")
    )
    personality_section = (
        f"=== PERSONALITY ===\n"
        f"Profile: {profile.describe()}\n\n"
        f"Per-trait behavioural instructions:\n"
        f"{trait_instructions}"
    )

    # -------------------------------------------- MODE-SPECIFIC ADDENDUM
    mode_addendum = ""
    if mode == "fight_flight":
        mode_addendum = (
            "\n=== FIGHT-OR-FLIGHT MODE ===\n"
            "You are in fight-or-flight state. "
            "Your response should be short, reactive, and emotionally heightened."
        )
    elif mode == "instinct":
        mode_addendum = (
            "\n=== INSTINCT MODE ===\n"
            "You have no clear relevant memories. "
            "Respond from instinct based on your personality alone."
        )

    # -------------------------------------------------------------- RULES
    rules_section = (
        "=== RULES ===\n"
        "- Stay fully in character at all times.\n"
        "- Respond in 2-4 sentences maximum.\n"
        "- Do not mention game mechanics, memory systems, or personality scores."
    )

    # -------------------------------------------------------- FULL PROMPT
    prompt = "\n\n".join([
        task_section,
        input_section,
        history_section,
        background_section,
        memory_section,
        personality_section,
    ])

    if mode_addendum:
        prompt += "\n" + mode_addendum

    prompt += "\n\n" + rules_section
    prompt += f"\n\n{config.name}:"

    return llm.generate(prompt)

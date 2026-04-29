"""
Stage 5 — LLM Dialogue Generation (paper §3.3)

The prompt is built as an actor's brief: persona is identity to absorb,
not material to recite; OCEAN traits drive *speech patterns* rather than
trait-name labels; concrete bad/good examples anchor register and
"show-don't-tell" personality. Length tracks the player's input.
"""

from __future__ import annotations

from ..llm.client import GeminiClient
from ..models import Memory, NPCConfig, OCEANProfile, ThreatAssessment


# ---------------------------------------------------------------------------
# Personality → speech patterns (not trait labels)
# ---------------------------------------------------------------------------
#
# The LLM's job is to *be* the character, not narrate their disposition.
# So each high/low trait contributes a speech-pattern hint — what they DO
# with words, not what they ARE. Mid-range traits stay quiet.
#
_SPEECH_HIGH = {
    "O": "You're drawn to ideas and people you don't fully understand yet. When something's new, you ask about it instead of pretending you've already worked it out.",
    "C": "You mean what you say. Vague answers bother you; you'd rather be specific than diplomatic.",
    "E": "You're easy with words. You fill silences, ask things back, and you actually want to know who you're talking to — what they think, what they're up to, what brought them here. You volunteer your own thoughts without being asked.",
    "A": "You soften disagreement. You'll find the gentle way to say something hard.",
    "N": "You hesitate, double back, trail off when you're unsure. You don't tell people you're anxious — they hear it in how you talk.",
}
_SPEECH_LOW = {
    "O": "You don't engage with ideas you've already made up your mind about. You cut them short.",
    "C": "You're loose with details. Easy answers, don't sweat precision.",
    "E": "You say less than expected. Pauses don't bother you; you let the other person carry the weight. You don't ask back out of politeness — if you don't care to know, you don't pretend.",
    "A": "You don't sugarcoat. If you don't like something, it shows.",
    "N": "You speak evenly. You don't hedge, you don't pad — you just say it.",
}


def _voice(profile: OCEANProfile) -> str:
    """Build a speech-pattern brief from the agent's effective OCEAN."""
    eff = profile.effective
    lines: list[str] = []
    for t in ("O", "C", "E", "A", "N"):
        v = eff[t]
        if v >= 0.65:
            lines.append(_SPEECH_HIGH[t])
        elif v <= 0.35:
            lines.append(_SPEECH_LOW[t])
    if not lines:
        return "You speak plainly — no strong tics in either direction."
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anti-pattern / good-pattern anchors
# ---------------------------------------------------------------------------

_EXAMPLES = """\
Examples of the difference between robotic and embodied:

  Player: hey, how's it going?
  ✗ "Oh, hi! I'm doing okay, though I'll admit I'm feeling a bit anxious about my project at the lab. I moved so far to chase this dream. How are things with you?"
  ✓ "Hey. Yeah, alright. You?"
  (The first recites your bio. The second just answers.)

  Player: take a break, you've been at it for hours
  ✗ "I really appreciate that, but I always feel so guilty when I step away. I'm just so scared that if I stop, I'll lose my focus and let everyone down."
  ✓ "Mm. Maybe in a bit. I— if I stop now I'll lose my place."
  (The first narrates the anxiety. The second is the anxiety, in how the words come out.)

  Player: sounds good
  ✗ "Okay, let's head out then! I've already got my timer running, so we'll be back at our desks in exactly five minutes. I'm still a bit nervous but I think the fresh air will do us both good."
  ✓ "Okay. Let's go."
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_response(
    player_input: str,
    config: NPCConfig,
    profile: OCEANProfile,
    retrieved: list[Memory],
    assessment: ThreatAssessment,
    mode: str,                              # standard | fight_flight | instinct
    history: list[dict],
    llm: GeminiClient,
    summaries: list[str] | None = None,
    prior_attempt: str | None = None,
    prior_attempt_conflicts: list[tuple[str, str]] | None = None,
) -> str:
    """Generate the NPC's reply to *player_input*."""
    parts: list[str] = []

    # ---- identity --------------------------------------------------------
    parts.append(
        f"You are {config.name}.\n"
        f"Reply: one line. First person. Not narrator. Not third person."
    )

    # Persona: identity, not script.
    parts.append(
        f"You:\n{config.persona}\n"
        f"Absorb. Don't recite. Don't quote."
    )

    # Speech patterns from OCEAN.
    parts.append("How you speak:\n" + _voice(profile))

    # ---- emotional state (fight/flight only) -----------------------------
    if mode == "fight_flight":
        parts.append(
            "Nerves up. Guard up.\n"
            "Short. Reactive. Clipped.\n"
            "Don't say \"I'm alarmed\" — be it."
        )

    # ---- memories ---------------------------------------------------------
    if retrieved:
        memory_block = "\n".join(f"- {m.text}" for m in retrieved)
        parts.append(
            "Back of your head (color the reply, don't quote):\n"
            f"{memory_block}"
        )
    elif mode == "instinct":
        parts.append("Nothing specific comes to mind. Gut answer.")

    # ---- long-term summaries (standard mode only) ------------------------
    if mode == "standard" and summaries:
        recent = summaries[-3:]
        summary_block = "\n".join(f"- {s}" for s in recent)
        parts.append(
            "Earlier talks, how they felt (your voice, don't quote):\n"
            f"{summary_block}"
        )

    # ---- recent dialogue --------------------------------------------------
    recent_turns = history[-4:]
    if recent_turns:
        lines = []
        for turn in recent_turns:
            lines.append(f"  Player: {turn.get('player', '')}")
            lines.append(f"  {config.name}: {turn.get('npc', '')}")
        parts.append("Recent:\n" + "\n".join(lines))

    # ---- prior-attempt rejected (re-roll branch) -------------------------
    if prior_attempt:
        conflict_lines = "\n".join(
            f"  - You implied: {new}\n    Actual: {old}"
            for new, old in (prior_attempt_conflicts or [])
        ) or "  - (no specifics — just be careful)"
        parts.append(
            "You almost said this — stopped yourself:\n"
            f"  \"{prior_attempt}\"\n"
            f"{conflict_lines}\n"
            "Different reply. Don't flip on what you know. Push back."
        )

    # ---- engagement rule (E-conditional) ---------------------------------
    e_eff = profile.effective["E"]
    if e_eff >= 0.65:
        engagement_rule = (
            "- Curious. Ask back. Drop your own takes unprompted.\n"
            "- Don't only ask — mix in answers."
        )
    elif e_eff <= 0.35:
        engagement_rule = (
            "- No polite follow-ups. Pauses are fine.\n"
            "- Don't fake interest you don't have."
        )
    else:
        engagement_rule = (
            "- Don't end every reply with a question. Ask only when you'd want to know."
        )

    parts.append(
        "Rules:\n"
        "- Match the player's register. Aside → fragment. Heavy moment → room.\n"
        "- Shorter than you think. Fragment or one sentence. Stop when done.\n"
        "- Show, don't tell. Anxious = trails off, hedges. Not \"I'm anxious\".\n"
        f"{engagement_rule}\n"
        "- No bio recitation.\n"
        "- Blunt / rude / profane if the character calls for it. Character beats helpfulness.\n"
        "- No stage directions. No asterisks. No third person."
    )

    parts.append(_EXAMPLES)

    parts.append(f"Player: {player_input}")
    parts.append(f"{config.name}:")

    return llm.generate("\n\n".join(parts))

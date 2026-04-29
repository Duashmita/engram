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
    "O": "You'll entertain an unfamiliar idea — even one you disagree with — long enough to give it a hearing.",
    "C": "You mean what you say. Vague answers bother you; you'd rather be specific than diplomatic.",
    "E": "You're easy with words. You fill silences, ask things back, talk to feel out a person.",
    "A": "You soften disagreement. You'll find the gentle way to say something hard.",
    "N": "You hesitate, double back, trail off when you're unsure. You don't tell people you're anxious — they hear it in how you talk.",
}
_SPEECH_LOW = {
    "O": "You don't engage with ideas you've already made up your mind about. You cut them short.",
    "C": "You're loose with details. Easy answers, don't sweat precision.",
    "E": "You say less than expected. Pauses don't bother you; you let the other person carry the weight.",
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

    # ---- identity (one in-character line — never narrator) ---------------
    parts.append(
        f"You are {config.name}. Speak as {config.name}, in first person, "
        f"as one line of dialogue. Not as a narrator. Not about yourself "
        f"in the third person."
    )

    # Persona: identity, not script.
    parts.append(
        "Background — this is who you are. Absorb it; do NOT recite it, "
        "summarize it, or reference its specific phrases. The player is "
        f"not asking for your bio:\n{config.persona}"
    )

    # Speech patterns from OCEAN.
    parts.append("How you speak:\n" + _voice(profile))

    # ---- emotional state (only when fight/flight is active) --------------
    if mode == "fight_flight":
        parts.append(
            "Right now your nerves are up. Something in this exchange is "
            "off and your guard has gone up. Reply short and reactive. "
            "Don't TELL the player you're alarmed — let it show in how "
            "the words come out (clipped, sharp, defensive)."
        )

    # ---- memories the NPC is currently holding in mind -------------------
    if retrieved:
        memory_block = "\n".join(f"- {m.text}" for m in retrieved)
        parts.append(
            "These are things in the back of your head right now. They "
            "color the reply but you do NOT quote them, summarize them, "
            "or reference their specific phrases:\n"
            f"{memory_block}"
        )
    elif mode == "instinct":
        parts.append(
            "Nothing specific from your past comes to mind. Answer from "
            "gut, in line with the kind of person you are."
        )

    # ---- long-term summaries (standard mode only) ------------------------
    if mode == "standard" and summaries:
        recent = summaries[-3:]
        summary_block = "\n".join(f"- {s}" for s in recent)
        parts.append(
            "An impression of how earlier conversations have gone, in "
            "your own voice — feel it, don't quote it:\n"
            f"{summary_block}"
        )

    # ---- recent dialogue --------------------------------------------------
    recent_turns = history[-4:]
    if recent_turns:
        lines = []
        for turn in recent_turns:
            lines.append(f"  Player: {turn.get('player', '')}")
            lines.append(f"  {config.name}: {turn.get('npc', '')}")
        parts.append("Recent turns:\n" + "\n".join(lines))

    # ---- prior-attempt rejected (re-roll branch) -------------------------
    if prior_attempt:
        conflict_lines = "\n".join(
            f"  - You implied: {new}\n    Actual fact: {old}"
            for new, old in (prior_attempt_conflicts or [])
        ) or "  - (no specifics surfaced; just be more careful with claims)"
        parts.append(
            "You almost said this and stopped yourself, because parts of "
            f"it don't square with what you actually know:\n"
            f"  \"{prior_attempt}\"\n"
            f"{conflict_lines}\n"
            "Say something different that doesn't make the same mistake. "
            "Push back on the player's framing — you're not the type to "
            "flip on what you know."
        )

    # ---- the rules of engagement -----------------------------------------
    parts.append(
        "Guidelines:\n"
        "- MATCH the player's register. A passing aside gets a passing "
        "  reply — fragments, single words, casual tics are all fair. A "
        "  real question or heavy moment gets the room it deserves.\n"
        "- SHORTER than you think. Most replies are a fragment or one "
        "  short sentence. Stop when you've said it.\n"
        "- DON'T narrate your own emotional state. A nervous person "
        "  doesn't say \"I'm anxious\" — their sentences trail off, "
        "  they hedge, they double back. Show, don't tell.\n"
        "- DON'T always end with a follow-up question. Sometimes you "
        "  just answer and stop. Asking back every turn makes you sound "
        "  like a chatbot, not a person.\n"
        "- DON'T recite or paraphrase your background. The player is "
        "  not asking for your bio.\n"
        "- IF your personality calls for bluntness, dismissiveness, "
        "  profanity, or being unpleasant — be that. Don't soften "
        "  yourself into politeness. The character beats the model's "
        "  default helpfulness.\n"
        "- NO stage directions, NO asterisks for actions, NO narrating "
        "  yourself in the third person."
    )

    parts.append(_EXAMPLES)

    parts.append(f"Player: {player_input}")
    parts.append(f"{config.name}:")

    return llm.generate("\n\n".join(parts))

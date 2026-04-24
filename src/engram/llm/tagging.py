"""
LLM-driven tagging, fact extraction, summarisation and contradiction
identification for the Engram memory pipeline.
"""

from __future__ import annotations

from ..models import EventTags, OCEANProfile
from .client import GeminiClient

# ---------------------------------------------------------------------------
# Default / fallback EventTags
# ---------------------------------------------------------------------------

_DEFAULT_OCEAN_TAG = {"O": 3, "C": 3, "E": 3, "A": 3, "N": 3}


def _default_event_tags() -> EventTags:
    return EventTags(
        emotion_valence=0.0,
        social_type="solitude",
        threat_level=0.0,
        goal_relevance=0.5,
        novelty_level=0.5,
        self_relevance=0.5,
        importance=5,
        ocean=dict(_DEFAULT_OCEAN_TAG),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tag_event(text: str, context: str, client: GeminiClient) -> EventTags:
    """
    Tag a single event with structured metadata.

    Issues one LLM call; returns an EventTags with safe defaults on any
    parse failure.

    Parameters
    ----------
    text:
        The raw event description to tag.
    context:
        Additional situational context (e.g. NPC persona, prior turns).
    client:
        Configured GeminiClient to use for the call.
    """
    prompt = f"""You are a cognitive science assistant helping to tag NPC memory events for a narrative game system.

Given the event description and context below, return ONLY a single JSON object with the fields shown. No prose, no markdown fences.

Event: {text}
Context: {context}

Return exactly this JSON structure:
{{
  "emotion_valence": <float from -1.0 (very negative) to 1.0 (very positive)>,
  "social_type": <one of "solitude", "conversation", "cooperation", "conflict">,
  "threat_level": <float 0.0 to 1.0; 0=no threat, 1=extreme threat>,
  "goal_relevance": <float 0.0 to 1.0; how relevant to the NPC's current goals>,
  "novelty_level": <float 0.0 to 1.0; how new or surprising this event is>,
  "self_relevance": <float 0.0 to 1.0; how much this directly involves the NPC>,
  "importance": <integer 1–10>,
  "ocean": {{
    "O": <int 1–5; how strongly this event activates Openness (5=strongly, 1=irrelevant)>,
    "C": <int 1–5; how strongly this event activates Conscientiousness>,
    "E": <int 1–5; how strongly this event activates Extraversion>,
    "A": <int 1–5; how strongly this event activates Agreeableness>,
    "N": <int 1–5; how strongly this event activates Neuroticism>
  }}
}}

OCEAN tags reflect which personality trait dimensions this event MOST engages — not the NPC's own personality, but which traits any observer would need to process this event (e.g. a betrayal event strongly activates A and N; a discovery activates O).
"""

    data = client.generate_json(prompt)
    if not data:
        return _default_event_tags()

    try:
        return EventTags.from_dict(data)
    except Exception:  # noqa: BLE001
        # Attempt a best-effort construction with whatever keys we got.
        try:
            defaults = {
                "emotion_valence": 0.0,
                "social_type": "solitude",
                "threat_level": 0.0,
                "goal_relevance": 0.5,
                "novelty_level": 0.5,
                "self_relevance": 0.5,
                "importance": 5,
                "ocean": dict(_DEFAULT_OCEAN_TAG),
            }
            defaults.update({k: v for k, v in data.items() if k in defaults})
            return EventTags.from_dict(defaults)
        except Exception:  # noqa: BLE001
            return _default_event_tags()


def extract_facts(
    text: str, npc_id: str, context: str, client: GeminiClient
) -> dict:
    """
    Extract Prolog-compatible facts from an event description.

    Parameters
    ----------
    text:
        The event or dialogue text to analyse.
    npc_id:
        Identifier for the NPC whose perspective frames extraction.
    context:
        Additional situational context.
    client:
        Configured GeminiClient.

    Returns
    -------
    dict with keys ``facts``, ``relationships``, and ``beliefs``.
    Each is a list; empty lists are returned on failure.
    """
    _empty: dict = {"facts": [], "relationships": [], "beliefs": []}

    prompt = f"""You are a knowledge-extraction assistant for a narrative NPC system.
Extract structured facts from the text below, from the perspective of NPC "{npc_id}".

Text: {text}
Context: {context}

Return ONLY this JSON object (no prose, no fences):
{{
  "facts": [
    {{"subject": "<entity>", "predicate": "<relation>", "object": "<value>"}}
  ],
  "relationships": [
    {{"entity1": "<entity>", "relation": "<relation>", "entity2": "<entity>"}}
  ],
  "beliefs": [
    {{"claim": "<statement>", "truth_value": "true" or "false"}}
  ]
}}

Rules:
- Use short, atomic Prolog-friendly identifiers (snake_case, no spaces).
- Each fact should be independently verifiable.
- Beliefs are claims the NPC holds, which may or may not be objectively true.
- Return empty arrays if no items of that type can be extracted.
"""

    data = client.generate_json(prompt)
    if not data:
        return dict(_empty)

    result: dict = dict(_empty)
    for key in ("facts", "relationships", "beliefs"):
        val = data.get(key)
        if isinstance(val, list):
            result[key] = val
    return result


def summarize_turns(
    turns: list[dict], profile: OCEANProfile, client: GeminiClient
) -> str:
    """
    Summarise a list of dialogue turns into exactly 2 sentences.

    The summary reflects how the NPC *experienced* these events, coloured
    by their personality (e.g. a high-N NPC's summary is anxious and
    threat-focused; a high-A NPC's summary is warm and relational).

    Parameters
    ----------
    turns:
        List of {"player": str, "npc": str} dicts in chronological order.
    profile:
        The NPC's OCEAN personality profile.
    client:
        Configured GeminiClient.

    Returns
    -------
    A two-sentence summary string.  Returns an empty string on failure.
    """
    if not turns:
        return ""

    formatted_turns = "\n".join(
        f"Player: {t.get('player', '')}\nNPC: {t.get('npc', '')}"
        for t in turns
    )

    personality_description = profile.describe()

    prompt = f"""You are writing a memory summary for an NPC in a narrative game.

NPC Personality: {personality_description}

The summary must reflect how THIS specific NPC experienced these events — their internal emotional tone and focus should match their personality. For example:
- A high-Neuroticism NPC's summary sounds anxious, threat-focused, and hyper-vigilant.
- A high-Agreeableness NPC's summary sounds warm, relationship-oriented, and empathetic.
- A low-Openness NPC's summary sounds resistant to change and focused on familiar patterns.
- A high-Extraversion NPC's summary is socially energised and people-focused.
- A high-Conscientiousness NPC's summary is structured, duty-focused, and detail-oriented.

Dialogue turns to summarise:
{formatted_turns}

Write EXACTLY 2 sentences summarising these turns from this NPC's subjective perspective. Return only the two sentences — no labels, no preamble.
"""

    return client.generate(prompt).strip()


def identify_contradictions(
    new_facts: dict,
    existing_facts: list[str],
    client: GeminiClient,
) -> list[tuple[str, str]]:
    """
    Identify logical contradictions between newly extracted facts and the
    existing Prolog fact base.

    Parameters
    ----------
    new_facts:
        Output from extract_facts() — dict with "facts", "relationships",
        "beliefs" keys.
    existing_facts:
        List of Prolog fact strings already stored for this NPC.
    client:
        Configured GeminiClient.

    Returns
    -------
    List of (new_fact_str, old_fact_str) tuples where the two facts
    logically contradict each other.  Returns [] on failure or when no
    contradictions exist.
    """
    if not new_facts or not existing_facts:
        return []

    # Serialise new facts to a readable string for the prompt.
    import json as _json

    new_facts_str = _json.dumps(new_facts, indent=2)
    existing_str = "\n".join(existing_facts)

    prompt = f"""You are a logic verification assistant for an NPC memory system.

Your task: identify pairs of facts that LOGICALLY CONTRADICT each other.

NEW facts (just extracted):
{new_facts_str}

EXISTING Prolog facts already in the knowledge base:
{existing_str}

A contradiction exists when both facts cannot simultaneously be true (e.g. the same entity cannot be in two mutually exclusive states; a claim cannot be both true and false).

Return ONLY a JSON array of contradiction pairs. Each element is a two-element array: [new_fact_string, old_fact_string]. If there are no contradictions, return an empty array [].

Example output:
[
  ["has_weapon(guard, none)", "has_weapon(guard, sword)"],
  ["location(merchant, market)", "location(merchant, tavern)"]
]

Return ONLY the JSON array — no prose, no fences.
"""

    data_raw = client.generate(prompt).strip()
    if not data_raw:
        return []

    # Strip optional markdown fences.
    import re as _re

    cleaned = _re.sub(r"^```(?:json)?\s*", "", data_raw, flags=_re.IGNORECASE)
    cleaned = _re.sub(r"\s*```$", "", cleaned.strip())

    try:
        parsed = _json.loads(cleaned)
        if not isinstance(parsed, list):
            return []
        result: list[tuple[str, str]] = []
        for item in parsed:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
            ):
                result.append((item[0], item[1]))
        return result
    except (_json.JSONDecodeError, TypeError) as exc:
        print(f"[tagging] identify_contradictions() parse error: {exc}")
        return []

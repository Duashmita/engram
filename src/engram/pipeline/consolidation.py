"""
Stage 4 + Stage 6 — Prolog contradiction checking & memory consolidation.

Three entry points:

check_contradictions()
    Stage 4 (paper §3.4) — runs before response generation. Extracts
    candidate facts from the player input, checks each against the NPC's
    Prolog keystore, and returns the list of (new_fact, old_fact) pairs
    so the response prompt can flag the contradiction to the LLM.

consolidate()
    Stage 6a — called after every NPC turn. Encodes the exchange as a
    Memory, stores it in the session window, and triggers the
    MemoryManager's eviction / summary logic when the window fills.

post_session_fact_check()
    Stage 6b — called at session end. Reconciles new facts with the
    Prolog key-memory base, gated by Openness: high-O NPCs accept belief
    revision; low-O NPCs reject it.
"""

from __future__ import annotations

import re
import time

from ..llm.client import GeminiClient
from ..llm.tagging import extract_facts, tag_event
from ..memory.manager import MemoryManager
from ..models import Memory, NPCConfig, OCEANProfile
from ..observability import bus


# ---------------------------------------------------------------------------
# Stage 4 — pre-response contradiction check (paper §3.4)
# ---------------------------------------------------------------------------

def check_contradictions(
    text: str,
    config: NPCConfig,
    keystore,
    llm: GeminiClient,
    _emit_stage: str = "generic",
) -> list[tuple[str, str]]:
    """Detect contradictions between facts extracted from *text* and the
    NPC's Prolog fact base.

    Used for both:
    - **Pre-generation** check on the player's input (so the response prompt
      can flag a player's contradictory claim and let the NPC push back).
    - **Post-generation** check on the NPC's own response (so a hallucinated
      claim that conflicts with the fact base triggers a re-roll, paper §3.4).

    Returns ``(new_fact, conflicting_old_fact)`` tuples; empty list = clean.

    The ``_emit_stage`` kwarg lets the orchestrator label the emitted
    observability event ("pre" | "post" | "generic"); it has no effect on
    the returned value.
    """
    extracted = extract_facts(text, config.npc_id, config.persona, llm)
    candidates = _to_prolog_strings(extracted)

    contradictions: list[tuple[str, str]] = []
    for fact_str in candidates:
        found, old_fact = keystore.check_contradiction(fact_str)
        if found:
            contradictions.append((fact_str, old_fact))

    bus.emit(
        "contradiction_check",
        stage=_emit_stage,
        text=text[:200],
        conflicts=[{"new": new, "old": old} for new, old in contradictions],
    )
    return contradictions


# ---------------------------------------------------------------------------
# Stage 6a — per-turn consolidation
# ---------------------------------------------------------------------------

def consolidate(
    player_input: str,
    npc_response: str,
    config: NPCConfig,
    profile: OCEANProfile,
    memory_manager: MemoryManager,
    llm: GeminiClient,
    memory_id: str,
) -> Memory:
    """
    Encode one dialogue exchange as a Memory and persist it.

    Parameters
    ----------
    player_input:
        Raw player message.
    npc_response:
        NPC's generated reply.
    config:
        Static NPC configuration.
    profile:
        Current OCEAN personality profile.
    memory_manager:
        Memory store to write into.
    llm:
        Configured GeminiClient.
    memory_id:
        Unique identifier for this memory (caller-generated, e.g. UUID).

    Returns
    -------
    The newly created and stored Memory object.
    """
    # Step 1 — combine into a single text representation
    combined_text = f"Player: {player_input} | {config.name}: {npc_response}"

    # Step 2 — embed the combined text
    embedding = llm.embed(combined_text)

    # Step 3 — tag the combined text
    tags = tag_event(combined_text, f"{config.name} interaction", llm)

    # Step 4 — create Memory
    memory = Memory(
        id=memory_id,
        text=combined_text,
        tags=tags,
        embedding=embedding,
        source="session",
        timestamp=time.time(),
    )

    # Step 5 — store memory
    memory_manager.add_memory(memory)

    # Step 6 — record the raw turn (drives session window + eviction / summary)
    memory_manager.add_turn(player_input, npc_response)

    return memory


# ---------------------------------------------------------------------------
# Stage 6b — post-session Prolog fact reconciliation
# ---------------------------------------------------------------------------

def post_session_fact_check(
    recent_memories: list[Memory],
    config: NPCConfig,
    profile: OCEANProfile,
    keystore,   # KeyStore instance (duck-typed to avoid circular imports)
    llm: GeminiClient,
) -> None:
    """
    Reconcile newly extracted facts with the NPC's Prolog key-memory base.

    For each recent memory:
    - Extract facts via LLM.
    - For each fact, convert to a Prolog string and check for contradictions.
    - If a contradiction exists:
        - High-O NPC (O ≥ 0.5): retract old fact, assert new fact.
        - Low-O NPC  (O < 0.5): skip — low-Openness NPCs resist belief revision.
    - If no contradiction: assert the new fact unconditionally.

    Parameters
    ----------
    recent_memories:
        Memories from the just-completed session to process.
    config:
        Static NPC configuration (npc_id, persona used for extraction context).
    profile:
        Current OCEAN personality profile; Openness governs belief revision.
    keystore:
        KeyStore instance exposing check_contradiction(), assert_fact(),
        and retract_fact() methods.
    llm:
        Configured GeminiClient.
    """
    for memory in recent_memories:
        extracted = extract_facts(
            memory.text,
            config.npc_id,
            config.persona,
            llm,
        )

        # Flatten all extracted items into (prolog_string, raw_item) pairs
        fact_strings: list[str] = _to_prolog_strings(extracted)

        for fact_str in fact_strings:
            found, old_fact = keystore.check_contradiction(fact_str)

            if found:
                if profile.O >= 0.5:
                    # High Openness — accept the new belief
                    keystore.retract_fact(old_fact)
                    keystore.assert_fact(fact_str)
                    bus.emit("fact_revised", fact=fact_str, old=old_fact)
                else:
                    # Low Openness — reject new belief silently
                    bus.emit("fact_rejected", fact=fact_str, old=old_fact)
            else:
                keystore.assert_fact(fact_str)
                bus.emit("fact_asserted", fact=fact_str)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_prolog_strings(extracted: dict) -> list[str]:
    """
    Convert the output of extract_facts() into Prolog-compatible fact strings.

    Handles the three keys produced by extract_facts:
    - facts        → ``predicate(subject, object).``
    - relationships → ``relation(entity1, entity2).``
    - beliefs       → ``believes(claim, truth_value).``

    Invalid / incomplete entries are skipped.
    """
    result: list[str] = []

    for item in extracted.get("facts", []):
        try:
            s = _slugify(item["subject"])
            p = _slugify(item["predicate"])
            o = _slugify(item["object"])
            result.append(f"{p}({s}, {o}).")
        except (KeyError, TypeError):
            continue

    for item in extracted.get("relationships", []):
        try:
            e1 = _slugify(item["entity1"])
            rel = _slugify(item["relation"])
            e2 = _slugify(item["entity2"])
            result.append(f"{rel}({e1}, {e2}).")
        except (KeyError, TypeError):
            continue

    for item in extracted.get("beliefs", []):
        try:
            claim = _slugify(item["claim"])
            truth = str(item.get("truth_value", "true")).lower()
            result.append(f"believes({claim}, {truth}).")
        except (KeyError, TypeError):
            continue

    return result


def _slugify(value: str) -> str:
    """Normalise a string to a safe Prolog atom.

    Prolog atoms must start with a lowercase letter or underscore; digits at
    the start need quoting, so we prefix those with 'a_'.
    """
    value = re.sub(r"[^a-z0-9_]+", "_", str(value).lower().strip()).strip("_")
    if not value:
        return "unknown"
    if value[0].isdigit():
        value = "a_" + value
    return value

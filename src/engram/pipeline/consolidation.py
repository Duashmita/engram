"""
Stage 6 — Memory Consolidation

Two entry points:

consolidate()
    Called after every NPC turn.  Encodes the exchange as a Memory, stores it
    in the session window, and triggers the MemoryManager's eviction / summary
    logic when the window fills.

post_session_fact_check()
    Called at session end.  Extracts Prolog facts from recent memories and
    reconciles them with the NPC's key-memory fact base, gated by Openness:
    high-O NPCs accept belief revision; low-O NPCs reject it.
"""

from __future__ import annotations

import time

from ..llm.client import GeminiClient
from ..llm.tagging import extract_facts, tag_event
from ..memory.manager import MemoryManager
from ..models import Memory, NPCConfig, OCEANProfile


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
            contradiction = keystore.check_contradiction(fact_str)

            if contradiction:
                # contradiction is the conflicting existing fact string
                old_fact = contradiction
                if profile.O >= 0.5:
                    # High Openness — accept the new belief
                    keystore.retract_fact(old_fact)
                    keystore.assert_fact(fact_str)
                # else: Low Openness — reject new belief silently
            else:
                keystore.assert_fact(fact_str)


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
    """
    Normalise a string to a safe Prolog atom: lowercase, spaces → underscores,
    non-alphanumeric characters stripped.
    """
    import re
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = value.strip("_")
    return value or "unknown"

"""
Stage 2 — Personality-Weighted Memory Retrieval

Selects which memories to surface for response generation based on the
threat assessment's response mode.

- instinct mode  : no strong embedding match → tag-vector cosine retrieval
  (6D EventTags space; top-3)
- standard / fight_flight : scored embedding retrieval filtered to score ≥ 15
  (top-5, handled inside MemoryManager.retrieve)
"""

from __future__ import annotations

from ..llm.client import GeminiClient
from ..llm.tagging import tag_event
from ..memory.manager import MemoryManager
from ..models import Memory, ThreatAssessment


def retrieve(
    player_input: str,
    assessment: ThreatAssessment,
    memory_manager: MemoryManager,
    llm: GeminiClient,
) -> list[Memory]:
    """
    Retrieve memories relevant to *player_input* given the current
    *assessment* response mode.

    Parameters
    ----------
    player_input:
        The raw text the player just sent.
    assessment:
        ThreatAssessment produced by Stage 1.
    memory_manager:
        Memory store to query.
    llm:
        Configured GeminiClient (used for tagging in instinct mode).

    Returns
    -------
    List of Memory objects (may be empty).
    """
    # Embed once; needed for both branches
    query_embedding = llm.embed(player_input)

    if assessment.response_mode == "instinct":
        # No strong embedding match — use 6D tag-vector cosine similarity
        tags = tag_event(player_input, "player input", llm)
        tag_vector = tags.to_vector()
        return memory_manager.retrieve_by_tag_vector(tag_vector, top_k=3)

    # standard or fight_flight — scored embedding retrieval (score ≥ 15 gate
    # is enforced inside MemoryManager.retrieve)
    if not query_embedding:
        return []
    return memory_manager.retrieve(query_embedding)

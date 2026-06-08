"""
Stage 2 — Personality-Weighted Memory Retrieval (paper §3.2)

Two retrieval primitives the orchestrator selects between based on the
threat judgement and whether scored retrieval qualifies any memories:

- ``scored_retrieve``  — top-5 by personality-weighted score, hard ≥15 gate.
- ``tag_retrieve``     — instinct fallback; cosine on 6-D EventTags.
"""

from __future__ import annotations

from ..llm.client import GeminiClient
from ..llm.tagging import tag_event
from ..memory.manager import MemoryManager
from ..models import Memory


def scored_retrieve(
    query_embedding: list[float],
    memory_manager: MemoryManager,
) -> list[Memory]:
    """Top-5 memories with personality-weighted score ≥ RETRIEVAL_THRESHOLD.

    Returns ``[]`` when nothing qualifies. The orchestrator interprets an
    empty result (in the no-threat branch) as a switch to instinct mode.
    """
    if not query_embedding:
        return []
    return memory_manager.retrieve(query_embedding)


def tag_retrieve(
    player_input: str,
    memory_manager: MemoryManager,
    llm: GeminiClient,
    top_k: int = 3,
) -> list[Memory]:
    """Instinct-mode retrieval — tag-vector cosine over the 6-D EventTags space."""
    tags = tag_event(player_input, "player input", llm)
    return memory_manager.retrieve_by_tag_vector(tags.to_vector(), top_k=top_k)

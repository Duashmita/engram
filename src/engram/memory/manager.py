from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING

import numpy as np

from ..config import RETRIEVAL_THRESHOLD, TOP_K_RETRIEVAL
from ..models import Memory, OCEANProfile
from .session import SessionMemory
from .longterm import LongTermMemory
from .keystore import KeyStore
from ..llm.tagging import summarize_turns

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    mag_a = float(np.linalg.norm(va))
    mag_b = float(np.linalg.norm(vb))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (mag_a * mag_b))


# ---------------------------------------------------------------------------
# Personality-weighted scoring
# ---------------------------------------------------------------------------

def _score(memory: Memory, query_embedding: list[float], profile: OCEANProfile) -> float:
    """Compute the personality-weighted retrieval score for *memory*.

    Formula (from paper):
        score = (rag_score * 2) * sum(t_mem / t_agent for each trait) * importance

    Where:
        rag_score  = cosine similarity of query_embedding and memory.embedding
        t_mem      = memory.tags.ocean[trait] / 5  (normalised 0–1)
        t_agent    = profile.effective[trait], floored at 0.05
        ratio      = capped at 5.0 per trait
        importance = memory.tags.importance (1–10)
    """
    rag = _cosine(query_embedding, memory.embedding)
    traits = ("O", "C", "E", "A", "N")
    eff = profile.effective
    ocean_sum = sum(
        min(5.0, (memory.tags.ocean.get(t, 3) / 5.0) / max(0.05, eff[t]))
        for t in traits
    )
    return (rag * 2) * ocean_sum * memory.tags.importance


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Unified memory interface used by NPCAgent.

    Coordinates SessionMemory, LongTermMemory, and KeyStore.  Persists
    all raw Memory objects to ``<data_dir>/memories.json``.
    """

    def __init__(
        self,
        npc_id: str,
        data_dir: str,
        profile: OCEANProfile,
        llm_client,
    ) -> None:
        self.npc_id = npc_id
        self.data_dir = data_dir
        self.profile = profile
        self.llm_client = llm_client

        os.makedirs(data_dir, exist_ok=True)

        self.session = SessionMemory()
        self.longterm = LongTermMemory(os.path.join(data_dir, "longterm.json"))
        self.keystore = KeyStore(os.path.join(data_dir, "keystore.pl"))

        self.all_memories: list[Memory] = []
        self.load_memories()

    # ------------------------------------------------------------------
    # Memory storage
    # ------------------------------------------------------------------

    def add_memory(self, memory: Memory) -> None:
        """Append *memory* to the full memory list and persist."""
        self.all_memories.append(memory)
        self.save_memories()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def add_turn(self, player: str, npc: str) -> None:
        """Record a dialogue turn; trigger summary + eviction when the window overflows."""
        evicted = self.session.add_turn(player, npc)
        if evicted is not None:
            summary = summarize_turns(evicted, self.profile, self.llm_client)
            self.longterm.add_summary(summary)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_RETRIEVAL,
    ) -> list[Memory]:
        """Personality-weighted retrieval.

        Returns up to *top_k* memories whose score >= RETRIEVAL_THRESHOLD,
        sorted by score descending.

        If no memory qualifies (instinct-mode fallback), returns the top-k
        by raw cosine similarity instead.
        """
        scored: list[tuple[float, Memory]] = []
        for mem in self.all_memories:
            s = _score(mem, query_embedding, self.profile)
            scored.append((s, mem))

        qualified = [(s, m) for s, m in scored if s >= RETRIEVAL_THRESHOLD]

        if qualified:
            qualified.sort(key=lambda t: t[0], reverse=True)
            result = qualified[:top_k]
        else:
            # Instinct-mode fallback: pure cosine similarity, no threshold.
            scored.sort(key=lambda t: _cosine(query_embedding, t[1].embedding), reverse=True)
            result = scored[:top_k]

        output: list[Memory] = []
        for s, mem in result:
            mem.score = s
            output.append(mem)
        return output

    def retrieve_by_tag_vector(
        self,
        query_vec: list[float],
        top_k: int = 3,
    ) -> list[Memory]:
        """Cosine similarity on 6-D EventTags vectors — no score threshold.

        Used for instinct-mode retrieval (social/emotional tag matching).
        """
        if not self.all_memories:
            return []

        scored = sorted(
            self.all_memories,
            key=lambda m: _cosine(query_vec, m.tags.to_vector()),
            reverse=True,
        )
        return scored[:top_k]

    def retrieve_top_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[Memory]:
        """Raw cosine similarity on full embeddings — no score threshold.

        Used during threat assessment to surface contextually similar memories.
        """
        if not self.all_memories:
            return []

        scored = sorted(
            self.all_memories,
            key=lambda m: _cosine(query_embedding, m.embedding),
            reverse=True,
        )
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Key memory promotion
    # ------------------------------------------------------------------

    def promote_key_memories(self) -> list[Memory]:
        """Delegate to KeyStore.update() and return the promoted memories."""
        return self.keystore.update(self.all_memories, self.profile)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_memories(self) -> None:
        """Serialise all_memories to ``<data_dir>/memories.json``."""
        path = os.path.join(self.data_dir, "memories.json")
        os.makedirs(self.data_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                [m.to_dict() for m in self.all_memories],
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def load_memories(self) -> None:
        """Load all_memories from ``<data_dir>/memories.json`` if it exists."""
        path = os.path.join(self.data_dir, "memories.json")
        if not os.path.exists(path):
            self.all_memories = []
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.all_memories = [Memory.from_dict(d) for d in raw]
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            self.all_memories = []

"""
NPCAgent — full pipeline orchestrator for Engram (paper §3).

Wires together MemoryManager and the six pipeline stages into a single
per-NPC agent that can be stepped turn-by-turn and persisted across
sessions.

Per turn (paper §3):
    1. Threat assessment           (§3.1, pipeline.threat)
    2. Personality-weighted        (§3.2, pipeline.retrieval.scored_retrieve)
       retrieval, with hard ≥15
       gate; falls back to instinct
       tag retrieval when nothing
       qualifies
    3. Response-mode selection     (§3.3) — derived here from threat +
       scored-retrieval results
    4. In-loop Prolog              (§3.4, pipeline.consolidation.check_contradictions)
       contradiction check
    5. LLM dialogue generation     (§3.3, pipeline.response)
    6. Memory consolidation        (§3.5, pipeline.consolidation.consolidate)

At end_session() the agent promotes top-25% memories to the Prolog
keystore and runs Openness-gated belief revision (paper §3.4 / §3.5).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING

from .config import DECAY_RATE
from .memory.manager import MemoryManager
from .models import Memory, NPCConfig, OCEANProfile
from .pipeline.consolidation import (
    check_contradictions,
    consolidate,
    post_session_fact_check,
)
from .pipeline.response import generate_response
from .pipeline.retrieval import scored_retrieve, tag_retrieve
from .pipeline.threat import assess_threat

if TYPE_CHECKING:
    from .llm.client import GeminiClient


class NPCAgent:
    """Self-contained NPC with personality-parameterised memory."""

    def __init__(
        self,
        config: NPCConfig,
        llm: "GeminiClient",
        data_dir: str = "data",
    ) -> None:
        self.config = config
        self.llm = llm

        self._npc_dir = os.path.join(data_dir, config.npc_id)
        os.makedirs(self._npc_dir, exist_ok=True)
        self._state_path = os.path.join(self._npc_dir, "state.json")

        self.memory_manager = MemoryManager(
            npc_id=config.npc_id,
            data_dir=self._npc_dir,
            profile=config.profile,
            llm_client=llm,
        )

        self.turn_count: int = 0
        self.history: list[dict] = []
        self.session_memories: list[Memory] = []

        if os.path.exists(self._state_path):
            self._load_state()
        else:
            self._init_backstory()

    # ------------------------------------------------------------------
    @property
    def profile(self) -> OCEANProfile:
        return self.config.profile

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_turn(self, player_input: str) -> str:
        """Process one player input through the full 6-stage pipeline."""
        # Embed once; reused by threat assessment + scored retrieval.
        query_embedding = self.llm.embed(player_input)

        # Stage 1 — threat assessment (§3.1)
        assessment = assess_threat(
            player_input,
            query_embedding,
            self.profile,
            self.memory_manager,
            self.llm,
        )

        # Stage 2 + 3 — scored retrieval drives mode selection (§3.2 / §3.3)
        if assessment.is_threat:
            self.profile.apply_fight_flight(assessment.threat_magnitude)
            mode = "fight_flight"
            retrieved = scored_retrieve(query_embedding, self.memory_manager)
        else:
            retrieved = scored_retrieve(query_embedding, self.memory_manager)
            if retrieved:
                mode = "standard"
            else:
                mode = "instinct"
                retrieved = tag_retrieve(player_input, self.memory_manager, self.llm)

        # Stage 4 — response generation (§3.3) — standard mode also gets
        # the OCEAN-biased long-term summaries (§3.3).
        summaries = (
            self.memory_manager.longterm.get_summaries() if mode == "standard" else None
        )
        response = generate_response(
            player_input=player_input,
            config=self.config,
            profile=self.profile,
            retrieved=retrieved,
            assessment=assessment,
            mode=mode,
            history=self.history,
            llm=self.llm,
            summaries=summaries,
        )

        # Stage 5 — post-response Prolog contradiction check (paper §3.4).
        # Extract facts from the NPC's own response, query the keystore,
        # and re-roll if a conflict is detected. Re-roll is gated by
        # Openness: low-O NPCs cannot tolerate inconsistency with their
        # established knowledge and regenerate; high-O NPCs let the
        # response stand and the drift gets resolved (or rejected) at
        # end_session via post_session_fact_check.
        # Player-input facts are NOT checked here — they're handled at
        # end_session, which is the only place the Prolog DB ever changes.
        if response and self.profile.O < 0.5:
            response_conflicts = check_contradictions(
                response, self.config, self.memory_manager.keystore, self.llm,
            )
            if response_conflicts:
                response = generate_response(
                    player_input=player_input,
                    config=self.config,
                    profile=self.profile,
                    retrieved=retrieved,
                    assessment=assessment,
                    mode=mode,
                    history=self.history,
                    llm=self.llm,
                    summaries=summaries,
                    prior_attempt=response,
                    prior_attempt_conflicts=response_conflicts,
                )

        # Stage 6 — consolidation (§3.5)
        new_memory = consolidate(
            player_input,
            response,
            self.config,
            self.profile,
            self.memory_manager,
            self.llm,
            memory_id=str(uuid.uuid4()),
        )
        self.session_memories.append(new_memory)

        # Bookkeeping — fight/flight delta decay
        self.profile.decay(DECAY_RATE)
        self.turn_count += 1
        self.history.append({"player": player_input, "npc": response})

        return response

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def end_session(self) -> None:
        """Promote key memories, reconcile facts (Openness-gated), persist."""
        self.memory_manager.promote_key_memories()
        post_session_fact_check(
            self.session_memories,
            self.config,
            self.profile,
            self.memory_manager.keystore,
            self.llm,
        )
        self.save_state()
        self.session_memories = []

    # ------------------------------------------------------------------
    # Backstory initialisation (first run only)
    # ------------------------------------------------------------------

    def _init_backstory(self) -> None:
        """Embed and store each backstory entry; assert seed Prolog facts."""
        from .llm.tagging import tag_event

        n = len(self.config.backstory)
        for i, backstory_text in enumerate(self.config.backstory):
            print(
                f"    [{self.config.npc_id}] backstory {i + 1}/{n} ...",
                end="\r",
                flush=True,
            )
            embedding = self.llm.embed(backstory_text)
            tags = tag_event(backstory_text, f"{self.config.name} backstory", self.llm)
            memory = Memory(
                id=f"{self.config.npc_id}_backstory_{i}",
                text=backstory_text,
                tags=tags,
                embedding=embedding,
                source="backstory",
            )
            self.memory_manager.add_memory(memory)
        print(f"    [{self.config.npc_id}] backstory {n}/{n} done.     ")

        for fact in self.config.initial_facts:
            self.memory_manager.keystore.assert_fact(fact)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist runtime state (deltas, history, turn count) + memories."""
        state = {
            "turn_count": self.turn_count,
            "history": self.history,
            "profile_deltas": {
                "_dO": self.profile._dO,
                "_dC": self.profile._dC,
                "_dE": self.profile._dE,
                "_dA": self.profile._dA,
                "_dN": self.profile._dN,
            },
        }
        with open(self._state_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
        self.memory_manager.save_memories()

    def _load_state(self) -> None:
        try:
            with open(self._state_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[NPCAgent:{self.config.npc_id}] Warning: failed to load state "
                f"({exc}). Starting with default values."
            )
            self.turn_count = 0
            self.history = []
            return

        self.turn_count = int(state.get("turn_count", 0))
        self.history = list(state.get("history", []))
        deltas = state.get("profile_deltas", {})
        for attr in ("_dO", "_dC", "_dE", "_dA", "_dN"):
            setattr(self.profile, attr, float(deltas.get(attr, 0.0)))

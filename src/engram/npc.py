"""
NPCAgent — full pipeline orchestrator for Engram.

Wires together MemoryManager, PrologEngine, and all pipeline stages into a
single per-NPC agent that can be stepped turn-by-turn and persisted across
sessions.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING

from .config import DECAY_RATE
from .memory.manager import MemoryManager
from .models import Memory, NPCConfig, OCEANProfile
from .pipeline.consolidation import consolidate, post_session_fact_check
from .pipeline.response import generate_response
from .pipeline.retrieval import retrieve
from .pipeline.threat import assess_threat
from .prolog_engine import PrologEngine

if TYPE_CHECKING:
    from .llm.client import GeminiClient


class NPCAgent:
    """A fully self-contained NPC with personality-parameterised memory.

    Parameters
    ----------
    config:
        Static NPC configuration (name, persona, backstory, OCEAN profile,
        initial Prolog facts).
    llm:
        Configured GeminiClient instance shared by all pipeline stages.
    data_dir:
        Root directory under which ``{npc_id}/`` subdirectory is created for
        all persistent state.  Defaults to ``"data"``.
    """

    def __init__(
        self,
        config: NPCConfig,
        llm: "GeminiClient",
        data_dir: str = "data",
    ) -> None:
        self.config = config
        self.llm = llm

        # Per-NPC data directory
        self._npc_dir = os.path.join(data_dir, config.npc_id)
        os.makedirs(self._npc_dir, exist_ok=True)

        self._state_path = os.path.join(self._npc_dir, "state.json")

        # Memory subsystem
        self.memory_manager = MemoryManager(
            npc_id=config.npc_id,
            data_dir=self._npc_dir,
            profile=config.profile,
            llm_client=llm,
        )

        # Prolog engine (rules live alongside the keystore .pl file)
        rules_path = os.path.join(self._npc_dir, "ocean_rules.pl")
        self.prolog_engine = PrologEngine(rules_path=rules_path)

        # Mutable runtime state
        self.turn_count: int = 0
        self.history: list[dict] = []
        self.session_memories: list[Memory] = []

        # Load or initialise persistent state
        if os.path.exists(self._state_path):
            self._load_state()
        else:
            self._init_backstory()

    # ------------------------------------------------------------------
    # Profile convenience property
    # ------------------------------------------------------------------

    @property
    def profile(self) -> OCEANProfile:
        """The NPC's current OCEAN personality profile."""
        return self.config.profile

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_turn(self, player_input: str) -> str:
        """Process one player input through the full 6-stage pipeline.

        Stages
        ------
        1. Threat assessment (amygdala pass).
        2. Apply fight/flight OCEAN deltas if threat detected.
        3. Personality-weighted memory retrieval.
        4. LLM dialogue generation.
        5. Memory consolidation (encode + store the exchange).
        6. Session bookkeeping (decay, turn count, history).

        Parameters
        ----------
        player_input:
            Raw text input from the player.

        Returns
        -------
        The NPC's generated response string.
        """
        # Stage 1 — Threat Assessment
        assessment = assess_threat(
            player_input,
            self.profile,
            self.memory_manager,
            self.llm,
        )

        # Stage 2 — Fight/Flight personality shift
        if assessment.response_mode == "fight_flight":
            self.profile.apply_fight_flight(assessment.threat_magnitude)

        # Stage 3 — Memory Retrieval
        retrieved = retrieve(
            player_input,
            assessment,
            self.memory_manager,
            self.llm,
        )

        # Stage 4 — Response Generation
        response = generate_response(
            player_input,
            self.config,
            self.profile,
            retrieved,
            assessment,
            self.history,
            self.llm,
        )

        # Stage 5 — Memory Consolidation
        memory_id = str(uuid.uuid4())
        new_memory = consolidate(
            player_input,
            response,
            self.config,
            self.profile,
            self.memory_manager,
            self.llm,
            memory_id,
        )
        self.session_memories.append(new_memory)

        # Stage 6 — Bookkeeping
        self.profile.decay(DECAY_RATE)
        self.turn_count += 1
        self.history.append({"player": player_input, "npc": response})

        return response

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def end_session(self) -> None:
        """Finalise the session: promote key memories, reconcile facts, persist.

        Call once after all turns for the session are complete.  Resets
        ``session_memories`` so the next session starts clean.
        """
        # 1. Promote top-25% memories to Prolog key-memory store
        self.memory_manager.promote_key_memories()

        # 2. Post-session Prolog fact reconciliation (Openness-gated)
        post_session_fact_check(
            self.session_memories,
            self.config,
            self.profile,
            self.memory_manager.keystore,
            self.llm,
        )

        # 3. Persist everything
        self.save_state()

        # 4. Reset session buffer
        self.session_memories = []

    # ------------------------------------------------------------------
    # Backstory initialisation (first run only)
    # ------------------------------------------------------------------

    def _init_backstory(self) -> None:
        """Embed and store each backstory entry as a 'backstory' Memory.

        Also asserts the NPC's ``initial_facts`` into the Prolog engine.
        Only called when no saved state file is found (first run).
        """
        # Import tagging here to avoid a circular import at module level
        from .llm.tagging import tag_event

        n = len(self.config.backstory)
        for i, backstory_text in enumerate(self.config.backstory):
            print(
                f"    [{self.config.npc_id}] backstory {i + 1}/{n} ...",
                end="\r",
                flush=True,
            )
            embedding = self.llm.embed(backstory_text)
            tags = tag_event(
                backstory_text,
                f"{self.config.name} backstory",
                self.llm,
            )
            memory = Memory(
                id=f"{self.config.npc_id}_backstory_{i}",
                text=backstory_text,
                tags=tags,
                embedding=embedding,
                source="backstory",
            )
            self.memory_manager.add_memory(memory)
        print(f"    [{self.config.npc_id}] backstory {n}/{n} done.     ")

        # Assert initial Prolog facts (pre-seeded world knowledge)
        for fact in self.config.initial_facts:
            self.prolog_engine.assert_fact(fact)
            self.memory_manager.keystore.assert_fact(fact)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist agent runtime state to ``<npc_dir>/state.json``.

        Saves:
        - OCEAN profile deltas (baseline values are in config; only the
          temporary fight/flight deltas and the turn count change at runtime).
        - Conversation history.
        - Turn count.

        Also delegates memory persistence to ``memory_manager.save_memories()``.
        """
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
        """Restore agent runtime state from ``<npc_dir>/state.json``.

        Restores OCEAN deltas, conversation history, and turn count.
        Memory objects are reloaded by MemoryManager on construction.
        """
        try:
            with open(self._state_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)

            self.turn_count = int(state.get("turn_count", 0))
            self.history = list(state.get("history", []))

            deltas = state.get("profile_deltas", {})
            self.profile._dO = float(deltas.get("_dO", 0.0))
            self.profile._dC = float(deltas.get("_dC", 0.0))
            self.profile._dE = float(deltas.get("_dE", 0.0))
            self.profile._dA = float(deltas.get("_dA", 0.0))
            self.profile._dN = float(deltas.get("_dN", 0.0))

        except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
            print(
                f"[NPCAgent:{self.config.npc_id}] Warning: failed to load state "
                f"({exc}). Starting with default values."
            )
            self.turn_count = 0
            self.history = []

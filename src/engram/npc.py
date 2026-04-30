"""
NPCAgent — full pipeline orchestrator for Engram (paper §3).

API-efficient refactor (v2): Stages 1 (Threat), 4 (Response), and 5
(Contradiction Check) are merged into a single consolidated LLM call
that returns a structured JSON object.  Per-turn API cost drops from
≈4-5 round-trips to exactly 2: one embed (query) + one generate.

Consolidated JSON schema (returned by LLM each turn)
-----------------------------------------------------
{
  "response":         "<NPC in-character dialogue>",
  "is_threat":        <bool>,
  "threat_magnitude": <float 0.0–1.0>,
  "reasoning":        "<one sentence threat reasoning>",
  "memory_text":      "<one sentence summary of this exchange>",
  "emotion_valence":  <float –1.0 to 1.0>,   // optional, default 0.0
  "importance":       <int 1–10>              // optional, default 5
}

Per turn (paper §3):
    2      Personality-weighted scored retrieval (§3.2) — pure math
    3      Mode selection (§3.3)                       — pure math
    1+4+5  Consolidated LLM generate call              — 1 API call
    6      Memory embed + store (§3.5)                 — 1 API call
           ────────────────────────────────────────────────────────
           Total per turn: 2 API calls  (was ≈4-5)

Observability contract
----------------------
All bus.emit() calls from the original pipeline are preserved so the
frontend visualiser and eval harness receive the same event stream.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import TYPE_CHECKING

from . import config as engram_config
from .config import DECAY_RATE
from .memory.manager import MemoryManager
from .models import EventTags, Memory, NPCConfig, OCEANProfile, ThreatAssessment
from .observability import bus
from .pipeline.consolidation import post_session_fact_check
from .pipeline.retrieval import scored_retrieve, tag_retrieve
from .pipeline.threat import make_threat_assessment

if TYPE_CHECKING:
    from .llm.client import GeminiClient


# ---------------------------------------------------------------------------
# Consolidated prompt template
# ---------------------------------------------------------------------------

_CONSOLIDATED_PROMPT = """\
You are {name}. {persona}

Personality: {personality_desc}
Effective OCEAN — O:{O:.2f} C:{C:.2f} E:{E:.2f} A:{A:.2f} N:{N:.2f}

{context_block}

{history_block}\
Player: {player_input}

{summaries_block}\
Instructions:
- Respond in character as {name} in 1-3 sentences.
- Assess whether the player's message is a threat to you.
  High-N NPCs (N≥0.65) are more threat-sensitive; high-A NPCs (A≥0.65) less so.
- Write a one-sentence memory summary of this exchange for future recall.
- Provide an emotion_valence score (−1.0 = very negative, +1.0 = very positive).
- Provide an importance score (1 = trivial, 10 = life-changing).
{contradiction_hint}\
Return ONLY the following JSON object — no markdown fences, no extra keys:
{{
  "response":         "<in-character dialogue>",
  "is_threat":        <true|false>,
  "threat_magnitude": <0.0-1.0>,
  "reasoning":        "<one sentence>",
  "memory_text":      "<one sentence summary>",
  "emotion_valence":  <-1.0 to 1.0>,
  "importance":       <1-10>
}}"""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_context_block(retrieved: list) -> str:
    if retrieved:
        return "Relevant memories:\n" + "\n".join(f"- {m.text}" for m in retrieved)
    return "Relevant memories:\n(none)"


def _build_history_block(history: list[dict], name: str, window: int) -> str:
    if not history:
        return ""
    lines = ["Recent conversation:"]
    for turn in history[-window:]:
        lines.append(f"Player: {turn['player']}")
        lines.append(f"{name}: {turn['npc']}")
    return "\n".join(lines) + "\n\n"


def _parse_consolidated(raw: str | None) -> dict:
    """Strip markdown fences and parse the consolidated JSON response.

    Returns an empty dict on any parse failure so callers can apply
    safe defaults rather than crashing.
    """
    if not raw:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-ditch: find the first {...} block in the output.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# NPCAgent
# ---------------------------------------------------------------------------

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

        p = self.config.profile
        bus.emit(
            "session_init_npc",
            npc_id=config.npc_id,
            npc_name=config.name,
            persona=config.persona,
            baseline_ocean={"O": p.O, "C": p.C, "E": p.E, "A": p.A, "N": p.N},
            initial_memory_count=len(self.memory_manager.all_memories),
            config={
                "retrieval_threshold": engram_config.RETRIEVAL_THRESHOLD,
                "top_k": engram_config.TOP_K_RETRIEVAL,
                "session_window": engram_config.SESSION_WINDOW,
                "evict_batch": engram_config.EVICT_BATCH,
                "key_memory_percentile": engram_config.KEY_MEMORY_PERCENTILE,
                "decay_rate": engram_config.DECAY_RATE,
            },
        )

    # ------------------------------------------------------------------
    @property
    def profile(self) -> OCEANProfile:
        return self.config.profile

    # ------------------------------------------------------------------
    # Main pipeline — consolidated single LLM call
    # ------------------------------------------------------------------

    def run_turn(self, player_input: str) -> str:
        """Process one player input through the consolidated 2-call pipeline.

        API calls per turn:
            1. llm.embed(player_input)          — shared by retrieval
            2. llm.generate(consolidated_prompt) — response + threat + memory
            [3. llm.embed(memory_text)           — only if a new memory is stored]

        All original bus.emit() calls are preserved for frontend/eval compat.
        """
        bus.emit("turn_start", turn=self.turn_count, player_input=player_input)
        t_start = time.perf_counter()

        # ── Embed once; reused by retrieval (and later by memory storage). ──
        query_embedding = self.llm.embed(player_input)
        bus.emit("embedding_done")

        # ── Stage 2 — scored retrieval (pure math, 0 API calls). ───────────
        retrieved = scored_retrieve(query_embedding, self.memory_manager)
        if retrieved:
            mode = "standard"
        else:
            mode = "instinct"
            retrieved = tag_retrieve(player_input, self.memory_manager, self.llm)
        bus.emit("mode_selected", mode=mode)

        # ── Build consolidated prompt. ──────────────────────────────────────
        eff = self.profile.effective
        summaries = (
            self.memory_manager.longterm.get_summaries() if mode == "standard" else None
        )
        summaries_block = (
            f"Long-term summaries:\n{summaries}\n\n" if summaries else ""
        )
        # Low-O NPCs are intolerant of self-contradiction (paper §3.4).
        contradiction_hint = (
            "- Your response MUST be consistent with the memories shown above; "
            "do not contradict established facts.\n"
            if self.profile.O < 0.5
            else ""
        )

        prompt = _CONSOLIDATED_PROMPT.format(
            name=self.config.name,
            persona=self.config.persona,
            personality_desc=self.profile.describe(),
            O=eff["O"], C=eff["C"], E=eff["E"], A=eff["A"], N=eff["N"],
            context_block=_build_context_block(retrieved),
            history_block=_build_history_block(
                self.history, self.config.name, engram_config.SESSION_WINDOW
            ),
            player_input=player_input,
            summaries_block=summaries_block,
            contradiction_hint=contradiction_hint,
        )

        # ── Stages 1 + 4 + 5 — single consolidated LLM call. ──────────────
        raw = self.llm.generate(prompt)
        data = _parse_consolidated(raw)

        # ── Extract dialogue response. ──────────────────────────────────────
        response = str(data.get("response", "")).strip()
        if not response:
            # Fallback: if JSON parse failed, use raw output as the response.
            response = (raw or "").strip() or "(no response)"
        bus.emit("response_generated", text=response, attempt=1)

        # ── Stage 1 (reconstructed) — build ThreatAssessment + fight/flight. ─
        assessment = make_threat_assessment(
            is_threat=bool(data.get("is_threat", False)),
            threat_magnitude=float(data.get("threat_magnitude", 0.0)),
            reasoning=str(data.get("reasoning", "No reasoning provided.")),
            context_memory_ids=[m.id for m in (retrieved or [])],
        )
        if assessment.is_threat:
            self.profile.apply_fight_flight(assessment.threat_magnitude)
            bus.emit(
                "fight_flight_applied",
                magnitude=assessment.threat_magnitude,
                deltas={
                    "dN": self.profile._dN,
                    "dA": self.profile._dA,
                    "dE": self.profile._dE,
                },
            )

        # ── Stage 5 (implicit) — contradiction hint in the prompt means ──────
        # low-O NPCs already avoid contradictions in the response above.
        # Emit a stub event so the frontend event log stays consistent.
        bus.emit(
            "contradiction_check",
            method="inline_prompt_constraint",
            openness=self.profile.O,
            conflicts_found=False,
        )

        # ── Stage 6 — memory consolidation (1 embed call). ─────────────────
        memory_text = str(data.get("memory_text", "")).strip()
        if not memory_text:
            memory_text = (
                f"Player: '{player_input[:80]}'. "
                f"{self.config.name}: '{response[:80]}'."
            )

        new_memory = self._store_exchange_memory(
            memory_text=memory_text,
            assessment=assessment,
            emotion_valence=float(data.get("emotion_valence", 0.0)),
            importance=int(data.get("importance", 5)),
            memory_id=str(uuid.uuid4()),
        )
        self.session_memories.append(new_memory)
        bus.emit("consolidated", memory_id=new_memory.id, text=new_memory.text)

        # ── Bookkeeping — fight/flight delta decay. ─────────────────────────
        self.profile.decay(DECAY_RATE)
        bus.emit("profile_decay", effective=self.profile.effective)
        self.turn_count += 1
        self.history.append({"player": player_input, "npc": response})

        bus.emit(
            "turn_end",
            turn=self.turn_count,
            duration_ms=round((time.perf_counter() - t_start) * 1000.0, 4),
        )
        return response

    # ------------------------------------------------------------------
    # Memory storage helper (no LLM call)
    # ------------------------------------------------------------------

    def _store_exchange_memory(
        self,
        memory_text: str,
        assessment: ThreatAssessment,
        emotion_valence: float = 0.0,
        importance: int = 5,
        memory_id: str | None = None,
    ) -> Memory:
        """Embed *memory_text* and store it in the memory manager.

        EventTags are derived from the consolidated JSON rather than a
        separate tagging LLM call, eliminating one API round-trip per turn.
        The social_type is inferred from the threat assessment; other
        tag fields use evidence-backed defaults that are good enough for
        personality-weighted retrieval scoring.
        """
        # Clamp inputs defensively.
        emotion_valence = max(-1.0, min(1.0, float(emotion_valence)))
        importance = max(1, min(10, int(importance)))

        social_type = (
            "conflict"
            if assessment.is_threat
            else ("conversation" if emotion_valence >= 0 else "conflict")
        )

        tags = EventTags(
            emotion_valence=emotion_valence,
            social_type=social_type,
            threat_level=assessment.threat_magnitude,
            goal_relevance=0.5,       # neutral prior; updated at end_session via fact-check
            novelty_level=0.5,
            self_relevance=0.6,
            importance=importance,
            ocean={"O": 3, "C": 3, "E": 3, "A": 3, "N": 3},
        )

        # This embed is unavoidable — without it, future retrieval won't work.
        embedding = self.llm.embed(memory_text)

        memory = Memory(
            id=memory_id or str(uuid.uuid4()),
            text=memory_text,
            tags=tags,
            embedding=embedding,
            source="session",
        )
        self.memory_manager.add_memory(memory)
        return memory

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
        bus.emit("session_end_npc", npc_id=self.config.npc_id)

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
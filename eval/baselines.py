"""
Baseline + Engram agent adapters for the showcase eval.

All adapters share the ``Agent`` protocol so the runner doesn't have to
care which system it's stepping. ``run_turn(text) -> TurnRecord``
captures everything the metrics layer needs:

    response          str   — the NPC's text reply
    threat_level      float|None — what (if anything) the system flagged
    retrieved_ids     list[str]  — IDs of memories the system retrieved.
                                   Empty for systems with no retrieval;
                                   for cosine-RAG it's the top-1 backstory
                                   line ID; for Engram it's the IDs of
                                   memories scored above threshold (or
                                   tag-mode top-3 if instinct fired).
    stored_tags       dict|None  — full EventTags.to_dict() for the memory
                                   the system stored this turn. None for
                                   systems that don't tag (everything
                                   except the two Engram variants).
    mode              str|None   — Engram's mode for this turn
                                   ("standard" | "fight_flight" | "instinct").
                                   None for non-Engram systems.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

# Make src/ importable so ``engram`` resolves whether we're called as
# ``python -m eval.runner`` from the repo root or directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from engram.config import SESSION_WINDOW
from engram.models import Memory, NPCConfig, OCEANProfile
from engram.npc import NPCAgent
from engram.observability import bus
from engram.presets import get_preset

from .scenario import ALL_SYSTEMS  # re-exported for runner.py back-compat


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """One turn of one (system, personality) pair, captured for metrics."""
    turn_index: int
    phase: str                     # "session" | "probe"
    player_input: str
    response: str
    threat_level: float | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    stored_tags: dict | None = None
    mode: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Agent(Protocol):
    """Common contract every system implements."""
    name: str

    def run_turn(self, player_input: str, *, turn_index: int, phase: str) -> TurnRecord: ...
    def end_session(self) -> None: ...
    def reset_session(self) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _format_history(history: list[dict], name: str, window: int = SESSION_WINDOW) -> str:
    if not history:
        return ""
    recent = history[-window:]
    lines = ["Recent conversation:"]
    for turn in recent:
        lines.append(f"Player: {turn['player']}")
        lines.append(f"{name}: {turn['npc']}")
    return "\n".join(lines) + "\n"


def _safe_response(text: str | None) -> str:
    if text is None:
        return "(no response)"
    cleaned = text.strip()
    return cleaned or "(no response)"


# ---------------------------------------------------------------------------
# Baseline 1 — Standard cosine RAG over backstory
# ---------------------------------------------------------------------------

class CosineRAGAgent:
    """Naive cosine-RAG baseline.

    Retrieves the single most semantically similar backstory line for
    each player input, drops it into a persona-flavoured prompt, and
    appends a recent-history block. No tagging, no personality scoring,
    no threat assessment, no storage — same query always retrieves the
    same line regardless of OCEAN.
    """

    name = "cosine_rag"

    def __init__(
        self,
        config: NPCConfig,
        llm,
        backstory_memories: list[Memory] | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.history: list[dict] = []

        if backstory_memories:
            self._ids = [m.id for m in backstory_memories]
            self._texts = [m.text for m in backstory_memories]
            self._embeds = [m.embedding for m in backstory_memories]
        else:
            self._ids = [
                f"{config.npc_id}_backstory_{i}"
                for i in range(len(config.backstory))
            ]
            self._texts = list(config.backstory)
            self._embeds = [llm.embed(s) for s in config.backstory]

    def _retrieve(self, query_embedding: list[float]) -> tuple[str, str]:
        """Return ``(memory_id, text)`` for the top-1 backstory line."""
        if not self._embeds:
            return ("", "")
        scores = [_cosine(query_embedding, e) for e in self._embeds]
        best = int(np.argmax(scores))
        return (self._ids[best], self._texts[best])

    def run_turn(self, player_input: str, *, turn_index: int, phase: str) -> TurnRecord:
        query_embedding = self.llm.embed(player_input)
        ret_id, ret_text = self._retrieve(query_embedding)
        context_line = f"\nRelevant context: {ret_text}\n" if ret_text else ""
        history_block = _format_history(self.history, self.config.name)

        prompt = (
            f"You are {self.config.name}. {self.config.persona}"
            f"{context_line}\n"
            f"{history_block}"
            f"\nPlayer: {player_input}\n"
            f"{self.config.name}:"
        )
        response = _safe_response(self.llm.generate(prompt))
        self.history.append({"player": player_input, "npc": response})

        return TurnRecord(
            turn_index=turn_index,
            phase=phase,
            player_input=player_input,
            response=response,
            retrieved_ids=[ret_id] if ret_id else [],
        )

    def end_session(self) -> None:
        # No consolidation — baseline just keeps the rolling history.
        pass

    def reset_session(self) -> None:
        self.history = []


# ---------------------------------------------------------------------------
# Baseline 2 — Persona only, no retrieval at all
# ---------------------------------------------------------------------------

class PersonaOnlyAgent:
    """Persona text + recent history only. No retrieval of any kind."""

    name = "persona_only"

    def __init__(self, config: NPCConfig, llm) -> None:
        self.config = config
        self.llm = llm
        self.history: list[dict] = []

    def run_turn(self, player_input: str, *, turn_index: int, phase: str) -> TurnRecord:
        history_block = _format_history(self.history, self.config.name)
        prompt = (
            f"You are {self.config.name}. {self.config.persona}\n"
            f"{history_block}"
            f"\nPlayer: {player_input}\n"
            f"{self.config.name}:"
        )
        response = _safe_response(self.llm.generate(prompt))
        self.history.append({"player": player_input, "npc": response})

        return TurnRecord(
            turn_index=turn_index,
            phase=phase,
            player_input=player_input,
            response=response,
        )

    def end_session(self) -> None:
        pass

    def reset_session(self) -> None:
        self.history = []


# ---------------------------------------------------------------------------
# Baseline 3 — Long-context: dump full backstory + full history every turn
# ---------------------------------------------------------------------------

class LongContextAgent:
    """No retrieval, no truncation — everything goes in the prompt."""

    name = "long_context"

    def __init__(self, config: NPCConfig, llm) -> None:
        self.config = config
        self.llm = llm
        self.history: list[dict] = []

    def run_turn(self, player_input: str, *, turn_index: int, phase: str) -> TurnRecord:
        backstory_block = (
            "Backstory:\n"
            + "\n".join(f"- {line}" for line in self.config.backstory)
            + "\n"
            if self.config.backstory else ""
        )
        if self.history:
            hist_lines = ["Full conversation:"]
            for turn in self.history:
                hist_lines.append(f"Player: {turn['player']}")
                hist_lines.append(f"{self.config.name}: {turn['npc']}")
            history_block = "\n".join(hist_lines) + "\n"
        else:
            history_block = ""

        prompt = (
            f"You are {self.config.name}. {self.config.persona}\n\n"
            f"{backstory_block}"
            f"{history_block}"
            f"\nPlayer: {player_input}\n"
            f"{self.config.name}:"
        )
        response = _safe_response(self.llm.generate(prompt))
        self.history.append({"player": player_input, "npc": response})

        return TurnRecord(
            turn_index=turn_index,
            phase=phase,
            player_input=player_input,
            response=response,
        )

    def end_session(self) -> None:
        pass

    def reset_session(self) -> None:
        self.history = []


# ---------------------------------------------------------------------------
# Engram wrappers (full + no-OCEAN ablation)
# ---------------------------------------------------------------------------

class _EngramRetrievalCapture:
    """Subscribes to bus.retrieval_scored events for the duration of a
    single run_turn() call and exposes the most recent payload.
    """

    def __init__(self) -> None:
        self.last: dict | None = None
        self._unsub = None

    def __enter__(self) -> "_EngramRetrievalCapture":
        bus.activate()
        self._unsub = bus.subscribe(self._on_event)
        return self

    def __exit__(self, *exc) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        bus.deactivate()

    def _on_event(self, event: dict) -> None:
        if event.get("type") == "retrieval_scored":
            self.last = event.get("payload", {})


class _EngramAdapterBase:
    """Common scaffolding for both Engram-flavoured agents."""

    name = "engram"

    def __init__(self, npc_agent: NPCAgent) -> None:
        self.agent = npc_agent
        self.config = npc_agent.config

    def run_turn(self, player_input: str, *, turn_index: int, phase: str) -> TurnRecord:
        with _EngramRetrievalCapture() as capture:
            response = self.agent.run_turn(player_input)

        # ``mode`` isn't on NPCAgent directly but the capture's event
        # payload tells us whether it was scored vs tag-mode.
        retrieval_payload = capture.last or {}
        retrieved_ids = list(retrieval_payload.get("selected_ids") or [])
        retrieval_mode = retrieval_payload.get("mode")
        # Translate retrieval mode into the higher-level NPC mode label
        # (NPCAgent emits "mode_selected" too but we'd need a second
        # subscription; the retrieval mode is sufficient for metrics).
        mode = {"scored": "standard", "tag": "instinct"}.get(retrieval_mode)

        # Pull the just-stored memory's tags from session_memories.
        last_mem = (
            self.agent.session_memories[-1]
            if self.agent.session_memories
            else None
        )
        stored_tags = last_mem.tags.to_dict() if last_mem is not None else None
        threat_level = (
            float(last_mem.tags.threat_level) if last_mem is not None else None
        )

        return TurnRecord(
            turn_index=turn_index,
            phase=phase,
            player_input=player_input,
            response=_safe_response(response),
            threat_level=threat_level,
            retrieved_ids=retrieved_ids,
            stored_tags=stored_tags,
            mode=mode,
        )

    def end_session(self) -> None:
        self.agent.end_session()

    def reset_session(self) -> None:
        # New conversation, same agent: drop history and the in-flight
        # session memory buffer so probes are a clean slate.
        self.agent.history = []
        self.agent.session_memories = []


class EngramFullAgent(_EngramAdapterBase):
    """Vanilla Engram — full personality-weighted pipeline."""
    name = "engram_full"


class EngramNoOCEANAgent(_EngramAdapterBase):
    """Engram with OCEAN flattened to neutral 0.5 across the board.

    The pipeline still tags, scores, and runs the Prolog check — but
    every retrieval-formula trait ratio collapses toward 1.0, so the
    only signal is RAG-similarity × importance. Isolates how much
    Engram's behaviour comes specifically from the OCEAN term.
    """
    name = "engram_no_ocean"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_agent(
    system: str,
    personality: str,
    llm,
    *,
    data_dir: str,
) -> Agent:
    """Construct an agent of *system* type for the named *personality* preset.

    Engram variants persist state under ``data_dir/<personality>/`` —
    the runner is responsible for wiping that directory before calling
    this so each (system, personality) pair starts cold.
    """
    config = get_preset(personality)

    if system == "cosine_rag":
        return CosineRAGAgent(config, llm)
    if system == "persona_only":
        return PersonaOnlyAgent(config, llm)
    if system == "long_context":
        return LongContextAgent(config, llm)
    if system == "engram_full":
        npc = NPCAgent(config, llm, data_dir=data_dir)
        return EngramFullAgent(npc)
    if system == "engram_no_ocean":
        # Force the OCEAN profile to neutral BEFORE constructing NPCAgent
        # so backstory tagging + initial bus events see the flat profile.
        config.profile = OCEANProfile(
            name=f"{config.name} (no-OCEAN)",
            O=0.5, C=0.5, E=0.5, A=0.5, N=0.5,
        )
        npc = NPCAgent(config, llm, data_dir=data_dir)
        return EngramNoOCEANAgent(npc)

    raise ValueError(
        f"unknown system '{system}'. Choose from: {', '.join(ALL_SYSTEMS)}"
    )

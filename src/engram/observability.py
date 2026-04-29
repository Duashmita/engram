"""
Engram observability — side-effect-free pub/sub event bus.

Emits structured NDJSON events alongside the existing pipeline so an
external static visualizer can reconstruct what happened on every turn.

This module is **passive**: when no session is active, every emit is a
no-op. When a session is active, events are appended to an in-memory
buffer and (optionally) flushed to an NDJSON sink. All file/encode
errors are caught and reported to stderr — observability MUST NEVER
crash the pipeline.

EVENT SCHEMA
============

session_init       header              { npc_id, npc_name, persona, baseline_ocean: {O,C,E,A,N},
                                         initial_memory_count, config: {...} }
session_init_npc   per-NPC bootstrap   (same payload as session_init; emitted in NPCAgent.__init__)
turn_start         turn began          { turn, player_input }
embedding_done     embed call done     {}
threat_assessed    stage 1 result      { is_threat, magnitude, reasoning, context_memory_ids: [id] }
retrieval_scored   stage 2 result      { mode: "scored"|"tag"|"top_scored",
                                         scored: [{id, text, score, rag, ocean_sum, importance, qualified}],
                                         threshold, selected_ids }
mode_selected      stage 3 result      { mode: "standard"|"fight_flight"|"instinct" }
fight_flight_applied                   { magnitude, deltas: {dN, dA, dE} }
contradiction_check stage 4 result     { stage: "pre"|"post"|"generic", text, conflicts: [{new, old}] }
response_generated stage 5 result      { text, attempt: 1|2 }
consolidated       stage 6 done        { memory_id, text }
memory_added       store write         { id, source, importance, text, tags }
summary_added      LT summary write    { summary, total_summaries }
key_promoted       key-mem promotion   { memory_ids, count }
fact_asserted      Prolog write        { fact }
fact_revised       Prolog rewrite      { fact, old }
fact_rejected      Prolog rejected     { fact, old }
profile_decay      end-of-turn decay   { effective: {O,C,E,A,N} }
turn_end           turn done           { turn, duration_ms }
session_end        session done        {}
session_end_npc    per-NPC end         { npc_id }
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from threading import Lock


_ENV_PATH_KEY = "ENGRAM_VIZ_LOG"


class _EventBus:
    """In-memory + optional NDJSON-sink event bus.

    Singleton at module level (`bus`). Inactive by default — the very
    first thing `emit()` checks is `_active`, so calls during normal
    CLI usage are essentially free.
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._start_time: float = 0.0
        self._events: list[dict] = []
        self._sink_path: str | None = None
        self._fh = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_session(self, path: str | None = None, *, header: dict) -> None:
        """Open the bus, write a ``session_init`` event, and (optionally) a NDJSON sink.

        If *path* is None the env var ``ENGRAM_VIZ_LOG`` is consulted.
        If neither is set the bus runs in-memory only.
        """
        with self._lock:
            self._active = True
            self._start_time = time.perf_counter()
            self._events = []

            resolved_path = path or os.environ.get(_ENV_PATH_KEY) or None
            self._sink_path = resolved_path
            self._fh = None
            if resolved_path:
                try:
                    parent = os.path.dirname(resolved_path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    self._fh = open(resolved_path, "w", encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[observability] failed to open sink {resolved_path}: {exc}",
                        file=sys.stderr,
                    )
                    self._fh = None

        # Emit AFTER releasing the lock — emit acquires its own lock.
        self.emit("session_init", **header)

    def end_session(self) -> None:
        """Emit ``session_end`` and close any open NDJSON sink."""
        if not self._active:
            return
        self.emit("session_end")
        with self._lock:
            self._active = False
            if self._fh is not None:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception as exc:  # noqa: BLE001
                    print(f"[observability] sink close error: {exc}", file=sys.stderr)
                finally:
                    self._fh = None

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, type: str, **payload) -> None:  # noqa: A002 — match spec
        # Bail on the first line when inactive — keeps cost ~free.
        if not self._active:
            return
        try:
            t = round(time.perf_counter() - self._start_time, 4)
            event = {"t": t, "type": type, "payload": payload}
        except Exception as exc:  # noqa: BLE001
            print(f"[observability] event build error: {exc}", file=sys.stderr)
            return

        with self._lock:
            try:
                self._events.append(event)
            except Exception as exc:  # noqa: BLE001
                print(f"[observability] buffer append error: {exc}", file=sys.stderr)

            fh = self._fh
            if fh is not None:
                try:
                    fh.write(json.dumps(event, ensure_ascii=False, default=_json_default))
                    fh.write("\n")
                    fh.flush()
                except Exception as exc:  # noqa: BLE001
                    print(f"[observability] sink write error: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        return self._active

    def events(self) -> list[dict]:
        """Return a shallow copy of the buffered events (for tests)."""
        with self._lock:
            return list(self._events)

    # ------------------------------------------------------------------
    # Stage timing helper
    # ------------------------------------------------------------------

    @contextmanager
    def stage(self, name: str):
        """Time a stage block; emit ``stage_timing`` on exit. Use sparingly."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            try:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                self.emit("stage_timing", name=name, duration_ms=round(dur_ms, 4))
            except Exception as exc:  # noqa: BLE001
                print(f"[observability] stage_timing error: {exc}", file=sys.stderr)


def _json_default(obj):
    """Best-effort fallback for non-JSON-serialisable values."""
    try:
        return str(obj)
    except Exception:  # noqa: BLE001
        return None


# Module-level singleton.
bus = _EventBus()


# Auto-activate when ENGRAM_VIZ_LOG is set and the user hasn't called
# start_session() themselves. We don't have a header at import time,
# so we leave the bus dormant — a caller (chat.py / notebooks) is
# expected to invoke `bus.start_session(...)` explicitly when they
# want the file written. This module-level autodiscovery is documented
# in the docstring so behaviour stays predictable.

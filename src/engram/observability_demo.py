"""Tiny smoke test for the observability bus.

Run directly:

    python src/engram/observability_demo.py

Writes a synthetic NDJSON log to /tmp/engram_viz_test.ndjson and
prints "OK" if the file exists and is fully parseable.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    # Allow running both as `python src/engram/observability_demo.py`
    # (relative-imports unavailable) and as a module.
    here = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.dirname(here)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    from engram.observability import bus  # noqa: WPS433 — runtime import

    log_path = "/tmp/engram_viz_test.ndjson"
    if os.path.exists(log_path):
        os.remove(log_path)

    bus.start_session(
        log_path,
        header={
            "test": True,
            "npc_id": "demo",
            "npc_name": "Demo",
            "persona": "demo persona",
            "baseline_ocean": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
            "initial_memory_count": 0,
            "config": {"retrieval_threshold": 30.0},
        },
    )

    bus.emit("turn_start", turn=0, player_input="hello")
    bus.emit("embedding_done")
    bus.emit(
        "threat_assessed",
        is_threat=False,
        magnitude=0.0,
        reasoning="benign",
        context_memory_ids=[],
    )
    bus.emit(
        "retrieval_scored",
        mode="scored",
        scored=[],
        threshold=30.0,
        selected_ids=[],
    )
    bus.emit("mode_selected", mode="instinct")
    bus.emit("response_generated", text="hi", attempt=1)
    bus.emit("turn_end", turn=1, duration_ms=12.34)

    bus.end_session()

    if not os.path.exists(log_path):
        print("FAIL: log file not written")
        return 1

    with open(log_path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]

    try:
        events = [json.loads(ln) for ln in lines]
    except json.JSONDecodeError as exc:
        print(f"FAIL: NDJSON parse error: {exc}")
        return 1

    if not events or events[0]["type"] != "session_init":
        print(f"FAIL: first event must be session_init, got {events[:1]}")
        return 1

    if events[-1]["type"] != "session_end":
        print(f"FAIL: last event must be session_end, got {events[-1:]}")
        return 1

    print(f"OK ({len(events)} events written to {log_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

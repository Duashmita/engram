"""
Runner for the personality-vs-baselines showcase eval.

For each (system, personality) pair:
    1. Wipe the per-pair work directory.
    2. Instantiate the agent.
    3. Replay scenario.INPUTS in order, collecting per-turn TurnRecords.
    4. Call end_session() (Engram only — no-op for the others).
    5. reset_session() and replay scenario.PROBES.
    6. Write eval/results/traces/<system>/<personality>.json.

Usage:
    python -m eval.runner --systems all --personalities guard,merchant,clerk
    python -m eval.runner --quick                          # 3 inputs + 1 probe
    python -m eval.runner --systems engram_full --personalities guard
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings

# Suppress noisy SDK warnings before any engram import.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import logging as _logging
_logging.getLogger("engram").setLevel(_logging.ERROR)

# Make src/ importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from engram.config import LLM_CACHE_DIR
from engram.llm.cached_client import CachedGeminiClient
from engram.llm.client import GeminiClient

from .baselines import TurnRecord, make_agent
from .scenario import ALL_SYSTEMS, INPUTS, PERSONALITIES, PROBES


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------

def trace_path(out_dir: str, system: str, personality: str) -> str:
    return os.path.join(out_dir, "traces", system, f"{personality}.json")


def write_trace(
    out_dir: str,
    system: str,
    personality: str,
    session_turns: list[TurnRecord],
    probe_turns: list[TurnRecord],
) -> str:
    path = trace_path(out_dir, system, personality)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "system": system,
        "personality": personality,
        "session": [t.to_dict() for t in session_turns],
        "probes": [t.to_dict() for t in probe_turns],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Main per-pair driver
# ---------------------------------------------------------------------------

def run_pair(
    system: str,
    personality: str,
    llm,
    *,
    out_dir: str,
    inputs: list[str],
    probes,
    work_dir: str,
) -> str:
    """Run one (system, personality) pair end-to-end and write its trace JSON."""
    pair_work = os.path.join(work_dir, system, personality)
    if os.path.isdir(pair_work):
        shutil.rmtree(pair_work)
    os.makedirs(pair_work, exist_ok=True)

    print(f"\n  [run] {system} × {personality}")
    agent = make_agent(system, personality, llm, data_dir=pair_work)

    # ── Phase 1: scripted inputs ────────────────────────────────────────────
    session_turns: list[TurnRecord] = []
    for i, text in enumerate(inputs, start=1):
        print(f"    turn {i}/{len(inputs)} …", flush=True)
        record = agent.run_turn(text, turn_index=i, phase="session")
        session_turns.append(record)

    # End the session so Engram variants promote key memories + reconcile beliefs.
    agent.end_session()

    # ── Phase 2: probes (new session) ───────────────────────────────────────
    agent.reset_session()
    probe_turns: list[TurnRecord] = []
    for i, probe in enumerate(probes, start=1):
        print(f"    probe {i}/{len(probes)} …", flush=True)
        record = agent.run_turn(probe.input, turn_index=i, phase="probe")
        probe_turns.append(record)

    path = write_trace(out_dir, system, personality, session_turns, probe_turns)
    print(f"    → {os.path.relpath(path, _REPO_ROOT)}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_csv(arg: str, valid: list[str]) -> list[str]:
    if arg.strip().lower() == "all":
        return list(valid)
    items = [s.strip() for s in arg.split(",") if s.strip()]
    bad = [s for s in items if s not in valid]
    if bad:
        raise SystemExit(
            f"unknown values {bad}. Valid: {', '.join(valid)} (or 'all')"
        )
    return items


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval.runner",
        description="Personality-vs-baselines scenario runner",
    )
    p.add_argument("--systems", default="all",
                   help=f"comma list or 'all' ({', '.join(ALL_SYSTEMS)})")
    p.add_argument("--personalities", default="all",
                   help=f"comma list or 'all' ({', '.join(PERSONALITIES)})")
    p.add_argument("--out", default="eval/results", help="output root")
    p.add_argument("--work-dir", default="eval/work",
                   help="scratch directory for Engram per-pair state")
    p.add_argument("--quick", action="store_true",
                   help="use only the first 3 inputs + 1 probe")
    p.add_argument("--no-cache", action="store_true",
                   help="bypass LLM disk cache")
    p.add_argument("--model", default=None, help="override Gemini chat model")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    systems = _parse_csv(args.systems, ALL_SYSTEMS)
    personalities = _parse_csv(args.personalities, PERSONALITIES)

    inputs = INPUTS[:3] if args.quick else INPUTS
    probes = PROBES[:1] if args.quick else PROBES

    out_dir = os.path.abspath(args.out)
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    base = GeminiClient(chat_model=args.model) if args.model else GeminiClient()
    llm = CachedGeminiClient(base, cache_dir=LLM_CACHE_DIR, enabled=not args.no_cache)

    print(f"  systems       : {systems}")
    print(f"  personalities : {personalities}")
    print(f"  inputs        : {len(inputs)} (probe phase: {len(probes)})")
    print(f"  out           : {os.path.relpath(out_dir, _REPO_ROOT)}")

    for system in systems:
        for personality in personalities:
            run_pair(
                system, personality, llm,
                out_dir=out_dir,
                inputs=inputs,
                probes=probes,
                work_dir=work_dir,
            )

    stats = llm.cache_stats()
    if stats.get("enabled"):
        print(
            f"\n  cache: generate={stats.get('generate', 0)}, "
            f"embed={stats.get('embed', 0)}  ({LLM_CACHE_DIR}/)"
        )
    print("\n  done.")


if __name__ == "__main__":
    main()

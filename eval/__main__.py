"""
Single-shot pipeline: runner → metrics → report.

Usage:
    python -m eval                                    # full run
    python -m eval --quick                            # 3 inputs, 1 probe
    python -m eval --systems engram_full,cosine_rag   # subset
    python -m eval --skip-judge                       # math-only metrics
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import logging as _logging
_logging.getLogger("engram").setLevel(_logging.ERROR)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from engram.config import LLM_CACHE_DIR
from engram.llm.cached_client import CachedGeminiClient
from engram.llm.client import GeminiClient

from .metrics import compute_all
from .report import write_report
from .runner import _parse_csv, run_pair
from .scenario import ALL_SYSTEMS, INPUTS, PERSONALITIES, PROBES


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval",
        description="run the full personality-vs-baselines eval pipeline",
    )
    p.add_argument("--systems", default="all")
    p.add_argument("--personalities", default="all")
    p.add_argument("--out", default="eval/results")
    p.add_argument("--work-dir", default="eval/work")
    p.add_argument("--quick", action="store_true",
                   help="3 inputs + 1 probe; for fast iteration")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--skip-judge", action="store_true",
                   help="skip the two LLM-judge metrics (cheap math only)")
    p.add_argument("--skip-runner", action="store_true",
                   help="reuse existing traces; only recompute metrics + report")
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

    if not args.skip_runner:
        for system in systems:
            for personality in personalities:
                run_pair(
                    system, personality, llm,
                    out_dir=out_dir,
                    inputs=inputs,
                    probes=probes,
                    work_dir=work_dir,
                )

    compute_all(out_dir, systems, llm, skip_judge=args.skip_judge)
    write_report(out_dir, systems)


if __name__ == "__main__":
    main()

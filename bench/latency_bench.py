"""
Model latency benchmark for Engram's dialogue-generation pipeline.

Runs the pipeline's real prompts (threat assessment, dialogue generation,
memory tagging - see prompts.py) against a set of candidate LLM clients and
reports p50/p95 wall-clock latency per prompt.

Usage:
    python bench/latency_bench.py
    python bench/latency_bench.py --runs 10
    python bench/latency_bench.py --candidates gemini-3-flash,claude-haiku-4-5

Requires GEMINI_API_KEY, ANTHROPIC_API_KEY, INCEPTION_API_KEY, and/or
MOONSHOT_API_KEY in the environment (or the project's .env) for the cloud
candidates. The ollama-* candidates require a local `ollama serve` running
with the model already pulled (`ollama pull llama3.2:1b`, etc.). A
candidate whose key is missing or server unreachable is skipped, not
fatal.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SRC_DIR, _BENCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engram.llm.client import GeminiClient  # noqa: E402
from clients import ClaudeClient, KimiClient, MercuryClient, OllamaClient  # noqa: E402
from prompts import extract_real_prompts  # noqa: E402


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def _build_candidates() -> dict[str, callable]:
    """Return {name: zero-arg factory}. Factories raise if the key is missing."""
    return {
        "gemini-3-flash": lambda: GeminiClient(),
        "claude-haiku-4-5": lambda: ClaudeClient("claude-haiku-4-5"),
        "claude-sonnet-5": lambda: ClaudeClient("claude-sonnet-5"),
        "claude-opus-4-8": lambda: ClaudeClient("claude-opus-4-8"),
        "mercury-2": lambda: MercuryClient("mercury-2"),
        "kimi-k3": lambda: KimiClient("kimi-k3"),
        "ollama-llama3.2-1b": lambda: OllamaClient("llama3.2:1b"),
        "ollama-llama3.2-3b": lambda: OllamaClient("llama3.2:3b"),
        "ollama-qwen2.5-1.5b": lambda: OllamaClient("qwen2.5:1.5b"),
    }


# Which client method each prompt is benchmarked through, matching the real
# pipeline call site (threat.py / response.py use generate(); tagging.py
# uses generate_json()).
_PROMPT_METHOD = {
    "threat_assessment": "generate",
    "dialogue_generation": "generate",
    "memory_tagging": "generate_json",
}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_calls(client, method: str, prompt: str, runs: int) -> list[float]:
    """Return a list of wall-clock durations (seconds), one per run.

    Does one untimed warmup call first (absorbs connection setup) so the
    measured runs reflect steady-state latency, not cold-start.
    """
    call = getattr(client, method)
    call(prompt)  # warmup, untimed

    durations = []
    for _ in range(runs):
        start = time.perf_counter()
        call(prompt)
        durations.append(time.perf_counter() - start)
    return durations


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=8, help="timed calls per prompt (default: 8)")
    parser.add_argument(
        "--candidates", type=str, default=None,
        help="comma-separated candidate names to run (default: all)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="optional path to write raw per-run results as CSV",
    )
    args = parser.parse_args()

    all_candidates = _build_candidates()
    names = (
        [n.strip() for n in args.candidates.split(",")]
        if args.candidates else list(all_candidates)
    )
    for n in names:
        if n not in all_candidates:
            parser.error(f"unknown candidate '{n}'. choices: {', '.join(all_candidates)}")

    prompts = extract_real_prompts()
    print(f"Extracted {len(prompts)} real pipeline prompts: {', '.join(prompts)}\n")

    csv_rows: list[dict] = []
    results: dict[str, dict[str, list[float]]] = {}

    for name in names:
        try:
            client = all_candidates[name]()
        except Exception as exc:  # noqa: BLE001 — missing key, bad client, etc.
            print(f"[{name}] SKIPPED - {exc}")
            continue

        print(f"[{name}]")
        results[name] = {}
        for prompt_name, prompt_text in prompts.items():
            method = _PROMPT_METHOD[prompt_name]
            try:
                durations = _time_calls(client, method, prompt_text, args.runs)
            except Exception as exc:  # noqa: BLE001
                print(f"    {prompt_name:<20} FAILED - {exc}")
                continue

            results[name][prompt_name] = durations
            p50 = statistics.median(durations)
            p95 = _percentile(durations, 95)
            print(f"    {prompt_name:<20} p50={p50 * 1000:7.1f}ms  p95={p95 * 1000:7.1f}ms  (n={len(durations)})")

            for d in durations:
                csv_rows.append({"candidate": name, "prompt": prompt_name, "seconds": d})
        print()

    if args.csv and csv_rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["candidate", "prompt", "seconds"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Raw results written to {args.csv}")

    # Summary — total per-turn latency estimate (sum of the two generate()
    # calls that sit on the critical path every turn: threat + dialogue).
    print("Estimated per-turn latency (threat_assessment + dialogue_generation, p50):")
    for name, prompt_results in results.items():
        if "threat_assessment" in prompt_results and "dialogue_generation" in prompt_results:
            total = (
                statistics.median(prompt_results["threat_assessment"])
                + statistics.median(prompt_results["dialogue_generation"])
            )
            print(f"    {name:<20} {total * 1000:7.1f}ms")


if __name__ == "__main__":
    main()

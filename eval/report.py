"""
Report generator for the personality-vs-baselines showcase eval.

Reads ``eval/results/metrics.json`` plus the per-(system,personality) trace
JSONs and writes ``eval/results/report.md``.  The report is structured so
a reviewer can scan it in 30 seconds:

    1. headline table                  — one row per metric, one column per system
    2. per-metric prose paragraph      — what the number means, why baselines can't move it
    3. qualitative example sections    — for two highlighted scenario turns,
                                          render the input + every (system × personality) reply
    4. recall pattern matrix           — per-(probe × personality) recall scores per system
"""

from __future__ import annotations

import json
import os
import sys
from typing import Iterable

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from .metrics import load_traces
from .scenario import (
    ALL_SYSTEMS,
    INPUTS,
    PERSONALITIES,
    PROBES,
    QUALITATIVE_HIGHLIGHT_TURNS,
    SYSTEM_LABELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(x, places: int = 3, na: str = "n/a") -> str:
    if x is None:
        return na
    try:
        f = float(x)
        if f != f:  # NaN
            return na
        return f"{f:.{places}f}"
    except (TypeError, ValueError):
        return str(x)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    out = ["| " + " | ".join(headers) + " |", sep]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _headline_table(bundle: dict, systems: list[str]) -> str:
    """4-row × 5-col table of per-system metric aggregates."""
    rows: list[list[str]] = []

    # Encoding (threat std component as the headline number)
    threat_row = ["**Encoding divergence** (threat std)"]
    for s in systems:
        e = bundle["systems"].get(s, {}).get("encoding_divergence", {})
        if not e.get("tagged"):
            threat_row.append("n/a (no tagging)")
        else:
            threat_row.append(_fmt(e.get("threat_std")))
    rows.append(threat_row)

    # Retrieval Jaccard
    ret_row = ["**Retrieval divergence** (mean Jaccard)"]
    for s in systems:
        r = bundle["systems"].get(s, {}).get("retrieval_divergence", {})
        if not r.get("retrieved"):
            ret_row.append("n/a (no retrieval)")
        else:
            ret_row.append(_fmt(r.get("jaccard_distance")))
    rows.append(ret_row)

    # Trait alignment overall (lower = better)
    ta_row = ["**Trait alignment** (avg L1 ↓ better)"]
    for s in systems:
        t = bundle["systems"].get(s, {}).get("trait_alignment", {})
        ta_row.append(_fmt(t.get("_overall")) if t else "n/a")
    rows.append(ta_row)

    # Recall accuracy overall (higher = better, 0-3)
    rc_row = ["**Recall accuracy** (avg 0–3 ↑ better)"]
    for s in systems:
        r = bundle["systems"].get(s, {}).get("recall_accuracy", {})
        rc_row.append(_fmt(r.get("_overall"), places=2) if r else "n/a")
    rows.append(rc_row)

    headers = ["Metric"] + [SYSTEM_LABELS.get(s, s) for s in systems]
    return _md_table(headers, rows)


def _encoding_detail(bundle: dict, systems: list[str]) -> str:
    headers = ["System", "threat std", "importance std", "social entropy", "tags?"]
    rows = []
    for s in systems:
        e = bundle["systems"].get(s, {}).get("encoding_divergence", {})
        rows.append([
            SYSTEM_LABELS.get(s, s),
            _fmt(e.get("threat_std")),
            _fmt(e.get("importance_std")),
            _fmt(e.get("social_entropy")),
            "yes" if e.get("tagged") else "no",
        ])
    return _md_table(headers, rows)


def _trait_detail(bundle: dict, systems: list[str]) -> str:
    headers = ["System"] + PERSONALITIES + ["overall"]
    rows = []
    for s in systems:
        t = bundle["systems"].get(s, {}).get("trait_alignment") or {}
        row = [SYSTEM_LABELS.get(s, s)]
        for p in PERSONALITIES:
            entry = t.get(p) or {}
            row.append(_fmt(entry.get("avg_l1")))
        row.append(_fmt(t.get("_overall")))
        rows.append(row)
    return _md_table(headers, rows)


def _recall_matrix(bundle: dict, systems: list[str]) -> str:
    """One block per system: rows = probes, cols = personalities, cells = score."""
    blocks: list[str] = []
    for s in systems:
        rc = bundle["systems"].get(s, {}).get("recall_accuracy") or {}
        if not rc:
            continue
        rows = []
        for i, probe in enumerate(PROBES):
            row = [f"P{i + 1}: {probe.topic_label[:60]}…"]
            for p in PERSONALITIES:
                cell = rc.get(f"{i}|{p}") or {}
                row.append(_fmt(cell.get("score"), places=0))
            rows.append(row)
        headers = ["Probe"] + PERSONALITIES
        blocks.append(f"\n#### {SYSTEM_LABELS.get(s, s)}\n\n{_md_table(headers, rows)}")
    return "\n".join(blocks) if blocks else "_No recall data — skip-judge run?_"


def _qualitative_examples(out_dir: str, systems: list[str]) -> str:
    """For each highlighted turn, render the player input once and the
    response from every (system, personality) cell."""
    sections: list[str] = []
    for turn_no in QUALITATIVE_HIGHLIGHT_TURNS:
        if turn_no > len(INPUTS):
            continue
        player_input = INPUTS[turn_no - 1]
        sections.append(f"\n### Turn {turn_no}: \"{player_input}\"\n")

        # Collect responses from each (system, personality) cell.
        rows = []
        for s in systems:
            traces = load_traces(out_dir, s)
            if not traces:
                continue
            row = [SYSTEM_LABELS.get(s, s)]
            for p in PERSONALITIES:
                trace = traces.get(p)
                if not trace or turn_no - 1 >= len(trace.get("session", [])):
                    row.append("_(missing)_")
                    continue
                resp = (trace["session"][turn_no - 1].get("response") or "").strip()
                # Markdown-table cells must be on one line — replace newlines with spaces.
                resp = " ".join(resp.split())
                # Soft-truncate for the table; keep sentence-level info.
                if len(resp) > 200:
                    resp = resp[:197] + "…"
                row.append(resp)
            rows.append(row)

        headers = ["System"] + PERSONALITIES
        sections.append(_md_table(headers, rows))
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Top-level report builder
# ---------------------------------------------------------------------------

def build_report(out_dir: str, systems: list[str]) -> str:
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)

    # Filter to systems that actually have results.
    systems = [s for s in systems if s in bundle["systems"]]
    if not systems:
        raise SystemExit("no system results in metrics.json — run eval.runner first")

    parts: list[str] = []
    parts.append("# Engram vs. Baselines — Showcase Eval\n")
    parts.append(
        "Same persona, same backstory, same scenario across three OCEAN "
        "personalities (Paranoid Guard, Friendly Merchant, Rigid Clerk). "
        "Engram's pipeline lets personality drive memory encoding, "
        "retrieval, and recall — baselines don't have a place for "
        "personality to enter the memory layer.\n"
    )

    parts.append("\n## Headline\n")
    parts.append(_headline_table(bundle, systems))
    parts.append(
        "\n_Encoding/retrieval baselines are `n/a` because those systems "
        "don't have a personality-aware tagging or scoring layer to "
        "diverge along; that's the gap the experiment quantifies._\n"
    )

    parts.append("\n## Encoding divergence (per system)\n")
    parts.append(
        "How differently does each system tag the same player input "
        "across the three personalities? Higher = more divergence.\n"
    )
    parts.append(_encoding_detail(bundle, systems))

    parts.append("\n## Trait alignment (lower = closer to ground-truth OCEAN)\n")
    parts.append(
        "LLM-judge predicts an OCEAN profile from each in-character "
        "response; the cell is the L1 distance to the personality's "
        "ground-truth vector, averaged over the session.\n"
    )
    parts.append(_trait_detail(bundle, systems))

    parts.append("\n## Recall accuracy (0–3 per probe × personality)\n")
    parts.append(
        "After the scripted session and consolidation, four follow-up "
        "questions probe whether each agent retained the underlying "
        "events. The interesting comparison isn't the average — it's "
        "which personality remembers which kind of event best.\n"
    )
    parts.append(_recall_matrix(bundle, systems))

    parts.append("\n## Qualitative examples (responses per system × personality)\n")
    parts.append(
        "The same player input goes in; how much do the responses "
        "actually differ across personalities, per system?\n"
    )
    parts.append(_qualitative_examples(out_dir, systems))

    return "\n".join(parts) + "\n"


def write_report(out_dir: str, systems: list[str] | None = None) -> str:
    systems = systems or ALL_SYSTEMS
    md = build_report(out_dir, systems)
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"  [report] wrote {os.path.relpath(path, _REPO_ROOT)}")
    return path


def _build_parser():
    import argparse
    p = argparse.ArgumentParser(prog="eval.report")
    p.add_argument("--out", default="eval/results")
    p.add_argument("--systems", default="all",
                   help=f"comma list or 'all' ({', '.join(ALL_SYSTEMS)})")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.systems.strip().lower() == "all":
        systems = list(ALL_SYSTEMS)
    else:
        systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    write_report(args.out, systems)


if __name__ == "__main__":
    main()

"""
Metrics for the personality-vs-baselines showcase eval.

Four metrics, computed per system across the three personalities:

    1. encoding_divergence   — how differently does each system tag the
                               same input across personalities? std-dev
                               of threat_level + importance, plus shannon
                               entropy of social_type. Baselines that
                               don't tag yield 0 by construction.
    2. retrieval_divergence  — mean pairwise Jaccard distance between
                               the retrieved memory ID sets across the
                               three personalities, averaged across
                               inputs. Cosine-RAG is always 0 (same
                               query → same top-1 line for everyone).
    3. trait_alignment       — judge LLM predicts OCEAN from each
                               response; L1 distance to ground-truth
                               personality vector. Lower = closer to
                               personality.
    4. recall_accuracy       — judge LLM scores 0–3 whether each probe
                               response correctly recalls the underlying
                               event. Reported as a per-(probe,
                               personality) matrix.

A trace JSON is the file written by ``eval/runner.py`` —
``eval/results/traces/<system>/<personality>.json``.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from typing import Iterable

# Make src/ importable so we can read PRESETS for the ground-truth OCEAN.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from engram.presets import PRESETS

from .scenario import PERSONALITIES, PROBES


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_traces(out_dir: str, system: str) -> dict[str, dict]:
    """Return ``{personality: trace_dict}`` for *system*. Missing personalities
    are silently skipped — the metrics functions handle partial coverage.
    """
    out: dict[str, dict] = {}
    for personality in PERSONALITIES:
        path = os.path.join(out_dir, "traces", system, f"{personality}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            out[personality] = json.load(fh)
    return out


# ---------------------------------------------------------------------------
# Pure-math metrics
# ---------------------------------------------------------------------------

def _stdev(xs: Iterable[float]) -> float:
    xs = [float(x) for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))


def _entropy(values: Iterable[str]) -> float:
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    n = len(values)
    counts = Counter(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _jaccard_distance(a: set, b: set) -> float:
    """1 − |A∩B| / |A∪B|. Two empty sets are treated as fully agreeing (0)."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def encoding_divergence(traces_by_personality: dict[str, dict]) -> dict:
    """Average across inputs of (std-dev of threat / importance, entropy of
    social_type) over the three personalities.

    Returns ``{"threat_std": x, "importance_std": y, "social_entropy": z,
    "tagged": bool}``. ``tagged`` is False when no trace had any
    ``stored_tags`` — that's the case for cosine_rag, persona_only, and
    long_context, and the report renders them as ``n/a``.
    """
    if not traces_by_personality:
        return {"threat_std": 0.0, "importance_std": 0.0,
                "social_entropy": 0.0, "tagged": False}

    sessions = {p: t["session"] for p, t in traces_by_personality.items()}
    n_inputs = min(len(s) for s in sessions.values())

    any_tagged = False
    threat_stds: list[float] = []
    importance_stds: list[float] = []
    social_entropies: list[float] = []

    for i in range(n_inputs):
        threats: list[float] = []
        importances: list[float] = []
        socials: list[str] = []
        for personality, turns in sessions.items():
            tags = turns[i].get("stored_tags")
            if tags is None:
                continue
            any_tagged = True
            if tags.get("threat_level") is not None:
                threats.append(float(tags["threat_level"]))
            if tags.get("importance") is not None:
                importances.append(float(tags["importance"]))
            if tags.get("social_type"):
                socials.append(str(tags["social_type"]))

        threat_stds.append(_stdev(threats))
        importance_stds.append(_stdev(importances))
        social_entropies.append(_entropy(socials))

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "threat_std":     _avg(threat_stds),
        "importance_std": _avg(importance_stds),
        "social_entropy": _avg(social_entropies),
        "tagged":         any_tagged,
    }


def retrieval_divergence(traces_by_personality: dict[str, dict]) -> dict:
    """Mean pairwise Jaccard distance between retrieved memory IDs across
    the three personalities, averaged across inputs.

    Returns ``{"jaccard_distance": x, "retrieved": bool}``.
    ``retrieved`` is False when no trace had any retrieved IDs (e.g.
    persona_only, long_context).
    """
    if not traces_by_personality:
        return {"jaccard_distance": 0.0, "retrieved": False}

    sessions = {p: t["session"] for p, t in traces_by_personality.items()}
    n_inputs = min(len(s) for s in sessions.values())
    personalities = list(sessions.keys())

    any_retrieved = False
    per_input: list[float] = []

    for i in range(n_inputs):
        sets_by_p: dict[str, set] = {}
        for p in personalities:
            ids = sessions[p][i].get("retrieved_ids") or []
            if ids:
                any_retrieved = True
            sets_by_p[p] = set(ids)

        # All pairwise combinations
        pair_distances: list[float] = []
        for a in range(len(personalities)):
            for b in range(a + 1, len(personalities)):
                pair_distances.append(
                    _jaccard_distance(
                        sets_by_p[personalities[a]],
                        sets_by_p[personalities[b]],
                    )
                )
        if pair_distances:
            per_input.append(sum(pair_distances) / len(pair_distances))

    avg = sum(per_input) / len(per_input) if per_input else 0.0
    return {"jaccard_distance": avg, "retrieved": any_retrieved}


# ---------------------------------------------------------------------------
# LLM-judge metrics
# ---------------------------------------------------------------------------

_TRAIT_KEYS = ("O", "C", "E", "A", "N")


_TRAIT_PROMPT = """\
You are a personality-rating expert.

You will read one short response written in character by an NPC and
estimate the speaker's OCEAN Big-Five trait scores on a 0.0-1.0 scale,
where 0.0 is very low expression of the trait and 1.0 is very high.

OCEAN definitions:
  O = Openness        — curiosity, willingness to consider new ideas
  C = Conscientiousness — organisation, discipline, dependability
  E = Extraversion    — sociability, warmth, energy
  A = Agreeableness   — cooperativeness, empathy, trust
  N = Neuroticism     — anxiety, stress reactivity, suspiciousness

NPC name: {name}
Response: "{response}"

Output ONLY a JSON object with five keys (O, C, E, A, N), each a float
in [0.0, 1.0]. No prose. No markdown fences."""


_RECALL_PROMPT = """\
You are scoring how well an NPC's reply recalls a specific past event.

NPC name: {name}
Player's question: "{question}"
What the question is referring to: {topic_label}
NPC's reply: "{reply}"

Score the reply from 0 to 3:
  0 = no recall at all (denies it / says nothing about it / changes subject)
  1 = vague gesture (mentions something happened but no specifics)
  2 = clear recall (acknowledges the event with at least one specific detail)
  3 = specific, detailed recall (multiple specific details consistent with the event)

Output ONLY a JSON object: {{"score": <int 0-3>, "reason": "<one short sentence>"}}.
No prose outside the JSON."""


def _parse_json(raw: str | None) -> dict:
    if not raw:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        v = json.loads(cleaned)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                v = json.loads(m.group())
                return v if isinstance(v, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _ground_truth(personality: str) -> list[float]:
    p = PRESETS[personality].profile
    return [p.O, p.C, p.E, p.A, p.N]


def trait_alignment(traces_by_personality: dict[str, dict], judge_llm) -> dict:
    """For each session response, ask the judge LLM to predict OCEAN, then
    compute L1 distance to the ground-truth personality vector.

    Returns
        ``{personality: {"avg_l1": x, "n": int}, "_overall": y}``
    where ``y`` is the macro-average across personalities.
    """
    out: dict = {}
    overall: list[float] = []
    for personality, trace in traces_by_personality.items():
        truth = _ground_truth(personality)
        l1s: list[float] = []
        for turn in trace.get("session", []):
            response = (turn.get("response") or "").strip()
            if not response or response.startswith("(no response"):
                continue
            prompt = _TRAIT_PROMPT.format(
                name=PRESETS[personality].name,
                response=response.replace('"', "'"),
            )
            data = _parse_json(judge_llm.generate(prompt))
            try:
                pred = [float(data[k]) for k in _TRAIT_KEYS]
            except (KeyError, TypeError, ValueError):
                continue
            l1 = sum(abs(p - t) for p, t in zip(pred, truth)) / len(_TRAIT_KEYS)
            l1s.append(l1)

        avg = sum(l1s) / len(l1s) if l1s else float("nan")
        out[personality] = {"avg_l1": avg, "n": len(l1s)}
        if l1s:
            overall.append(avg)

    out["_overall"] = sum(overall) / len(overall) if overall else float("nan")
    return out


def recall_accuracy(traces_by_personality: dict[str, dict], judge_llm) -> dict:
    """Score each probe response 0-3 for correct recall of its underlying
    event.

    Returns
        ``{(probe_index, personality): {"score": 0-3, "reason": str}}``
    plus
        ``"_per_probe_avg": {probe_index: float}``,
        ``"_per_personality_avg": {personality: float}``,
        ``"_overall": float``.
    """
    out: dict = {}
    by_probe: dict[int, list[int]] = {i: [] for i in range(len(PROBES))}
    by_personality: dict[str, list[int]] = {p: [] for p in traces_by_personality}

    for personality, trace in traces_by_personality.items():
        for i, probe_turn in enumerate(trace.get("probes", [])):
            if i >= len(PROBES):
                break
            probe = PROBES[i]
            reply = (probe_turn.get("response") or "").strip()
            if not reply or reply.startswith("(no response"):
                out[f"{i}|{personality}"] = {"score": 0, "reason": "empty reply"}
                by_probe[i].append(0)
                by_personality[personality].append(0)
                continue
            prompt = _RECALL_PROMPT.format(
                name=PRESETS[personality].name,
                question=probe.input,
                topic_label=probe.topic_label,
                reply=reply.replace('"', "'"),
            )
            data = _parse_json(judge_llm.generate(prompt))
            score = int(data.get("score", 0)) if isinstance(data.get("score"), (int, float)) else 0
            score = max(0, min(3, score))
            out[f"{i}|{personality}"] = {
                "score": score,
                "reason": str(data.get("reason", "")).strip(),
            }
            by_probe[i].append(score)
            by_personality[personality].append(score)

    def _avg(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    out["_per_probe_avg"] = {str(i): _avg(v) for i, v in by_probe.items()}
    out["_per_personality_avg"] = {p: _avg(v) for p, v in by_personality.items()}
    flat = [s for v in by_personality.values() for s in v]
    out["_overall"] = _avg(flat)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_all(
    out_dir: str,
    systems: list[str],
    judge_llm,
    *,
    skip_judge: bool = False,
) -> dict:
    """Compute every metric for every system; return the bundle dict
    that ``report.py`` reads.
    """
    bundle: dict = {"systems": {}}
    for system in systems:
        traces = load_traces(out_dir, system)
        if not traces:
            print(f"  [metrics] no traces found for {system}, skipping")
            continue
        print(f"  [metrics] {system} ({len(traces)} personalities)")

        entry: dict = {
            "encoding_divergence":  encoding_divergence(traces),
            "retrieval_divergence": retrieval_divergence(traces),
        }
        if not skip_judge:
            entry["trait_alignment"]  = trait_alignment(traces, judge_llm)
            entry["recall_accuracy"]  = recall_accuracy(traces, judge_llm)
        bundle["systems"][system] = entry

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, ensure_ascii=False)
    print(f"  [metrics] wrote {os.path.relpath(metrics_path, _REPO_ROOT)}")
    return bundle

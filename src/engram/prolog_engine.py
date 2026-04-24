"""
PrologEngine — wraps pyswip for Engram memory strength computation and
dynamic fact management.

Degrades gracefully when pyswip / SWI-Prolog is not available: all strength
computations fall back to a pure-Python implementation of the same formulae.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import EventTags, OCEANProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional pyswip import
# ---------------------------------------------------------------------------
try:
    from pyswip import Prolog as _SWIProlog  # type: ignore

    _PYSWIP_AVAILABLE = True
except (ImportError, OSError):
    _SWIProlog = None  # type: ignore
    _PYSWIP_AVAILABLE = False
    warnings.warn(
        "[PrologEngine] pyswip or SWI-Prolog is not available. "
        "Memory strength will be computed in pure Python and contradiction "
        "checking will fall back to string matching.",
        RuntimeWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Built-in OCEAN strength rules (written to the .pl file on init)
# ---------------------------------------------------------------------------

_BUILTIN_RULES = """\
:- set_prolog_flag(verbose, silent).
% Engram built-in OCEAN memory-strength rules — auto-generated.

apply_openness_filter(Novelty, O, FN) :- FN is Novelty * (0.3 + 0.7 * O).
apply_social_weight(Social, E, WS) :- WS is Social * (0.2 + 0.8 * E).
apply_threat_sensitivity(Threat, N, TS) :-
    Raw is Threat * (0.3 + 1.2 * N),
    (Raw > 1.0 -> TS = 1.0 ; TS = Raw).
apply_goal_relevance(Goal, C, GW) :- GW is Goal * (0.2 + 0.8 * C).
apply_cooperative_bias(Social, Emotion, A, CW) :-
    (Emotion >= 0 -> CW is Social * (0.5 + 0.5 * A)
     ; CW is Social * (0.5 + 0.5 * (1.0 - A))).
compute_strength(FN, WS, TS, GW, CW, SR, S) :-
    Raw is (FN*0.15 + WS*0.15 + TS*0.25 + GW*0.2 + CW*0.1 + SR*0.15),
    (Raw > 1.0 -> S = 1.0 ; S = Raw).
process_memory(O, C, E, A, N, EV, SS, TL, GR, NL, SR, Strength) :-
    apply_openness_filter(NL, O, FN),
    apply_social_weight(SS, E, WS),
    apply_threat_sensitivity(TL, N, TS),
    apply_goal_relevance(GR, C, GW),
    apply_cooperative_bias(SS, EV, A, CW),
    compute_strength(FN, WS, TS, GW, CW, SR, Strength).
"""


# ---------------------------------------------------------------------------
# Pure-Python fallback
# ---------------------------------------------------------------------------

def _python_process_memory(profile: "OCEANProfile", tags: "EventTags") -> float:
    """Compute memory encoding strength without SWI-Prolog.

    Mirrors the Prolog ``process_memory/12`` predicate exactly.
    """
    o, c, e, a, n = profile.vector()
    ev = tags.emotion_valence
    ss = tags.social_score  # property on EventTags
    tl = tags.threat_level
    gr = tags.goal_relevance
    nl = tags.novelty_level
    sr = tags.self_relevance

    fn = nl * (0.3 + 0.7 * o)
    ws = ss * (0.2 + 0.8 * e)
    ts = min(1.0, tl * (0.3 + 1.2 * n))
    gw = gr * (0.2 + 0.8 * c)
    cw = ss * (0.5 + 0.5 * a) if ev >= 0 else ss * (0.5 + 0.5 * (1.0 - a))
    raw = fn * 0.15 + ws * 0.15 + ts * 0.25 + gw * 0.2 + cw * 0.1 + sr * 0.15
    return min(1.0, raw)


# ---------------------------------------------------------------------------
# Simple string-based contradiction helper (fallback)
# ---------------------------------------------------------------------------

def _string_check_contradiction(
    new_fact: str, existing_facts: list[str]
) -> tuple[bool, str]:
    """Detect contradictions via string matching when Prolog is unavailable.

    Only works for the canonical ``fact(NpcId, Subject, Predicate, Object)``
    schema.  Returns ``(True, conflicting_fact)`` or ``(False, '')``.
    """
    import re

    pattern = re.compile(
        r"fact\(\s*(\w+)\s*,\s*'?([^',)]+)'?\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*\)"
    )
    m = pattern.match(new_fact.strip())
    if m is None:
        return False, ""

    npc_id, subject, predicate, _obj = m.groups()

    for existing in existing_facts:
        em = pattern.match(existing.strip())
        if em is None:
            continue
        e_npc, e_subject, e_predicate, e_obj = em.groups()
        if (
            e_npc == npc_id
            and e_subject == subject
            and e_predicate == predicate
            and e_obj != _obj
        ):
            return True, existing
    return False, ""


# ---------------------------------------------------------------------------
# PrologEngine
# ---------------------------------------------------------------------------

class PrologEngine:
    """Wraps SWI-Prolog via pyswip for Engram.

    Responsibilities:
    - Write/load built-in OCEAN strength rules to a ``.pl`` file.
    - Compute personality-parameterised memory strength via Prolog (or Python
      fallback).
    - Dynamic ``assertz`` / ``retract`` for conversational facts.
    - Contradiction detection against the existing fact base.
    """

    def __init__(self, rules_path: str | None = None) -> None:
        """Initialise the engine.

        Parameters
        ----------
        rules_path:
            Path to the ``.pl`` file used to persist built-in rules and
            asserted facts.  If *None*, a temporary path inside the current
            working directory is used.
        """
        self.available: bool = _PYSWIP_AVAILABLE
        self._rules_path: str = rules_path or os.path.join(
            os.getcwd(), "engram_rules.pl"
        )

        # Write the built-in rules so they survive process restarts.
        self._write_rules()

        if self.available:
            self._prolog: "_SWIProlog | None" = _SWIProlog()
            try:
                self._prolog.consult(self._rules_path)
            except Exception as exc:
                logger.warning(
                    "[PrologEngine] Failed to consult %s: %s — falling back to Python.",
                    self._rules_path,
                    exc,
                )
                self.available = False
                self._prolog = None
        else:
            self._prolog = None

    # ------------------------------------------------------------------
    # Rules file management
    # ------------------------------------------------------------------

    def _write_rules(self) -> None:
        """Write built-in rules to ``self._rules_path`` (creates dirs as needed)."""
        dir_part = os.path.dirname(self._rules_path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(self._rules_path, "w", encoding="utf-8") as fh:
            fh.write(_BUILTIN_RULES)

    # ------------------------------------------------------------------
    # Memory strength computation
    # ------------------------------------------------------------------

    def process_memory(self, profile: "OCEANProfile", tags: "EventTags") -> float:
        """Return the encoding strength for a memory given *profile* and *tags*.

        Delegates to SWI-Prolog ``process_memory/12`` when available;
        otherwise uses the pure-Python fallback.

        Returns
        -------
        float in [0.0, 1.0]
        """
        if not self.available or self._prolog is None:
            return _python_process_memory(profile, tags)

        o, c, e, a, n = profile.vector()
        ev = float(tags.emotion_valence)
        ss = float(tags.social_score)
        tl = float(tags.threat_level)
        gr = float(tags.goal_relevance)
        nl = float(tags.novelty_level)
        sr = float(tags.self_relevance)

        goal = (
            f"process_memory("
            f"{o:.6f}, {c:.6f}, {e:.6f}, {a:.6f}, {n:.6f}, "
            f"{ev:.6f}, {ss:.6f}, {tl:.6f}, {gr:.6f}, {nl:.6f}, {sr:.6f}, "
            f"Strength)"
        )

        try:
            results = list(self._prolog.query(goal))
            if results:
                return float(results[0]["Strength"])
        except Exception as exc:
            logger.warning(
                "[PrologEngine] process_memory query failed: %s — using Python fallback.",
                exc,
            )

        # Prolog call failed at runtime — fall back silently.
        return _python_process_memory(profile, tags)

    # ------------------------------------------------------------------
    # Dynamic fact manipulation
    # ------------------------------------------------------------------

    def assert_fact(self, fact_str: str) -> None:
        """Assert *fact_str* into the Prolog runtime.

        ``fact_str`` must be a valid Prolog term string (without trailing
        period), e.g. ``"fact(rico, sofia, knows, ally)"``.

        No-op when Prolog is unavailable.
        """
        cleaned = fact_str.rstrip(". \t\n")
        if self._prolog is not None:
            try:
                self._prolog.assertz(cleaned)
            except Exception as exc:
                logger.warning("[PrologEngine] assertz('%s') failed: %s", cleaned, exc)

    def retract_fact(self, fact_str: str) -> None:
        """Retract *fact_str* from the Prolog runtime.

        No-op when Prolog is unavailable.
        """
        cleaned = fact_str.rstrip(". \t\n")
        if self._prolog is not None:
            try:
                self._prolog.retract(cleaned)
            except Exception as exc:
                logger.warning("[PrologEngine] retract('%s') failed: %s", cleaned, exc)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, goal: str) -> list[dict]:
        """Run *goal* against the Prolog database.

        Returns a list of solution dicts (variable bindings).  Returns an
        empty list when Prolog is unavailable or the query fails.
        """
        if self._prolog is None:
            return []
        try:
            return list(self._prolog.query(goal))
        except Exception as exc:
            logger.warning("[PrologEngine] query('%s') failed: %s", goal, exc)
            return []

    # ------------------------------------------------------------------
    # Contradiction checking
    # ------------------------------------------------------------------

    def check_contradiction(
        self, new_fact: str, existing_facts: list[str]
    ) -> tuple[bool, str]:
        """Determine whether *new_fact* contradicts any fact in *existing_facts*.

        Strategy
        --------
        1. If Prolog is available and a ``contradicts/2`` rule is loaded,
           query it.
        2. Otherwise fall back to the built-in string-pattern matcher for the
           canonical ``fact(NpcId, Subject, Predicate, Object)`` schema.

        Returns
        -------
        ``(True, conflicting_fact_str)`` if a contradiction is found, else
        ``(False, '')``.
        """
        if self._prolog is not None:
            # Attempt a Prolog-native contradiction check via assertz + query.
            # We assert all existing facts temporarily, query, then clean up.
            # This lets complex rules (if any) fire.
            try:
                for ef in existing_facts:
                    self._prolog.assertz(ef.rstrip(". \t\n"))

                # Try contradicts/2 if the user has defined it.
                has_contradicts = list(
                    self._prolog.query(
                        f"current_predicate(contradicts/2)"
                    )
                )
                if has_contradicts:
                    cleaned_new = new_fact.rstrip(". \t\n")
                    solutions = list(
                        self._prolog.query(
                            f"contradicts({cleaned_new}, Conflict)"
                        )
                    )
                    if solutions:
                        conflict_str = str(solutions[0].get("Conflict", ""))
                        return True, conflict_str
                else:
                    # Fall back to string matching even when Prolog is available
                    # but contradicts/2 is absent.
                    result = _string_check_contradiction(new_fact, existing_facts)
                    return result

            except Exception as exc:
                logger.warning(
                    "[PrologEngine] check_contradiction failed: %s — using string fallback.",
                    exc,
                )
            finally:
                # Clean up temporarily asserted facts.
                for ef in existing_facts:
                    try:
                        self._prolog.retract(ef.rstrip(". \t\n"))
                    except Exception:
                        pass

        # No Prolog runtime — pure string matching.
        return _string_check_contradiction(new_fact, existing_facts)

    # ------------------------------------------------------------------
    # File consultation
    # ------------------------------------------------------------------

    def consult(self, path: str) -> None:
        """Load a Prolog source file into the runtime.

        No-op when Prolog is unavailable.
        """
        if self._prolog is None:
            return
        try:
            self._prolog.consult(path)
        except Exception as exc:
            logger.warning("[PrologEngine] consult('%s') failed: %s", path, exc)

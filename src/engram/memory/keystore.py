from __future__ import annotations

import logging
import math
import os
import re
from typing import TYPE_CHECKING

from ..config import KEY_MEMORY_PERCENTILE

if TYPE_CHECKING:
    from ..models import Memory, OCEANProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional pyswip import — degrade gracefully when SWI-Prolog is unavailable.
# ---------------------------------------------------------------------------
try:
    from pyswip import Prolog as _SWIProlog  # type: ignore

    _PROLOG_AVAILABLE = True
except (ImportError, OSError):
    _SWIProlog = None  # type: ignore
    _PROLOG_AVAILABLE = False
    logger.warning(
        "[KeyStore] pyswip or SWI-Prolog is not available. "
        "KeyStore will operate in list-only mode with no contradiction checking."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_atom(text: str) -> str:
    """Escape single quotes for use inside a Prolog atom string."""
    return text.replace("'", "\\'")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _personality_score(memory: "Memory", profile: "OCEANProfile") -> float:
    """Personality-weighted score — identical formula used in MemoryManager."""
    if not memory.embedding:
        return 0.0
    # rag_score is cosine vs the zero vector when we have no query; use
    # the memory's stored score as a proxy, or fall back to importance only.
    # During update() we call this without a query embedding, so we treat the
    # rag component as 1.0 (pure personality × importance ranking).
    rag = 1.0
    traits = ("O", "C", "E", "A", "N")
    eff = profile.effective
    ocean_sum = sum(
        min(5.0, (memory.tags.ocean.get(t, 3) / 5.0) / max(0.05, eff[t]))
        for t in traits
    )
    return (rag * 2) * ocean_sum * memory.tags.importance


# ---------------------------------------------------------------------------
# KeyStore
# ---------------------------------------------------------------------------

class KeyStore:
    """Manages the top-25% personality-weighted memories as Prolog facts.

    When pyswip / SWI-Prolog is unavailable the class degrades to a plain
    list store and ``check_contradiction`` always returns ``(False, '')``.
    """

    def __init__(self, pl_path: str) -> None:
        self.pl_path = pl_path
        self._key_memories: list["Memory"] = []
        self._fact_strings: list[str] = []   # asserted conversational facts

        if _PROLOG_AVAILABLE:
            self._prolog: "_SWIProlog | None" = _SWIProlog()
            for flag in (
                "set_prolog_flag(verbose_load, silent)",
                "set_prolog_flag(debug, false)",
                "nodebug",
            ):
                try:
                    list(self._prolog.query(flag))
                except Exception:
                    pass
            if os.path.exists(pl_path):
                try:
                    self._prolog.consult(pl_path)
                except Exception as exc:
                    logger.warning("[KeyStore] Failed to consult %s: %s", pl_path, exc)
        else:
            self._prolog = None

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        all_memories: list["Memory"],
        profile: "OCEANProfile",
    ) -> list["Memory"]:
        """Recompute which memories are key memories and rewrite the .pl file.

        Returns the promoted key memory list.
        """
        if not all_memories:
            self._key_memories = []
            self._write_pl()
            return []

        scored = sorted(
            all_memories,
            key=lambda m: _personality_score(m, profile),
            reverse=True,
        )

        cutoff = max(1, math.ceil(len(scored) * KEY_MEMORY_PERCENTILE))
        # KEY_MEMORY_PERCENTILE = 0.75 means *top 25%*, i.e. the top quarter
        # sits at or above the 75th percentile.
        top_n = max(1, math.ceil(len(scored) * (1.0 - KEY_MEMORY_PERCENTILE)))
        key = scored[:top_n]

        self._key_memories = key
        self._write_pl()

        return list(key)

    # ------------------------------------------------------------------
    # Contradiction checking
    # ------------------------------------------------------------------

    def check_contradiction(self, new_fact_str: str) -> tuple[bool, str]:
        """Check whether *new_fact_str* contradicts any existing Prolog fact.

        Returns ``(True, conflicting_fact)`` or ``(False, '')``.

        Without pyswip this always returns ``(False, '')``.
        """
        if self._prolog is None:
            return False, ""

        # Parse simple ``fact(NpcId, Subject, Predicate, Object)`` form.
        m = re.match(
            r"fact\(\s*(\w+)\s*,\s*'?([^',)]+)'?\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*\)",
            new_fact_str.strip(),
        )
        if m is None:
            return False, ""

        npc_id, subject, predicate, obj = m.groups()

        try:
            query = (
                f"fact({npc_id}, '{_safe_atom(subject)}', "
                f"'{_safe_atom(predicate)}', OldObj), "
                f"OldObj \\= '{_safe_atom(obj)}'."
            )
            results = list(self._prolog.query(query))
            if results:
                conflicting = (
                    f"fact({npc_id}, '{subject}', '{predicate}', "
                    f"'{results[0].get('OldObj', '?')}')"
                )
                return True, conflicting
        except Exception as exc:
            logger.warning("[KeyStore] check_contradiction query error: %s", exc)

        return False, ""

    # ------------------------------------------------------------------
    # Dynamic fact manipulation
    # ------------------------------------------------------------------

    def assert_fact(self, fact_str: str) -> None:
        """Assert *fact_str* into the Prolog runtime and track it locally."""
        cleaned = fact_str.rstrip(". \t\n")
        if cleaned not in self._fact_strings:
            self._fact_strings.append(cleaned)

        if self._prolog is not None:
            try:
                self._prolog.assertz(cleaned)
            except Exception as exc:
                logger.warning("[KeyStore] assertz('%s') failed: %s", cleaned, exc)

    def retract_fact(self, fact_str: str) -> None:
        """Retract *fact_str* from the Prolog runtime and remove it locally."""
        cleaned = fact_str.rstrip(". \t\n")
        if cleaned in self._fact_strings:
            self._fact_strings.remove(cleaned)

        if self._prolog is not None:
            try:
                self._prolog.retract(cleaned)
            except Exception as exc:
                logger.warning("[KeyStore] retract('%s') failed: %s", cleaned, exc)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_key_memory_texts(self) -> list[str]:
        """Return the text of all current key memories."""
        return [m.text for m in self._key_memories]

    # ------------------------------------------------------------------
    # Internal: write / reload Prolog file
    # ------------------------------------------------------------------

    def _write_pl(self) -> None:
        os.makedirs(os.path.dirname(self.pl_path) or ".", exist_ok=True)

        lines: list[str] = [
            "% Engram KeyStore — auto-generated, do not edit by hand.",
            "",
            "% key_memory(Id, Text).",
        ]
        for mem in self._key_memories:
            safe_id = _safe_atom(mem.id)
            safe_text = _safe_atom(mem.text)
            lines.append(f"key_memory('{safe_id}', '{safe_text}').")

        if self._fact_strings:
            lines.append("")
            lines.append("% fact(NpcId, Subject, Predicate, Object).")
            for fs in self._fact_strings:
                lines.append(f"{fs}.")

        lines.append("")

        with open(self.pl_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        if self._prolog is not None:
            try:
                self._prolog.consult(self.pl_path)
            except Exception as exc:
                logger.warning("[KeyStore] Failed to reload %s: %s", self.pl_path, exc)

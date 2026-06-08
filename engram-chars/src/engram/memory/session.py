from __future__ import annotations

from ..config import SESSION_WINDOW, EVICT_BATCH


class SessionMemory:
    """Rolling window of NPC–player dialogue turns.

    When the window overflows, the oldest ``batch`` turns are evicted and
    returned so the caller can generate a long-term summary from them.
    """

    def __init__(self, window: int = SESSION_WINDOW, batch: int = EVICT_BATCH) -> None:
        self.window = window
        self.batch = batch
        self.turns: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(self, player: str, npc: str) -> list[dict] | None:
        """Append a turn and evict if the window is full.

        Returns the evicted turns (for summary generation) when eviction
        occurs, otherwise returns ``None``.
        """
        self.turns.append({"player": player, "npc": npc})

        if len(self.turns) > self.window:
            evicted = self.turns[: self.batch]
            self.turns = self.turns[self.batch :]
            return evicted

        return None

    def get_turns(self) -> list[dict]:
        """Return the current turn window."""
        return list(self.turns)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "batch": self.batch,
            "turns": self.turns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SessionMemory:
        obj = cls(window=d.get("window", SESSION_WINDOW),
                  batch=d.get("batch", EVICT_BATCH))
        obj.turns = list(d.get("turns", []))
        return obj

from __future__ import annotations

import json
import os

from ..observability import bus


class LongTermMemory:
    """Persists OCEAN-biased summaries of evicted session turns to a JSON file.

    JSON schema: ``{"summaries": ["...", "..."]}``.
    The file is created automatically if it does not exist.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.summaries: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_summary(self, summary: str) -> None:
        """Append *summary* to the in-memory list and persist immediately."""
        self.summaries.append(summary)
        self._save()
        bus.emit("summary_added", summary=summary, total_summaries=len(self.summaries))

    def get_summaries(self) -> list[str]:
        """Return a copy of the current summaries list."""
        return list(self.summaries)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"summaries": self.summaries}, fh, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            # Create an empty store on disk so the file is always present.
            self.summaries = []
            self._save()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.summaries = list(data.get("summaries", []))
        except (json.JSONDecodeError, OSError):
            self.summaries = []

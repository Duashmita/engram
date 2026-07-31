"""
Extracts the exact prompts Engram's pipeline sends to the LLM, so the
latency benchmark measures real production prompts instead of toy text.

Each pipeline function normally calls llm.generate()/generate_json() as its
last step. We swap in a recording stub that captures the prompt instead of
calling out, so the real prompt-assembly logic in threat.py / response.py /
tagging.py never has to be duplicated (and can't drift from what's benched).
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from engram.models import EventTags, Memory, ThreatAssessment  # noqa: E402
from engram.pipeline.response import generate_response  # noqa: E402
from engram.pipeline.threat import assess_threat  # noqa: E402
from engram.llm.tagging import tag_event  # noqa: E402
from engram.presets import get_preset  # noqa: E402


class _RecordingLLM:
    """Stub LLM that records the prompt instead of calling out."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        self.last_prompt = prompt
        return ""

    def generate_json(self, prompt: str) -> dict:
        self.last_prompt = prompt
        return {}


class _EmptyMemoryManager:
    """Just enough of MemoryManager's surface for assess_threat's bookkeeping."""

    def retrieve_top_scored(self, embedding, top_k=3):
        return []


def _sample_memory() -> Memory:
    return Memory(
        id="sample_1",
        text="A stranger asked about the warehouse shipment last week.",
        tags=EventTags(
            emotion_valence=-0.3, social_type="conversation", threat_level=0.2,
            goal_relevance=0.6, novelty_level=0.5, self_relevance=0.7,
            importance=6, ocean={"O": 2, "C": 3, "E": 3, "A": 3, "N": 4},
        ),
        embedding=[0.0] * 8,
        source="session",
    )


def extract_real_prompts() -> dict[str, str]:
    """Return {name: prompt_text} for the pipeline's three LLM-call sites."""
    config = get_preset("guard")  # paranoid dock guard — high-N, representative NPC
    recorder = _RecordingLLM()

    # Threat assessment prompt — ambiguous input so the regex pattern floor
    # doesn't short-circuit before the LLM call.
    assess_threat(
        "I don't like where this is going, but I can't say why.",
        [0.0] * 8,
        config.profile,
        _EmptyMemoryManager(),
        recorder,
    )
    threat_prompt = recorder.last_prompt

    # Dialogue-generation prompt.
    generate_response(
        player_input="What do you know about the warehouse fire?",
        config=config,
        profile=config.profile,
        retrieved=[_sample_memory()],
        assessment=ThreatAssessment(is_threat=False, threat_magnitude=0.1, reasoning=""),
        mode="standard",
        history=[{"player": "Evening.", "npc": "Evening."}],
        llm=recorder,
        summaries=["He seemed nervous asking about the docks again."],
    )
    response_prompt = recorder.last_prompt

    # Memory-tagging prompt — used in consolidation and instinct-mode retrieval.
    tag_event(
        "Player: What do you know about the warehouse fire? | "
        f"{config.name}: Nothing. Why are you asking.",
        f"{config.name} interaction",
        recorder,
    )
    tag_prompt = recorder.last_prompt

    return {
        "threat_assessment": threat_prompt,
        "dialogue_generation": response_prompt,
        "memory_tagging": tag_prompt,
    }

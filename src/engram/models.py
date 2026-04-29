from __future__ import annotations
from dataclasses import dataclass, field
import time


@dataclass
class OCEANProfile:
    name: str
    O: float  # openness, 0–1
    C: float  # conscientiousness, 0–1
    E: float  # extraversion, 0–1
    A: float  # agreeableness, 0–1
    N: float  # neuroticism, 0–1
    # fight/flight temporary deltas — decay back to 0
    _dO: float = field(default=0.0, compare=False, repr=False)
    _dC: float = field(default=0.0, compare=False, repr=False)
    _dE: float = field(default=0.0, compare=False, repr=False)
    _dA: float = field(default=0.0, compare=False, repr=False)
    _dN: float = field(default=0.0, compare=False, repr=False)

    @property
    def effective(self) -> dict[str, float]:
        """Current OCEAN values with fight/flight deltas, clamped to [0,1]."""
        return {
            t: max(0.0, min(1.0, getattr(self, t) + getattr(self, f'_d{t}')))
            for t in ('O', 'C', 'E', 'A', 'N')
        }

    def apply_fight_flight(self, magnitude: float) -> None:
        self._dN = min(0.3, magnitude * 0.4)
        self._dA = -min(0.2, magnitude * 0.25)
        self._dE = -min(0.15, magnitude * 0.2)

    def decay(self, rate: float = 0.1) -> None:
        for attr in ('_dO', '_dC', '_dE', '_dA', '_dN'):
            v = getattr(self, attr)
            setattr(self, attr, 0.0 if abs(v) < 0.01 else v * (1 - rate))

    def vector(self) -> list[float]:
        e = self.effective
        return [e['O'], e['C'], e['E'], e['A'], e['N']]

    def to_dict(self) -> dict:
        return {'name': self.name, 'O': self.O, 'C': self.C,
                'E': self.E, 'A': self.A, 'N': self.N}

    @classmethod
    def from_dict(cls, d: dict) -> OCEANProfile:
        return cls(**{k: v for k, v in d.items()
                      if k in ('name', 'O', 'C', 'E', 'A', 'N')})

    def describe(self) -> str:
        """Human-readable personality description for LLM prompts."""
        labels = []
        for trait, val in zip(('Openness', 'Conscientiousness', 'Extraversion',
                                'Agreeableness', 'Neuroticism'), self.vector()):
            level = 'High' if val >= 0.65 else ('Low' if val <= 0.35 else 'Moderate')
            labels.append(f'{level} {trait}')
        return ', '.join(labels)


@dataclass
class EventTags:
    emotion_valence: float   # –1.0 to 1.0
    social_type: str         # solitude|conversation|cooperation|conflict
    threat_level: float      # 0.0 to 1.0
    goal_relevance: float    # 0.0 to 1.0
    novelty_level: float     # 0.0 to 1.0
    self_relevance: float    # 0.0 to 1.0
    importance: int = 5      # 1–10 LLM-judged
    ocean: dict = field(default_factory=lambda: {'O': 3, 'C': 3, 'E': 3, 'A': 3, 'N': 3})

    @property
    def social_score(self) -> float:
        return {'solitude': 0.0, 'conversation': 0.5,
                'cooperation': 0.75, 'conflict': 1.0}.get(self.social_type, 0.5)

    def to_vector(self) -> list[float]:
        return [self.emotion_valence, self.social_score, self.threat_level,
                self.goal_relevance, self.novelty_level, self.self_relevance]

    def to_dict(self) -> dict:
        return {
            'emotion_valence': self.emotion_valence,
            'social_type': self.social_type,
            'threat_level': self.threat_level,
            'goal_relevance': self.goal_relevance,
            'novelty_level': self.novelty_level,
            'self_relevance': self.self_relevance,
            'importance': self.importance,
            'ocean': self.ocean,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EventTags:
        return cls(**d)


@dataclass
class Memory:
    id: str
    text: str
    tags: EventTags
    embedding: list[float]
    source: str              # backstory|session
    timestamp: float = field(default_factory=time.time)
    score: float = 0.0       # personality-weighted retrieval score

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'text': self.text,
            'tags': self.tags.to_dict(),
            'embedding': self.embedding,
            'source': self.source,
            'timestamp': self.timestamp,
            'score': self.score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Memory:
        d = dict(d)
        d['tags'] = EventTags.from_dict(d['tags'])
        return cls(**d)


@dataclass
class ThreatAssessment:
    is_threat: bool
    threat_magnitude: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class NPCConfig:
    npc_id: str
    name: str
    persona: str
    backstory: list[str]
    profile: OCEANProfile
    initial_facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'npc_id': self.npc_id,
            'name': self.name,
            'persona': self.persona,
            'backstory': self.backstory,
            'profile': self.profile.to_dict(),
            'initial_facts': self.initial_facts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NPCConfig:
        d = dict(d)
        d['profile'] = OCEANProfile.from_dict(d['profile'])
        return cls(**d)

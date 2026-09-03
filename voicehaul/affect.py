"""Affect representation.

A compact 6-dimensional affect state, chosen as a tractable projection of the
kind of high-dimensional expression space that Hume's Expression Measurement
API returns (48+ categories). The projection is deliberate: metrics defined on
a low-dimensional, interpretable basis stay auditable, while the adapter layer
(see adapters.py) maps a real 48-dim measurement into this basis.
"""

from dataclasses import dataclass, replace
from typing import Dict, List

DIMS: List[str] = ["distress", "anger", "sadness", "joy", "calmness", "confusion"]

# Projection weights from the compact basis onto valence / arousal.
_VALENCE = {"distress": -0.8, "anger": -0.7, "sadness": -0.9,
            "joy": 1.0, "calmness": 0.5, "confusion": -0.3}
_AROUSAL = {"distress": 0.7, "anger": 1.0, "sadness": -0.4,
            "joy": 0.6, "calmness": -0.8, "confusion": 0.3}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass(frozen=True)
class Affect:
    distress: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    joy: float = 0.0
    calmness: float = 0.5
    confusion: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {d: getattr(self, d) for d in DIMS}

    def nudge(self, **deltas: float) -> "Affect":
        out = {}
        for d in DIMS:
            out[d] = _clip(getattr(self, d) + deltas.get(d, 0.0))
        return replace(self, **out)

    @property
    def valence(self) -> float:
        return sum(_VALENCE[d] * getattr(self, d) for d in DIMS)

    @property
    def arousal(self) -> float:
        return sum(_AROUSAL[d] * getattr(self, d) for d in DIMS)

    @property
    def negative_load(self) -> float:
        """Aggregate negative affect; the quantity an empathic agent should reduce."""
        return (0.45 * self.distress + 0.30 * self.anger
                + 0.15 * self.sadness + 0.10 * self.confusion)

    @classmethod
    def from_measurement(cls, scores: Dict[str, float]) -> "Affect":
        """Build from an arbitrary emotion->score mapping (e.g. a 48-dim readout).

        Unknown keys are routed through a coarse lexical map; this is the single
        place where a richer taxonomy is collapsed into the metric basis.
        """
        buckets = {d: [] for d in DIMS}
        alias = {
            "anxiety": "distress", "fear": "distress", "distress": "distress",
            "pain": "distress", "horror": "distress", "awkwardness": "distress",
            "anger": "anger", "contempt": "anger", "disgust": "anger",
            "annoyance": "anger", "irritation": "anger",
            "sadness": "sadness", "disappointment": "sadness", "grief": "sadness",
            "tiredness": "sadness", "boredom": "sadness",
            "joy": "joy", "amusement": "joy", "excitement": "joy",
            "satisfaction": "joy", "relief": "joy", "gratitude": "joy",
            "calmness": "calmness", "contentment": "calmness", "concentration": "calmness",
            "confusion": "confusion", "doubt": "confusion", "surprise": "confusion",
            "realization": "confusion",
        }
        for k, v in scores.items():
            tgt = alias.get(k.strip().lower())
            if tgt:
                buckets[tgt].append(float(v))
        return cls(**{d: _clip(max(vals) if vals else 0.0) for d, vals in buckets.items()})

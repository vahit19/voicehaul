"""Calibration drift: does quality decay as the session runs long?

Separates a policy that is uniformly mediocre from one that starts well and
degrades - two very different engineering problems that one mean hides."""

from typing import List, Tuple

from ..env import Episode
from .stats import bootstrap_ci, mean, ols_slope


def calibration_drift(episodes: List[Episode]) -> Tuple[float, Tuple[float, float]]:
    """Mean per-episode OLS slope of calibration vs turn index, in points/10 turns."""
    slopes = [ols_slope([t.calibration for t in ep.turns]) * 1000.0
              for ep in episodes if len(ep.turns) >= 5]
    return mean(slopes), bootstrap_ci(slopes)

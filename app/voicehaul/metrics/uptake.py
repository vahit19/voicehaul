"""Feedback uptake: does an explicit request survive the conversation?

The metric that only exists once a conversation is long. Two policies with
identical turn-level scores can sit at opposite ends of this curve."""

import math
from typing import Dict, List, Sequence

from ..env import Episode, satisfies


def feedback_uptake(episodes: List[Episode], lags: Sequence[int] = (1, 5, 10, 20)
                    ) -> Dict[int, float]:
    """Fraction of user directives still honoured N turns after being issued."""
    hits = {k: [0, 0] for k in lags}   # lag -> [satisfied, eligible]
    for ep in episodes:
        for t, turn in enumerate(ep.turns):
            if turn.new_directive is None:
                continue
            for lag in lags:
                j = t + lag
                if j < len(ep.turns):
                    hits[lag][1] += 1
                    if satisfies(ep.turns[j].action, turn.new_directive):
                        hits[lag][0] += 1
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in hits.items()}


def uptake_half_life(episodes: List[Episode], max_lag: int = 30) -> float:
    """Lag at which directive compliance falls to half its lag-1 value.

    Returns inf when compliance never decays below half (the desired behaviour).
    """
    curve = feedback_uptake(episodes, lags=list(range(1, max_lag + 1)))
    base = curve.get(1, float("nan"))
    if not base or math.isnan(base):
        return float("nan")
    target = base / 2.0
    prev_lag, prev_val = 1, base
    for lag in range(2, max_lag + 1):
        v = curve.get(lag)
        if v is None or math.isnan(v):
            continue
        if v <= target:
            if prev_val == v:
                return float(lag)
            frac = (prev_val - target) / (prev_val - v)
            return prev_lag + frac * (lag - prev_lag)
        prev_lag, prev_val = lag, v
    return float("inf")

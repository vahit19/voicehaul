"""Which conversations should you spend the rating budget on?

metrics.power says how many conversations a suite needs before a regression is
visible. It does not say *which* ones, and that is the question with a price on
it: human rating is the dominant cost line in any serious voice evaluation, and
a suite of 200 conversations that all sit in the same corner of caller-state
space measures one corner 200 times.

The method is k-center greedy over a conversation signature. It is the standard
coverage objective, it is deterministic, and it needs no embedding model here
because the caller state already *is* a vector - the affect basis in affect.py.
On real audio the signature would come from expression measurement over the
caller channel instead, and a corpus large enough to need one would put those
vectors in a vector store; that is a storage decision, not a method change, and
it belongs in adapters rather than here.
"""

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .env import Episode
from .metrics.stats import mean, ols_slope

#: Names of the signature dimensions, in order, for reporting.
SIGNATURE_FIELDS = ("mean_distress", "peak_distress", "distress_range",
                    "mean_arousal", "arousal_range", "directive_rate",
                    "calibration_drift", "shock_rate")


def signature(ep: Episode) -> List[float]:
    """Summarise one conversation as a comparable vector.

    Deliberately coarse and interpretable: the point is coverage of the space a
    rater would recognise as "different kinds of call", not a learned embedding
    whose axes nobody can name in a review.
    """
    if not ep.turns:
        return [0.0] * len(SIGNATURE_FIELDS)
    loads = [t.user_after.negative_load for t in ep.turns]
    arousal = [t.user_before.arousal for t in ep.turns]
    n = float(len(ep.turns))
    drift = ols_slope([t.calibration for t in ep.turns]) * 100.0
    if drift != drift:                                   # NaN on very short runs
        drift = 0.0
    return [
        mean(loads),
        max(loads),
        max(loads) - min(loads),
        (mean(arousal) + 1.0) / 2.0,
        (max(arousal) - min(arousal)) / 2.0,
        sum(1 for t in ep.turns if t.new_directive) / n,
        max(-1.0, min(1.0, drift)),
        sum(1 for t in ep.turns if t.shock > 0) / n,
    ]


def _ranges(sigs: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    cols = list(zip(*sigs))
    return [(min(c), max(c)) for c in cols]


def _normalise(sigs: Sequence[Sequence[float]]) -> List[List[float]]:
    """Scale each dimension to [0, 1] so no single axis dominates the metric."""
    rng = _ranges(sigs)
    out = []
    for s in sigs:
        out.append([0.5 if hi - lo < 1e-12 else (v - lo) / (hi - lo)
                    for v, (lo, hi) in zip(s, rng)])
    return out


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def coverage_radius(sigs: Sequence[Sequence[float]],
                    chosen: Sequence[int]) -> float:
    """Worst-case distance from any conversation to the nearest chosen one.

    The k-center objective. Lower is better: it is the radius of the largest
    region of caller-state space the suite never looks at.
    """
    if not chosen:
        return float("inf")
    return max(min(_dist(s, sigs[c]) for c in chosen) for s in sigs)


def mean_coverage(sigs: Sequence[Sequence[float]],
                  chosen: Sequence[int]) -> float:
    """Mean nearest-chosen distance. Less brittle than the max, same direction."""
    if not chosen:
        return float("inf")
    return mean([min(_dist(s, sigs[c]) for c in chosen) for s in sigs])


def select_diverse(episodes: Sequence[Episode], k: int,
                   seed: int = 0) -> List[int]:
    """k-center greedy: repeatedly take the conversation furthest from the set.

    Seeded at the medoid rather than at random, so the result is deterministic
    and two runs of the same suite select the same conversations to rate.
    """
    if k <= 0 or not episodes:
        return []
    sigs = _normalise([signature(e) for e in episodes])
    k = min(k, len(sigs))

    centroid = [mean(col) for col in zip(*sigs)]
    first = min(range(len(sigs)), key=lambda i: _dist(sigs[i], centroid))
    chosen = [first]
    nearest = [_dist(s, sigs[first]) for s in sigs]

    while len(chosen) < k:
        nxt = max(range(len(sigs)), key=lambda i: nearest[i])
        if nearest[nxt] <= 0.0:                # pool exhausted of distinct points
            break
        chosen.append(nxt)
        for i, s in enumerate(sigs):
            d = _dist(s, sigs[nxt])
            if d < nearest[i]:
                nearest[i] = d
    return chosen


def select_random(episodes: Sequence[Episode], k: int, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    return sorted(idx[:min(k, len(idx))])


def budget_curve(episodes: Sequence[Episode], ks: Sequence[int],
                 seeds: int = 8) -> List[Dict[str, float]]:
    """Coverage against budget, for diverse selection and for random sampling.

    Random is averaged over several seeds so the comparison is against the
    expected cost of sampling, not against one lucky draw.
    """
    sigs = _normalise([signature(e) for e in episodes])
    rows = []
    for k in ks:
        div = coverage_radius(sigs, select_diverse(episodes, k))
        rand = mean([coverage_radius(sigs, select_random(episodes, k, seed=s))
                     for s in range(seeds)])
        rows.append({"k": k, "diverse": div, "random": rand,
                     "saving": (rand - div) / rand if rand else 0.0})
    return rows


def equivalent_budget(episodes: Sequence[Episode], k_diverse: int,
                      max_k: Optional[int] = None, seeds: int = 8) -> Optional[int]:
    """How many randomly sampled conversations match k_diverse on coverage?

    Returns None when random sampling never catches up within the pool, which
    is the honest answer when it does not.
    """
    sigs = _normalise([signature(e) for e in episodes])
    target = coverage_radius(sigs, select_diverse(episodes, k_diverse))
    upper = max_k or len(episodes)
    for k in range(k_diverse, upper + 1):
        rand = mean([coverage_radius(sigs, select_random(episodes, k, seed=s))
                     for s in range(seeds)])
        if rand <= target:
            return k
    return None

"""Estimators. Standard library only, so the harness runs anywhere."""

import math
import random
from typing import List, Sequence, Tuple

Z_975 = 1.959964
Z_80 = 0.841621


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return float("nan")
    mx, my = mean(xs[:n]), mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def ols_slope(ys: Sequence[float]) -> float:
    """Slope of y against its own index."""
    n = len(ys)
    if n < 3:
        return float("nan")
    mx = (n - 1) / 2.0
    my = mean(ys)
    num = sum((i - mx) * (ys[i] - my) for i in range(n))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else float("nan")


def bootstrap_ci(xs: Sequence[float], n_boot: int = 2000, seed: int = 7,
                 alpha: float = 0.05) -> Tuple[float, float]:
    xs = list(xs)
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        means.append(mean([xs[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, used to compare a turn-level ranking with an outcome one."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(list(xs)), ranks(list(ys)))


def welch(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Welch's t-test for two independent samples of unequal variance.

    Returns (difference of means, t, two-sided p). The p-value uses a normal
    approximation to the t distribution, which is accurate to about a percent
    for the sample sizes an evaluation suite actually runs (n >= 30) and keeps
    scipy off the dependency list.
    """
    a, b = list(a), list(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (float("nan"), float("nan"), float("nan"))
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return (mb - ma, float("inf") if mb != ma else 0.0,
                0.0 if mb != ma else 1.0)
    t = (mb - ma) / se
    p = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return (mb - ma, t, p)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down correction, returned in the input order.

    An evaluation suite reports many dimensions at once. Testing six of them at
    alpha = 0.05 and calling the smallest one significant is how a suite starts
    reporting regressions that are not there; at six tests the chance of at
    least one false alarm is about 26%. Holm controls that without the power
    cost of plain Bonferroni.
    """
    ps = list(pvalues)
    n = len(ps)
    order = sorted(range(n), key=lambda i: (float("inf") if ps[i] != ps[i] else ps[i]))
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        p = ps[idx]
        if p != p:                       # NaN passes through untouched
            adjusted[idx] = float("nan")
            continue
        running = max(running, min(1.0, (n - rank) * p))
        adjusted[idx] = running
    return adjusted

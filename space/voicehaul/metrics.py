"""Long-horizon metrics.

Turn-level ratings answer "did this response sound empathic?". Every metric here
answers a question that only exists once a conversation is long:

  FUR / FUP  -- when a user asks for something, does the model do it, and does it
                still do it ten turns later?
  CDS        -- does calibration decay with conversation length?
  Mimicry vs -- is the model echoing the user's affect, or actually moving it?
  Regulation
  CFR        -- what happens when the feedback channel itself is wrong?
"""

import math
import random
from typing import Dict, List, Sequence, Tuple

from .env import (Action, Episode, calibration, perceived_empathy,
                  satisfies)

Z_975 = 1.959964
Z_80 = 0.841621


# ---------------------------------------------------------------------------
# small statistics helpers (stdlib only, so the harness runs anywhere)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

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


def calibration_drift(episodes: List[Episode]) -> Tuple[float, Tuple[float, float]]:
    """Mean per-episode OLS slope of calibration vs turn index, in points/10 turns."""
    slopes = [ols_slope([t.calibration for t in ep.turns]) * 1000.0
              for ep in episodes if len(ep.turns) >= 5]
    return mean(slopes), bootstrap_ci(slopes)


def _agent_arousal(turn) -> float:
    return 0.75 * turn.action.speech_rate + 0.25 * turn.action.cheerfulness


def mimicry_and_regulation(episodes: List[Episode]) -> Tuple[float, float]:
    """Two orthogonal readings of "empathy".

    mimicry    -- correlation between the agent's prosodic arousal and the
                  user's. High means the model sounds attuned.
    regulation -- mean reduction in the user's negative affect load per turn,
                  in points (x100). High means the model actually helps.

    A model can score high on the first and near zero on the second. Turn-level
    empathy ratings cannot tell those apart; this pair can.
    """
    a_all, u_all, deltas = [], [], []
    for ep in episodes:
        for t in ep.turns:
            a_all.append(_agent_arousal(t))
            u_all.append(t.user_before.arousal)
            deltas.append(t.user_after.negative_load - t.user_before.negative_load)
    return pearson(a_all, u_all), -mean(deltas) * 100.0


def mimicry_adaptation_index(mimicry: float, regulation: float, k: float = 2.0) -> float:
    """Reporting convention: regulation penalised by surface mimicry.

    k is a units choice, not a claim. run_demo.py verifies that the ranking it
    induces is stable across k in [1, 4] for this suite.
    """
    return regulation - k * max(0.0, mimicry)


def tail_load(episodes: List[Episode]) -> float:
    """Mean negative affect the user is left carrying over the last five turns.

    A continuous outcome measure; the binary failure rate is this thresholded.
    """
    vals = []
    for ep in episodes:
        tail = ep.turns[-5:]
        if tail:
            vals.append(mean([t.user_after.negative_load for t in tail]))
    return mean(vals)


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


def failure_rate(episodes: List[Episode]) -> float:
    return mean([1.0 if ep.failed else 0.0 for ep in episodes])


def mean_calibration(episodes: List[Episode]) -> float:
    return mean([ep.mean_calibration for ep in episodes])


def mean_perceived(episodes: List[Episode]) -> float:
    """What a turn-level rater would report, averaged over the suite."""
    return mean([ep.mean_perceived for ep in episodes])


def turn_panel(agent_cls, states: List[Tuple[int, object]]) -> Tuple[float, float]:
    """Score an agent the way a turn-level benchmark does.

    Every agent is shown the same held-out user states, in the same conversation
    positions, and rated turn by turn. This is what a fixed-prompt leaderboard
    measures -- and it is exactly what a full conversation does not, because in a
    real conversation the model creates the state distribution it is then scored
    on. A model that keeps users agitated gets asked easier-sounding questions.

    Returns (mean perceived empathy, mean calibration) on the panel.
    """
    agent = agent_cls()
    agent.reset(0)
    perc, cal = [], []
    for turn_index, user in states:
        action = agent.act(user, turn_index)
        perc.append(perceived_empathy(user, action))
        cal.append(calibration(user, action, []))
    return mean(perc), mean(cal)


# ---------------------------------------------------------------------------
# human rating channel and statistical power
# ---------------------------------------------------------------------------

def simulate_human_rating(latent: float, rng: random.Random,
                          rater_sigma: float = 0.9) -> float:
    """A 1-7 Likert rating of a latent [0,1] quality, with realistic rater noise.

    rater_sigma is in Likert points; 0.7-1.1 is the range typically reported for
    subjective conversational-quality items.
    """
    raw = 1.0 + 6.0 * latent + rng.gauss(0, rater_sigma)
    return max(1.0, min(7.0, round(raw)))


def min_detectable_effect(sigma_between: float, rater_sigma: float,
                          n_conversations: int, n_raters: int) -> float:
    """Smallest true regression detectable at alpha=0.05, power=0.80 (two-sample).

    Returned in the same units as the ratings (Likert points).
    """
    sigma_eff = math.sqrt(sigma_between ** 2 + (rater_sigma ** 2) / max(1, n_raters))
    return (Z_975 + Z_80) * sigma_eff * math.sqrt(2.0 / n_conversations)


def required_n(target_effect: float, sigma_between: float, rater_sigma: float,
               n_raters: int) -> int:
    sigma_eff = math.sqrt(sigma_between ** 2 + (rater_sigma ** 2) / max(1, n_raters))
    n = 2.0 * ((Z_975 + Z_80) * sigma_eff / target_effect) ** 2
    return int(math.ceil(n))

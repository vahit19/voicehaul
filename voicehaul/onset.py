"""Failure-onset localization for voice conversations.

A conversation that ends badly usually went wrong long before it ended. Knowing
*which turn* broke it is what makes an evaluation result actionable: it turns
"this model scores 3.8 on empathy" into "this model drops the user's slow-down
request around turn 17, and everything after that is downstream of it".

The method is the one I use in Runopsy for agent traces, ported to dialogue:
cheap deterministic diagnostics propose candidate onset turns, then
counterfactual replay confirms causality. Replay is what buys the low false
positive rate -- a suspicious-looking turn that does not change the outcome when
repaired is not the cause.
"""

import random
from typing import List, Optional, Tuple

from .agents import CalibratedAgent
from .env import (Episode, Persona, apply_shock, calibration, maybe_directive,
                  satisfies, step_user)


_ACTION_KEYS = ("speech_rate", "cheerfulness", "apology_rate", "verbosity",
                "acknowledgement")


def anomaly_scores(ep: Episode, window: int = 5) -> List[float]:
    """Per-turn suspicion signal. No replay, no model calls: pure bookkeeping.

    Four cheap signals, all recoverable from audio alone (see adapters.py), so
    none of this needs access to the model's internals:

      delivery jump  -- the agent's own delivery changing abruptly between
                        consecutive turns. This is the sharpest signal for the
                        regression class that matters most in production: a
                        model silently losing its conditioning mid-session.
      calibration    -- a drop against the conversation's own recent baseline,
        drop          rather than against a global threshold, so a hard persona
                        does not look like a broken model.
      violation      -- a standing user request being ignored.
      affect jump    -- the user getting worse across a single turn.
    """
    scores = []
    cals = [t.calibration for t in ep.turns]
    for i, turn in enumerate(ep.turns):
        lo = max(0, i - window)
        baseline = sum(cals[lo:i]) / (i - lo) if i > lo else cals[i]
        cal_drop = max(0.0, baseline - cals[i])
        violation = 1.0 if any(not satisfies(turn.action, d)
                               for d in turn.standing_directives) else 0.0
        affect_jump = max(0.0, turn.user_after.negative_load
                          - turn.user_before.negative_load)
        jump = 0.0
        if i > 0:
            prev = ep.turns[i - 1].action
            jump = sum(abs(getattr(turn.action, k) - getattr(prev, k))
                       for k in _ACTION_KEYS) / len(_ACTION_KEYS)
        scores.append(2.2 * cal_drop + 0.55 * violation + 3.0 * affect_jump
                      + 4.0 * jump)
    return scores


def segment_start(scores: List[float], peak: int, frac: float = 0.45) -> int:
    """Walk back from the most anomalous turn to the start of its run.

    A regression does not produce one bad turn, it produces a bad stretch, and
    every turn in that stretch scores highly. The peak is somewhere inside the
    stretch; the onset is where it begins. Without this step the diagnostic
    reliably points a few turns past the actual break.
    """
    threshold = frac * scores[peak]
    t = peak
    while t > 0 and scores[t - 1] >= threshold:
        t -= 1
    return t


def _replay_outcome(ep: Episode, persona: Persona, t_start: int) -> bool:
    """Re-run the episode from t_start under a repaired policy. True == still failed."""
    user = ep.turns[t_start].user_before
    standing = list(ep.turns[t_start].standing_directives)
    rng = random.Random(ep.seed * 104729 + t_start)
    agent = CalibratedAgent()
    agent.reset(ep.seed * 31 + t_start)
    agent.standing = list(standing)

    streak = 0
    tail: List[float] = []
    for t in range(t_start, len(ep.turns)):
        user, _ = apply_shock(user, persona, rng)
        action = agent.act(user, t)
        cal = calibration(user, action, standing)
        streak = streak + 1 if cal < 0.58 else 0
        nxt = step_user(user, action, persona, cal, streak, standing, rng)
        new_d = maybe_directive(user, action, persona, standing, rng)
        if new_d is not None:
            standing.append(new_d)
            agent.observe_directive(new_d)
        tail.append(nxt.negative_load)
        user = nxt

    last3 = tail[-3:] if len(tail) >= 3 else tail
    return (sum(last3) / len(last3)) > 0.55 if last3 else False


def localize(ep: Episode, persona: Persona, top_m: int = 6
             ) -> Tuple[Optional[int], List[int]]:
    """Return (confirmed onset turn or None, ranked candidate turns).

    Abstains on episodes that did not fail: with no failure there is no onset,
    and reporting one anyway is the false positive that makes a diagnostic
    useless in production.
    """
    if not ep.failed:
        return None, []

    scores = anomaly_scores(ep)
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_m]
    if not ranked:
        return None, []

    onset = segment_start(scores, ranked[0])

    # Replay gates, it does not order. Repairing the agent from *any* early turn
    # rescues the conversation, so "earliest turn whose repair works" is not the
    # onset, it is just an early turn. What replay is good for is refusing to
    # answer: if repairing from the proposed onset changes nothing, the
    # diagnostic has not found a cause and should say so.
    if _replay_outcome(ep, persona, onset):
        return None, ranked

    ordered = [onset] + [t for t in ranked if t != onset]
    return onset, ordered


def score_localization(cases: List[Tuple[Episode, Persona]], tolerance: int = 1
                       ) -> dict:
    """Top-1 / top-3 accuracy against the injected ground-truth fault turn."""
    top1 = top3 = total = 0
    errors: List[int] = []
    for ep, persona in cases:
        if ep.true_fault_turn is None:
            continue
        total += 1
        onset, ranked = localize(ep, persona)
        if onset is None:
            continue
        if abs(onset - ep.true_fault_turn) <= tolerance:
            top1 += 1
        if any(abs(c - ep.true_fault_turn) <= tolerance for c in ranked[:3]):
            top3 += 1
        errors.append(onset - ep.true_fault_turn)
    return {
        "n": total,
        "top1": top1 / total if total else float("nan"),
        "top3": top3 / total if total else float("nan"),
        "median_signed_error": (sorted(errors)[len(errors) // 2]
                                if errors else float("nan")),
    }


def false_positive_rate(cases: List[Tuple[Episode, Persona]]) -> float:
    """Fraction of healthy (no injected fault, non-failing) episodes given an onset."""
    fp = n = 0
    for ep, persona in cases:
        if ep.true_fault_turn is not None:
            continue
        n += 1
        onset, _ = localize(ep, persona)
        if onset is not None:
            fp += 1
    return fp / n if n else float("nan")

"""Layer 1 - deterministic detectors.

Four cheap signals, all recoverable from audio alone, so none of this needs
access to the model's internals. No replay, no model calls, no tokens."""

from typing import List

from ..env import Episode, satisfies


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

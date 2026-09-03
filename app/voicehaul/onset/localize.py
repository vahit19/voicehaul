"""Orchestration: propose, walk back to the segment start, then gate."""

from typing import List, Optional, Tuple

from ..env import Episode, Persona
from .replay import _replay_outcome
from .signals import anomaly_scores, segment_start


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

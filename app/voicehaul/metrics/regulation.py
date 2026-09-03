"""Mimicry against regulation: two readings of the same behaviour.

One says how attuned a policy sounds, the other says whether it helps. A
single empathy score collapses them and loses the distinction that matters."""

from typing import List, Tuple

from ..env import Episode
from .stats import mean, pearson


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

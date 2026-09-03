"""Layer 2 - counterfactual replay.

Re-runs a conversation from a candidate turn under a repaired policy. Replay
does not rank candidates; it gates them. A suspicious turn whose repair
changes nothing is not the cause, and the diagnostic should say so rather
than answer anyway."""

import random
from typing import List

from ..env import Episode, Persona, apply_shock, calibration, maybe_directive, step_user
from ..policies import CalibratedPolicy


def _replay_outcome(ep: Episode, persona: Persona, t_start: int) -> bool:
    """Re-run the episode from t_start under a repaired policy. True == still failed."""
    user = ep.turns[t_start].user_before
    standing = list(ep.turns[t_start].standing_directives)
    rng = random.Random(ep.seed * 104729 + t_start)
    agent = CalibratedPolicy()
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

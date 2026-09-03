"""Conversation-level outcomes, and the fixed-context turn panel.

turn_panel() scores a policy the way a prompt-set leaderboard does. The gap
between it and the outcome measures below is this package's headline result."""

from typing import List, Tuple

from ..env import (Episode, PERSONAS, calibration, caller_utterance,
                   perceived_empathy)
from .stats import mean


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


def failure_rate(episodes: List[Episode]) -> float:
    return mean([1.0 if ep.failed else 0.0 for ep in episodes])


def mean_calibration(episodes: List[Episode]) -> float:
    return mean([ep.mean_calibration for ep in episodes])


def mean_perceived(episodes: List[Episode]) -> float:
    """What a turn-level rater would report, averaged over the suite."""
    return mean([ep.mean_perceived for ep in episodes])


def turn_panel(agent_cls, states: List[Tuple[int, object]]) -> Tuple[float, float]:
    """Score a policy the way a turn-level benchmark does.

    Every policy is shown the same held-out caller states, in the same
    conversation positions, and rated turn by turn. This is what a fixed-prompt
    leaderboard measures - and it is exactly what a full conversation does not,
    because in a real conversation the model creates the state distribution it
    is later scored on. A model that keeps callers agitated gets asked
    easier-sounding questions.

    A policy that needs words rather than an affect vector - a real language
    model - is given the caller's utterance for that state. Without this the
    model receives an empty turn and the panel measures its response to
    nothing, which was a silent and complete corruption of these two numbers.

    Returns (mean perceived empathy, mean calibration) on the panel.
    """
    import random as _random

    agent = agent_cls()
    agent.reset(0)
    speaks = hasattr(agent, "hear")
    rng = _random.Random(20260904)
    persona = PERSONAS[0]

    perc, cal = [], []
    for turn_index, user in states:
        if speaks:
            agent.hear(caller_utterance(persona, user, max(1, turn_index),
                                        None, False, rng))
        action = agent.act(user, turn_index)
        perc.append(perceived_empathy(user, action))
        cal.append(calibration(user, action, []))
    return mean(perc), mean(cal)

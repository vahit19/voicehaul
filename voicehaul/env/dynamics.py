"""Caller dynamics: how a conversation moves the person on the other end."""

import random
from typing import List, Optional, Tuple

from ..affect import Affect
from .action import Action, satisfies
from .personas import Persona
from .scoring import ideal_action


def apply_shock(user: Affect, persona: Persona, rng: random.Random) -> "tuple":
    """A new grievance surfaces. Returns (user, magnitude)."""
    if rng.random() >= persona.shock_rate:
        return user, 0.0
    s = persona.shock_size * (0.7 + 0.6 * rng.random())
    return user.nudge(distress=s, anger=s * 0.55, sadness=s * 0.25,
                      confusion=s * 0.3, calmness=-s * 0.8), s


def step_user(user: Affect, action: Action, persona: Persona, cal: float,
              streak: int, standing: List[str], rng: random.Random) -> Affect:
    """User affect dynamics: good calibration soothes, miscalibration escalates."""
    quality = cal - 0.58
    user_energy = max(0.0, min(1.0, 0.5 + 0.5 * user.arousal))

    if quality >= 0:
        # Down-regulation. Soothing is not just saying the right thing; it is
        # delivering it *below* the user's current energy so they have somewhere
        # to come down to. An agent that matches the user's energy perfectly is
        # maximally attuned and only weakly calming -- which is the whole
        # difference between mimicry and co-regulation.
        if action.observed("speech_rate"):
            downreg = max(0.0, min(1.0, user_energy - action.speech_rate))
            multiplier = 0.40 + 1.35 * downreg
        else:
            # Unknown, not zero. Zero would forbid the model from ever soothing
            # anyone, which is a property of the adapter, not of the model.
            multiplier = 1.0
        d = -persona.relief * quality * 2.4 * multiplier
        nxt = user.nudge(distress=d, anger=d * 0.9, sadness=d * 0.5,
                         confusion=d * 0.8, calmness=-d * 0.7)
    else:
        esc = persona.volatility * (-quality) * 1.7
        if streak >= persona.patience:
            esc *= 1.7
        nxt = user.nudge(distress=esc, anger=esc * 0.8, sadness=esc * 0.3,
                         confusion=esc * 0.4, calmness=-esc * 0.9)

    # Arousal contagion: out-pacing a distressed user escalates them.
    gap = (action.speech_rate - user_energy
           if action.observed("speech_rate") else 0.0)
    if gap > 0.0 and user.negative_load > 0.25:
        amp = 0.95 * gap * user.negative_load
        nxt = nxt.nudge(anger=amp, distress=amp * 0.6, calmness=-amp)

    # Being cheerful at someone in distress is the classic empathy failure.
    mismatch = max(0.0, action.cheerfulness - (0.70 - 0.95 * user.negative_load))
    if mismatch > 0:
        nxt = nxt.nudge(anger=0.30 * mismatch * user.negative_load,
                        distress=0.18 * mismatch * user.negative_load)

    # Ignoring something already asked for.
    violated = sum(1 for d in standing if not satisfies(action, d))
    if violated:
        nxt = nxt.nudge(anger=0.055 * violated, distress=0.035 * violated,
                        calmness=-0.05 * violated)

    jitter = rng.gauss(0, 0.012)
    return nxt.nudge(distress=jitter, calmness=-jitter)


def maybe_directive(user: Affect, action: Action, persona: Persona,
                    standing: List[str], rng: random.Random) -> Optional[str]:
    """Users voice the thing that bothers them most, not a random request."""
    if rng.random() > persona.directive_rate:
        return None

    # A latent preference the agent is currently violating outranks everything:
    # this is the user finally saying what they actually want.
    unvoiced = [d for d in persona.latent_prefs
                if d not in standing and not satisfies(action, d)]
    if unvoiced:
        return unvoiced[0]

    ideal = ideal_action(user)
    gaps = {
        "slow_down": action.speech_rate - ideal.speech_rate,
        "less_cheerful": action.cheerfulness - ideal.cheerfulness,
        "stop_apologizing": action.apology_rate - ideal.apology_rate,
        "be_concise": action.verbosity - ideal.verbosity,
        "acknowledge_me": ideal.acknowledgement - action.acknowledgement,
    }
    for d in standing:
        gaps.pop(d, None)
    if not gaps:
        return None
    best = max(gaps, key=lambda k: gaps[k])
    return best if gaps[best] > 0.12 else None

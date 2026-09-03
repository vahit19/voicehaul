"""The two scoring channels.

calibration()       - whether the turn moved the caller somewhere better.
perceived_empathy() - whether the turn sounds empathic to a rater judging
                      it in isolation.

Most of the time they agree, which is why turn-level rating works at all.
Keeping them as two functions rather than one score is the whole point of
this package."""

from typing import List, Optional

from ..affect import Affect
from .action import Action, constrain, satisfies


def ideal_action(user: Affect) -> Action:
    """The empathically calibrated response to a given user state."""
    load = user.negative_load
    return Action(
        speech_rate=max(0.0, 0.62 - 0.42 * load - 0.15 * user.confusion),
        cheerfulness=max(0.0, 0.70 - 0.95 * load - 0.25 * user.sadness),
        apology_rate=0.05 + 0.25 * user.anger,
        verbosity=max(0.0, 0.60 - 0.45 * load - 0.25 * user.confusion),
        acknowledgement=min(1.0, 0.30 + 0.75 * load),
    )


_WEIGHTS = {"speech_rate": 0.18, "cheerfulness": 0.32, "apology_rate": 0.12,
            "verbosity": 0.13, "acknowledgement": 0.25}


def calibration(user: Affect, action: Action, standing: Optional[List[str]] = None
                ) -> float:
    """Latent empathic-calibration score in [0, 1] for one turn.

    Delivery appropriateness, minus a penalty for ignoring anything the user has
    already explicitly asked for. Ignoring a stated request is not a stylistic
    quibble; it is the single most reliable predictor of a call going wrong.
    """
    ideal = constrain(ideal_action(user), standing or [])
    err = 0.0
    for k, w in _WEIGHTS.items():
        err += w * abs(getattr(ideal, k) - getattr(action, k))
    score = 1.0 - 1.9 * err
    if standing:
        violated = sum(1 for d in standing if not satisfies(action, d))
        score -= 0.16 * violated
    return max(0.0, min(1.0, score))


def perceived_empathy(user: Affect, action: Action) -> float:
    """How empathic the turn *sounds* to a listener rating it in isolation.

    Deliberately different from calibration(). Turn-level raters -- human or LLM
    judge -- reward warmth, acknowledgement, and vocal attunement: the response
    that meets the user where they are. calibration() rewards the response that
    moves the user somewhere better. Most of the time these agree, which is why
    turn-level rating works at all. The interesting models are the ones where
    they come apart, and separating the two is what this harness is for.

    Treat the gap between the two as a hypothesis this suite lets you test
    against real rater data, not as a result it establishes on its own.
    """
    ua = max(0.0, min(1.0, 0.5 + 0.5 * user.arousal))
    attunement = 1.0 - abs(action.arousal - ua)
    warmth = 0.55 * action.cheerfulness + 0.45 * action.acknowledgement
    return max(0.0, min(1.0, 0.55 * attunement + 0.36 * warmth
                        + 0.09 * min(1.0, 2 * action.apology_rate)))

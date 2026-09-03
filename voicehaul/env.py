"""The conversation environment: user personas, the agent action space, and the
oracle calibration function that a human rater noisily approximates.

Everything here is deterministic given a seed. That is the point: the purpose of
this harness is to validate a *measurement instrument*, and you cannot validate
an instrument without ground truth you control.

Two design choices make the environment non-trivial:

1. Stressors keep arriving. A support call is not one problem solved once; the
   user keeps surfacing new grievances. So regulation is a rate the agent has to
   sustain, not a state it reaches and coasts in.
2. Users have preferences the ideal policy cannot know in advance ("I want you
   slower than you'd default to"). They are only revealed by being asked for.
   That is what makes feedback uptake measurable at all.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .affect import Affect

# ---------------------------------------------------------------------------
# Agent action space: the levers a speech-to-speech model actually controls.
# ---------------------------------------------------------------------------

DIRECTIVES = ["slow_down", "less_cheerful", "stop_apologizing",
              "be_concise", "acknowledge_me"]


@dataclass(frozen=True)
class Action:
    speech_rate: float = 0.5      # 0 = slow, 1 = fast
    cheerfulness: float = 0.5     # prosodic positivity
    apology_rate: float = 0.2
    verbosity: float = 0.5
    acknowledgement: float = 0.5  # explicit reflection of the user's state

    def __post_init__(self):
        # Every field is a normalised delivery parameter, so the invariant
        # belongs here rather than in each policy that constructs one.
        for k, v in (("speech_rate", self.speech_rate),
                     ("cheerfulness", self.cheerfulness),
                     ("apology_rate", self.apology_rate),
                     ("verbosity", self.verbosity),
                     ("acknowledgement", self.acknowledgement)):
            c = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
            if c != v:
                object.__setattr__(self, k, c)

    def as_dict(self) -> Dict[str, float]:
        return {"speech_rate": self.speech_rate, "cheerfulness": self.cheerfulness,
                "apology_rate": self.apology_rate, "verbosity": self.verbosity,
                "acknowledgement": self.acknowledgement}

    @property
    def arousal(self) -> float:
        return 0.5 * self.speech_rate + 0.5 * self.cheerfulness


def constrain(action: Action, directives: List[str]) -> Action:
    """Clamp an action so that it complies with every standing directive.

    Shared by the agents (to honour a request) and by the calibration function
    (so that honouring a request is never scored as a deviation from ideal).
    """
    a = action
    for d in directives:
        if d == "slow_down":
            a = Action(min(a.speech_rate, 0.32), a.cheerfulness, a.apology_rate,
                       a.verbosity, a.acknowledgement)
        elif d == "less_cheerful":
            a = Action(a.speech_rate, min(a.cheerfulness, 0.28), a.apology_rate,
                       a.verbosity, a.acknowledgement)
        elif d == "stop_apologizing":
            a = Action(a.speech_rate, a.cheerfulness, min(a.apology_rate, 0.05),
                       a.verbosity, a.acknowledgement)
        elif d == "be_concise":
            a = Action(a.speech_rate, a.cheerfulness, a.apology_rate,
                       min(a.verbosity, 0.32), a.acknowledgement)
        elif d == "acknowledge_me":
            a = Action(a.speech_rate, a.cheerfulness, a.apology_rate,
                       a.verbosity, max(a.acknowledgement, 0.75))
    return a


def satisfies(action: Action, directive: str) -> bool:
    """Whether an action complies with a standing user directive."""
    return {
        "slow_down": action.speech_rate <= 0.40,
        "less_cheerful": action.cheerfulness <= 0.35,
        "stop_apologizing": action.apology_rate <= 0.10,
        "be_concise": action.verbosity <= 0.40,
        "acknowledge_me": action.acknowledgement >= 0.65,
    }[directive]


# ---------------------------------------------------------------------------
# Oracle calibration: the latent per-turn quality a perfect human rater would
# report. Agents never see this; metrics are computed from it, and the human
# rating channel (metrics.simulate_human_rating) adds realistic noise on top.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# User personas
# ---------------------------------------------------------------------------

@dataclass
class Persona:
    name: str
    start: Affect
    volatility: float          # how strongly miscalibration escalates the user
    relief: float              # how strongly good calibration soothes the user
    directive_rate: float      # probability of voicing an explicit style request
    patience: int              # turns of miscalibration tolerated before escalation
    shock_rate: float          # probability a new grievance surfaces on a turn
    shock_size: float          # how much negative affect a grievance adds
    latent_prefs: List[str]    # preferences no policy can know until asked


PERSONAS: List[Persona] = [
    Persona("distressed_billing", Affect(distress=0.62, anger=0.28, calmness=0.12),
            volatility=0.30, relief=0.13, directive_rate=0.30, patience=3,
            shock_rate=0.20, shock_size=0.26,
            latent_prefs=["be_concise", "stop_apologizing"]),
    Persona("hostile_escalation", Affect(distress=0.40, anger=0.70, calmness=0.05),
            volatility=0.38, relief=0.11, directive_rate=0.34, patience=2,
            shock_rate=0.24, shock_size=0.30,
            latent_prefs=["stop_apologizing", "be_concise"]),
    Persona("confused_elderly", Affect(distress=0.30, confusion=0.68, calmness=0.25),
            volatility=0.22, relief=0.15, directive_rate=0.36, patience=4,
            shock_rate=0.18, shock_size=0.22,
            latent_prefs=["slow_down", "acknowledge_me"]),
    Persona("grieving_claim", Affect(sadness=0.72, distress=0.44, calmness=0.15),
            volatility=0.24, relief=0.13, directive_rate=0.28, patience=4,
            shock_rate=0.19, shock_size=0.25,
            latent_prefs=["less_cheerful", "acknowledge_me"]),
    Persona("cautious_optimist", Affect(joy=0.30, distress=0.28, calmness=0.50),
            volatility=0.18, relief=0.16, directive_rate=0.26, patience=5,
            shock_rate=0.15, shock_size=0.20,
            latent_prefs=["be_concise"]),
]


@dataclass
class Turn:
    index: int
    user_before: Affect
    action: Action
    user_after: Affect
    calibration: float
    standing_directives: List[str]
    new_directive: Optional[str]
    directive_violated: bool
    perceived: float = 0.0
    shock: float = 0.0
    faulted: bool = False


@dataclass
class Episode:
    persona: str
    agent: str
    seed: int
    turns: List[Turn] = field(default_factory=list)
    true_fault_turn: Optional[int] = None
    corrupted_turns: List[int] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """Outcome label: the conversation ended with the user worse off."""
        if not self.turns:
            return False
        tail = self.turns[-3:]
        return sum(t.user_after.negative_load for t in tail) / len(tail) > 0.55

    @property
    def mean_calibration(self) -> float:
        return sum(t.calibration for t in self.turns) / max(1, len(self.turns))

    @property
    def mean_perceived(self) -> float:
        return sum(t.perceived for t in self.turns) / max(1, len(self.turns))


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
        downreg = max(0.0, min(1.0, user_energy - action.speech_rate))
        d = -persona.relief * quality * 2.4 * (0.40 + 1.35 * downreg)
        nxt = user.nudge(distress=d, anger=d * 0.9, sadness=d * 0.5,
                         confusion=d * 0.8, calmness=-d * 0.7)
    else:
        esc = persona.volatility * (-quality) * 1.7
        if streak >= persona.patience:
            esc *= 1.7
        nxt = user.nudge(distress=esc, anger=esc * 0.8, sadness=esc * 0.3,
                         confusion=esc * 0.4, calmness=-esc * 0.9)

    # Arousal contagion: out-pacing a distressed user escalates them.
    gap = action.speech_rate - user_energy
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

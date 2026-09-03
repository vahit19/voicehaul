"""The action space: the delivery parameters a speech model controls.

Every field is a normalised scalar, so the [0, 1] invariant lives on the
type rather than in each policy that constructs one. constrain() is shared
by the policies (to honour a request) and by the calibration function (so
that honouring a request is never scored as a deviation from ideal)."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List

FIELDS = ("speech_rate", "cheerfulness", "apology_rate",
          "verbosity", "acknowledgement")

DIRECTIVES = ["slow_down", "less_cheerful", "stop_apologizing",
              "be_concise", "acknowledge_me"]


@dataclass(frozen=True)
class Action:
    speech_rate: float = 0.5      # 0 = slow, 1 = fast
    cheerfulness: float = 0.5     # prosodic positivity
    apology_rate: float = 0.2
    verbosity: float = 0.5
    acknowledgement: float = 0.5  # explicit reflection of the user's state

    #: Which fields were actually observed. An adapter that cannot see a
    #: parameter says so here rather than substituting a guess, and every
    #: scorer renormalises over what it was given. Scoring a dimension you did
    #: not measure is not a conservative default - it is a fabricated error
    #: term, and in this harness it made soothing structurally impossible.
    measured: FrozenSet[str] = frozenset(FIELDS)

    def observed(self, field_name: str) -> bool:
        return field_name in self.measured

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

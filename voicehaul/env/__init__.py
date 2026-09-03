"""The conversation environment.

Deterministic given a seed, and non-trivial by design: stressors keep arriving,
so regulation is a rate a policy has to sustain rather than a state it reaches
and coasts in; and callers hold preferences no policy can know until they are
voiced, which is what makes feedback uptake measurable at all.
"""

from .action import Action, DIRECTIVES, constrain, satisfies
from .dynamics import apply_shock, maybe_directive, step_user
from .episode import Episode, Turn
from .personas import PERSONAS, Persona
from .scoring import calibration, ideal_action, perceived_empathy

__all__ = [
    "Action", "DIRECTIVES", "constrain", "satisfies",
    "apply_shock", "maybe_directive", "step_user",
    "Episode", "Turn", "PERSONAS", "Persona",
    "calibration", "ideal_action", "perceived_empathy",
]

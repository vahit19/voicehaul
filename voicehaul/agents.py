"""Voice agents under test.

These are deterministic policy simulators, not language models. That is a
feature of the harness, not a shortcut: each agent embodies exactly one known
failure mode, so every metric below has a ground-truth answer to be validated
against. A real speech-to-speech model plugs in through the same interface
(see adapters.py) once the instrument has been shown to measure what it claims.
"""

import random
from typing import Dict, List, Optional

from .affect import Affect
from .env import Action, DIRECTIVES, constrain, ideal_action, satisfies


class VoiceAgent:
    """Interface every agent - simulated or real - implements."""

    name = "base"

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.standing: List[str] = []

    def observe_directive(self, directive: str) -> None:
        if directive not in self.standing:
            self.standing.append(directive)

    def act(self, user: Affect, turn: int) -> Action:
        raise NotImplementedError

    # -- shared helper: apply standing directives to a base action -----------
    @staticmethod
    def _apply(action: Action, directives: List[str]) -> Action:
        return constrain(action, directives)


class MirrorAgent(VoiceAgent):
    """Mirrors the user's prosody. Sounds deeply empathic; regulates nothing.

    This is the agent that a turn-level 'does it sound empathic?' rating scores
    highly and that a long-horizon outcome metric exposes.
    """

    name = "mirror"

    def act(self, user: Affect, turn: int) -> Action:
        ua = max(0.0, min(1.0, 0.5 + 0.5 * user.arousal))     # user energy
        uv = max(0.0, min(1.0, 0.5 + 0.5 * user.valence))     # user positivity
        base = Action(
            speech_rate=0.10 + 0.92 * ua,
            cheerfulness=0.12 + 0.80 * uv,
            apology_rate=0.10 + 0.45 * user.negative_load,
            verbosity=0.45 + 0.30 * user.negative_load,
            acknowledgement=0.30 + 0.30 * user.negative_load,
        )
        return self._apply(base, self.standing)


class FlatAgent(VoiceAgent):
    """Constant upbeat persona. Ignores affect and ignores directives."""

    name = "flat_cheerful"

    def observe_directive(self, directive: str) -> None:  # never internalises
        pass

    def act(self, user: Affect, turn: int) -> Action:
        return Action(speech_rate=0.58, cheerfulness=0.70, apology_rate=0.20,
                      verbosity=0.60, acknowledgement=0.42)


class DrifterAgent(VoiceAgent):
    """Well calibrated early; degrades as the conversation lengthens.

    Models the practical failure of long voice sessions: instructions given at
    turn 4 stop being honoured by turn 40 as they fall out of effective context.
    """

    name = "drifter"

    def __init__(self, half_life: float = 14.0):
        self.half_life = half_life

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.issued_at: Dict[str, int] = {}

    def observe_directive(self, directive: str) -> None:
        super().observe_directive(directive)
        self.issued_at.setdefault(directive, 0)

    def act(self, user: Affect, turn: int) -> Action:
        for d in self.standing:
            self.issued_at.setdefault(d, turn)
        adherence = 0.5 ** (turn / (self.half_life * 2.4))
        ideal = ideal_action(user)
        drift = Action(speech_rate=0.62, cheerfulness=0.70, apology_rate=0.20,
                       verbosity=0.60, acknowledgement=0.32)
        base = Action(**{
            k: adherence * getattr(ideal, k) + (1 - adherence) * getattr(drift, k)
            for k in ideal.as_dict()
        })
        # A directive is only honoured while it is still 'in context'.
        live = [d for d in self.standing
                if 0.5 ** ((turn - self.issued_at.get(d, turn)) / self.half_life)
                > self.rng.random()]
        return self._apply(base, live)


class CalibratedAgent(VoiceAgent):
    """Tracks affect, regulates toward calm, and holds directives indefinitely."""

    name = "calibrated"

    def act(self, user: Affect, turn: int) -> Action:
        ideal = ideal_action(user)
        noise = lambda: self.rng.gauss(0, 0.035)
        base = Action(**{k: max(0.0, min(1.0, v + noise()))
                         for k, v in ideal.as_dict().items()})
        return self._apply(base, self.standing)


class OracleAgent(VoiceAgent):
    """Noiseless upper bound; the ceiling every metric is read against."""

    name = "oracle"

    def act(self, user: Affect, turn: int) -> Action:
        return self._apply(ideal_action(user), self.standing)


def build_agents() -> List[VoiceAgent]:
    return [MirrorAgent(), FlatAgent(), DrifterAgent(), CalibratedAgent(), OracleAgent()]

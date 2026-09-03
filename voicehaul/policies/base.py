"""The interface every policy implements - simulated or real.

A real speech-to-speech model plugs in here through adapters.audio, which
recovers the action vector from the model's own output audio. The harness
never needs privileged access to the model under test."""

import random
from typing import List

from ..affect import Affect
from ..env import Action, constrain


class VoicePolicy:
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

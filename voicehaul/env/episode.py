"""Trace types: one turn, and one conversation."""

from dataclasses import dataclass, field
from typing import List, Optional

from ..affect import Affect
from .action import Action


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

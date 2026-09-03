"""Registries.

Policies, personas and gate dimensions are looked up by name so a suite can be
described entirely by a config file, and so a new policy or a new dimension is
one decorator rather than an edit to five call sites.
"""

from typing import Callable, Dict, Iterable, List

from .env import PERSONAS, Persona
from .policies import (CalibratedPolicy, DrifterPolicy, FlatPolicy,
                       MirrorPolicy, OraclePolicy, VoicePolicy)

_POLICIES: Dict[str, Callable[[], VoicePolicy]] = {}
_DIMENSIONS: Dict[str, "Dimension"] = {}


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

def register_policy(name: str, factory: Callable[[], VoicePolicy]) -> None:
    _POLICIES[name] = factory


def policy(name: str):
    """Decorator form, for a policy defined outside this package."""
    def deco(cls):
        register_policy(name, cls)
        return cls
    return deco


def get_policy(name: str) -> VoicePolicy:
    if name not in _POLICIES:
        raise KeyError("unknown policy {!r}; known: {}".format(
            name, ", ".join(sorted(_POLICIES))))
    return _POLICIES[name]()


def policy_names() -> List[str]:
    return list(_POLICIES)


for _n, _c in [("mirror", MirrorPolicy), ("flat_cheerful", FlatPolicy),
               ("drifter", DrifterPolicy), ("calibrated", CalibratedPolicy),
               ("oracle", OraclePolicy)]:
    register_policy(_n, _c)


# ---------------------------------------------------------------------------
# personas
# ---------------------------------------------------------------------------

def get_persona(name: str) -> Persona:
    for p in PERSONAS:
        if p.name == name:
            return p
    raise KeyError("unknown persona {!r}; known: {}".format(
        name, ", ".join(p.name for p in PERSONAS)))


def personas_for(names: Iterable[str]) -> List[Persona]:
    return [get_persona(n) for n in names]


# ---------------------------------------------------------------------------
# gate dimensions
# ---------------------------------------------------------------------------

class Dimension:
    """One measured quantity, with the direction that counts as an improvement.

    Two kinds, and the distinction is the point of this package:

    ``kind="conversation"``
        Measured on the conversations the policy itself produced. `measure`
        returns one value per conversation, which is what makes a two-sample
        test legitimate - conversations are the independent unit, turns are not.

    ``kind="panel"``
        Measured on held-out caller states that every policy answers, the way a
        fixed-prompt leaderboard works. `measure` returns one value per block of
        held-out states, so blocks are the independent unit.

    ``gating`` decides whether a dimension can block a release. A dimension whose
    sign is only interpretable next to another number - calibration drift, which
    can rise simply because it started low - is reported but does not vote.
    """

    def __init__(self, name: str, measure, higher_is_better: bool,
                 unit: str = "", blurb: str = "", kind: str = "conversation",
                 gating: bool = True):
        if kind not in ("conversation", "panel"):
            raise ValueError("kind must be 'conversation' or 'panel'")
        self.name = name
        self.measure = measure
        self.higher_is_better = higher_is_better
        self.unit = unit
        self.blurb = blurb
        self.kind = kind
        self.gating = gating

    #: Backwards-compatible alias for the conversation-level extractor.
    @property
    def per_episode(self):
        return self.measure

    def improvement(self, delta: float) -> float:
        """Signed so that positive always means better."""
        return delta if self.higher_is_better else -delta


def register_dimension(dim: Dimension) -> None:
    _DIMENSIONS[dim.name] = dim


def dimensions(kind: str = None, gating: bool = None) -> List[Dimension]:
    out = list(_DIMENSIONS.values())
    if kind is not None:
        out = [d for d in out if d.kind == kind]
    if gating is not None:
        out = [d for d in out if d.gating is gating]
    return out


def get_dimension(name: str) -> Dimension:
    return _DIMENSIONS[name]

"""Caller personas.

Each carries latent preferences no policy can know in advance - the thing
that makes feedback uptake measurable at all - plus the parameters that
govern how it escalates, how it is soothed, and how often a new grievance
surfaces."""

from dataclasses import dataclass
from typing import List

from ..affect import Affect


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

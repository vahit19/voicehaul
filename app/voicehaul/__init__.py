"""VoiceHaul - long-horizon evaluation and failure-onset diagnosis for
empathic voice agents.

A turn-level rating tells you whether a response sounded right. This measures
what only exists once a conversation is long: whether a model still honours what
the caller asked for twenty turns ago, whether its calibration decays, whether
it is regulating the caller's affect or mirroring it back, and which turn broke
a call that ended badly. Then it says how much human-rating budget is needed
before any of those answers means anything.

Core is standard library only. Inspect AI is an optional extra.
"""

from .config import SuiteConfig
from .env import Action, Episode, PERSONAS, Persona, Turn
from .policies import (CalibratedPolicy, DrifterPolicy, FlatPolicy,
                       MirrorPolicy, OraclePolicy, VoicePolicy)
from .runner import run_episode, run_suite

__version__ = "0.2.0"

__all__ = [
    "SuiteConfig", "Action", "Episode", "Persona", "Turn", "PERSONAS",
    "VoicePolicy", "MirrorPolicy", "FlatPolicy", "DrifterPolicy",
    "CalibratedPolicy", "OraclePolicy", "run_episode", "run_suite",
    "__version__",
]

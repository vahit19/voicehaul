"""Recovering the action vector from a model's own output audio.

The design property that makes this practical: the harness needs no
privileged access to the model under test. Both sides of the conversation
are scored from audio, so the same metrics apply to a model you own, one you
licence, and a competitor's public endpoint.
"""

from typing import Dict, Protocol

from ..affect import Affect
from ..env import Action


class ExpressionSource(Protocol):
    """Anything that turns a chunk of audio into an emotion-score mapping."""

    def measure(self, audio_path: str) -> Dict[str, float]:
        ...


class TranscriptSource(Protocol):
    def transcribe(self, audio_path: str) -> str:
        ...


# ---------------------------------------------------------------------------
# Recovering the action vector from the agent's own audio
# ---------------------------------------------------------------------------

_APOLOGY_MARKERS = ("sorry", "apologize", "apologise", "my apologies", "regret")
_ACK_MARKERS = ("i hear you", "that sounds", "i understand", "it makes sense",
                "i can see why", "that must")


def action_from_audio(audio_path: str, duration_s: float,
                      expression: ExpressionSource,
                      transcript: TranscriptSource) -> Action:
    """Estimate the agent's delivery parameters from its own output audio.

    speech_rate    -- words per second, normalised against a 2.2 wps reference.
    cheerfulness   -- prosodic positive affect from expression measurement.
    apology_rate   -- apology markers per utterance.
    verbosity      -- words per turn, normalised against a 45-word reference.
    acknowledgement-- explicit reflection markers per utterance.

    The two lexical features are deliberately crude and language-specific; for a
    multilingual suite they should be replaced by a small classifier trained on
    the annotation pipeline rather than by keyword lists.
    """
    text = transcript.transcribe(audio_path).lower()
    words = max(1, len(text.split()))
    scores = expression.measure(audio_path)
    agent_affect = Affect.from_measurement(scores)

    return Action(
        speech_rate=min(1.0, (words / max(0.5, duration_s)) / 2.2),
        cheerfulness=min(1.0, 0.7 * agent_affect.joy + 0.3 * max(0.0, agent_affect.valence)),
        apology_rate=min(1.0, sum(text.count(m) for m in _APOLOGY_MARKERS) / 1.0),
        verbosity=min(1.0, words / 45.0),
        acknowledgement=min(1.0, sum(text.count(m) for m in _ACK_MARKERS) / 1.0),
    )

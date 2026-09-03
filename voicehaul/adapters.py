"""Integration points for real systems.

The important design property of this harness: it needs no privileged access to
the model under test. Both sides of the conversation are scored from audio.

  - the user's affect trajectory comes from expression measurement on the user
    channel;
  - the agent's action (speech rate, prosodic positivity, apology rate,
    verbosity, acknowledgement) is recovered from expression measurement plus
    transcript statistics on the agent channel.

So the same metrics apply to a model you own, a model you licence, and a
competitor's public endpoint. Nothing below runs in the demo -- these are the
seams, kept honest and unmocked, where credentials and a real model go.
"""

from typing import Dict, List, Optional, Protocol

from .affect import Affect
from .env import Action


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


# ---------------------------------------------------------------------------
# Hume API seams
# ---------------------------------------------------------------------------

class HumeExpressionMeasurement:
    """Expression Measurement API -> the compact affect basis in affect.py.

    Requires HUME_API_KEY. The 48+ returned categories are collapsed by
    Affect.from_measurement; that projection is the one thing to re-derive
    empirically rather than hand-map, by regressing the compact basis on human
    ratings of the same clips.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client = None

    def measure(self, audio_path: str) -> Dict[str, float]:
        raise NotImplementedError(
            "Wire to the Hume Expression Measurement batch/stream endpoint. "
            "Return {emotion_name: score} for the prosody model."
        )


class HumeSpeechToSpeechAgent:
    """A real speech-to-speech model under test.

    Adapts a live voice session to the same act()/observe_directive() interface
    the simulated agents implement, so no metric code changes. Directives are not
    injected out of band: the simulated user speaks them, which is the point --
    we measure whether the model picks them up from the conversation itself.
    """

    name = "hume-s2s"

    def __init__(self, config_id: str, expression: ExpressionSource,
                 transcript: TranscriptSource):
        self.config_id = config_id
        self.expression = expression
        self.transcript = transcript

    def reset(self, seed: int) -> None:
        raise NotImplementedError("Open a session against the speech-to-speech API.")

    def observe_directive(self, directive: str) -> None:
        # Intentionally a no-op: a real model must hear the request in the audio.
        pass

    def act(self, user: Affect, turn: int) -> Action:
        raise NotImplementedError(
            "Synthesize the simulated user's turn, send it, receive the model's "
            "audio, then return action_from_audio(...) on the response."
        )


class HumanFeedbackChannel:
    """Human Feedback API as the rating channel.

    Replaces metrics.simulate_human_rating in a real run. The power analysis in
    run_demo.py is what sizes this call: it says how many conversations and how
    many raters per conversation are needed before a regression is detectable,
    which is a direct cost question.
    """

    def __init__(self, api_key: Optional[str] = None, n_raters: int = 3):
        self.api_key = api_key
        self.n_raters = n_raters

    def rate(self, audio_paths: List[str], rubric: str) -> List[float]:
        raise NotImplementedError(
            "Submit samples to the Human Feedback API and return per-sample "
            "aggregated scores on the given rubric."
        )

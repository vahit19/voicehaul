"""Hume API seams, kept honest and unmocked.

Nothing here runs in the demo. These are the three places credentials and a
real model go: an expression-measurement source, a speech-to-speech session,
and a human rating channel. metrics.power is what sizes the third one.
"""

from typing import Dict, List, Optional

from ..affect import Affect
from ..env import Action
from .audio import ExpressionSource, TranscriptSource


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


class HumeSpeechToSpeechPolicy:
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

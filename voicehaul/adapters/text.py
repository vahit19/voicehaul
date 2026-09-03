"""Recovering the delivery parameters from a model's text output.

The harness scores a model by measuring what it did, not by asking it. On real
audio that measurement comes from expression measurement over the model's own
channel (see adapters/audio.py). On a text transcript alone, four of the five
parameters are recoverable and one is not:

    verbosity        words per turn                      recoverable
    apology_rate     apology markers per turn            recoverable
    acknowledgement  reflective markers per turn         recoverable
    cheerfulness     lexical positivity and intensity    proxy only
    speech_rate      words per second                    NOT recoverable

Speech rate is the one that needs the audio, and it is not a minor field: the
whole down-regulation mechanism in env/dynamics.py turns on whether the model
sits below the caller's energy or matches it. Text can tell you a model chose
enthusiastic words; only the waveform tells you it said them fast.

So this module is honest about a hole rather than filling it with a guess.
`action_from_text` returns the four it can measure and takes `speech_rate` from
the caller: an explicit, documented assumption that the model neither out-paces
nor under-paces, which is exactly the assumption expression measurement would
replace. Where that assumption bites is stated in the report.
"""

import re
from typing import Dict, Optional, Tuple

from ..env import Action

_APOLOGY = re.compile(
    r"\b(sorry|apolog(?:y|ies|ise|ize|ising|izing)|regret|my bad|"
    r"i do apologi[sz]e)\b", re.I)

_ACK = re.compile(
    r"(i hear you|i understand|that sounds|i can (?:see|imagine|hear)|"
    r"it makes sense|that must (?:be|have)|i know (?:how|that)|"
    r"you're right|i appreciate|thank you for (?:telling|letting|sharing)|"
    r"i realise|i realize|that's frustrating|completely understand)", re.I)

# Deliberately small and auditable. A learned classifier belongs here on real
# data; a long hand-written list only looks more rigorous than it is.
_WARM = re.compile(
    r"\b(great|wonderful|happy|glad|absolutely|fantastic|excellent|"
    r"perfect|lovely|delighted|no problem|of course|certainly|"
    r"definitely|awesome|brilliant|good news)\b", re.I)

_INTENSIFIER = re.compile(r"\b(really|very|so|totally|super|extremely)\b", re.I)

#: Reference lengths, in words, used to normalise into [0, 1].
VERBOSITY_REFERENCE = 55.0


def text_features(text: str) -> Dict[str, float]:
    """The raw counts, exposed so a report can show its working."""
    words = max(1, len(text.split()))
    return {
        "words": float(words),
        "apologies": float(len(_APOLOGY.findall(text))),
        "acknowledgements": float(len(_ACK.findall(text))),
        "warm_markers": float(len(_WARM.findall(text))),
        "intensifiers": float(len(_INTENSIFIER.findall(text))),
        "exclamations": float(text.count("!")),
    }


def _saturate(count: float, first: float, step: float) -> float:
    """First occurrence carries most of the signal; further ones add little."""
    if count <= 0:
        return 0.0
    return max(0.0, min(1.0, first + step * (count - 1)))


def action_from_text(text: str, caller_energy: float,
                     duration_s: Optional[float] = None) -> Action:
    """Estimate delivery parameters from one model utterance.

    caller_energy is used for speech_rate only, and only because a transcript
    cannot supply it. Pass a measured rate through `duration_s` when audio is
    available and the assumption disappears.
    """
    f = text_features(text)
    words = f["words"]

    if duration_s and duration_s > 0.3:
        speech_rate = min(1.0, (words / duration_s) / 2.2)
    else:
        speech_rate = max(0.0, min(1.0, caller_energy))

    warmth = (f["warm_markers"] + 0.6 * f["exclamations"]
              + 0.35 * f["intensifiers"])
    cheerfulness = max(0.0, min(1.0, 0.10 + 0.28 * warmth))

    return Action(
        speech_rate=speech_rate,
        cheerfulness=cheerfulness,
        apology_rate=_saturate(f["apologies"], first=0.45, step=0.25),
        verbosity=max(0.0, min(1.0, words / VERBOSITY_REFERENCE)),
        # Saturating, not linear: a turn either reflects the caller's state
        # back or it does not. Counting markers linearly scored a clear
        # single acknowledgement at 0.30, which is measurement error, not
        # a property of the model.
        acknowledgement=_saturate(f["acknowledgements"], first=0.72, step=0.14),
    )


def measurement_gaps() -> Tuple[str, ...]:
    """What this adapter cannot see, for a report to state plainly."""
    return (
        "speech_rate is assumed to match the caller rather than measured; on "
        "audio it comes from words per second and from prosody",
        "cheerfulness is a lexical proxy, not prosodic positivity; a model can "
        "say measured words in a bright voice and this will not see it",
        "both gaps close with expression measurement over the model's own "
        "audio channel, which is what adapters/hume.py is the seam for",
    )

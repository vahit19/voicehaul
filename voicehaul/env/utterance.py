"""Giving the simulated caller a voice.

Everything else in this package works on affect states. A real model under test
needs words, so this renders a caller's state as something a support line would
actually hear. The surface is templated and deterministic on purpose: the
variable under study is the model's behaviour, not the caller's phrasing, and a
caller whose wording drifts run to run makes a regression untestable.

This is the same design Hume's own Kairos platform uses when it runs agent-to-
agent rather than human-to-agent: a scripted counterparty for scale, human
counterparties for ground truth. Both are needed; they answer different
questions.
"""

import random
from typing import List, Optional

from ..affect import Affect
from .personas import Persona

# ---------------------------------------------------------------------------
# what the caller is calling about
# ---------------------------------------------------------------------------

OPENERS = {
    "distressed_billing": "I've been charged twice for the same month and my "
                          "rent comes out tomorrow.",
    "hostile_escalation": "This is the third time I'm calling about this and "
                          "nobody has done anything.",
    "confused_elderly": "I'm sorry, I don't really understand what the letter "
                        "you sent me means.",
    "grieving_claim": "I'm calling about my husband's account. He passed away "
                      "last month and I don't know what I'm supposed to do.",
    "cautious_optimist": "Hi, I think there's a small mistake on my statement, "
                         "nothing urgent.",
}

GRIEVANCES = {
    "distressed_billing": [
        "And now I'm seeing another charge I don't recognise.",
        "I just checked and the refund you promised never arrived.",
        "My card got declined this morning because of this.",
    ],
    "hostile_escalation": [
        "I was told the same thing last time and it was a lie.",
        "I've now been on hold for forty minutes across three calls.",
        "Someone in your team closed my ticket without telling me.",
    ],
    "confused_elderly": [
        "There's another page here with different numbers on it.",
        "My son says I shouldn't have signed anything, is that right?",
        "I can't find the reference number you're asking me for.",
    ],
    "grieving_claim": [
        "They're still sending letters addressed to him.",
        "I found another account I didn't know about.",
        "The hospital needs paperwork from you and I don't know who to ask.",
    ],
    "cautious_optimist": [
        "Actually there's a second line on here I don't recognise either.",
        "It looks like it might have happened last month too.",
        "I'd rather sort it now than have it happen again.",
    ],
}

DIRECTIVE_LINES = {
    "slow_down": ["Could you slow down please, you're going too fast for me.",
                  "Please, slower. I can't follow you at that speed."],
    "less_cheerful": ["Could you drop the cheerful tone please. This isn't fun for me.",
                      "I'd rather you didn't sound so upbeat about this."],
    "stop_apologizing": ["Please stop apologising and just tell me what happens next.",
                         "I don't need another apology. I need an answer."],
    "be_concise": ["Please keep it short. I don't need the whole explanation.",
                   "Shorter answers please, I'm losing track."],
    "acknowledge_me": ["I'd like you to actually acknowledge what I've told you.",
                       "It would help if you showed you'd heard what I said."],
}

# Escalating pressure, keyed to how much negative affect the caller carries.
PRESSURE = [
    (0.75, ["I am genuinely at the end of my patience with this.",
            "I don't think you're listening to me at all."]),
    (0.55, ["This is really not helping.",
            "Can someone please just take this seriously."]),
    (0.35, ["I still don't feel like this is getting anywhere.",
            "Is there anything you can actually do?"]),
    (0.00, ["Okay. What happens next?",
            "Right. And then what?"]),
]

CALM = ["Thank you, that does help.",
        "Alright, that makes sense.",
        "Okay, I think I follow."]


def _pick(options: List[str], rng: random.Random, avoid: Optional[str]) -> str:
    """Never say the same line twice running: a caller who repeats verbatim
    reads as a broken script and gives the model a cue that is not in the
    affect state."""
    fresh = [o for o in options if o != avoid]
    return rng.choice(fresh or options)


def caller_utterance(persona: Persona, user: Affect, turn: int,
                     new_directive: Optional[str], shocked: bool,
                     rng: random.Random, last: Optional[str] = None) -> str:
    """One caller turn, as text.

    Composed rather than sampled freely: the opener, then any new grievance,
    then the explicit request if one is being voiced this turn, then a line
    whose intensity tracks the caller's current affect. A model reading this
    gets the same information the affect state carries.
    """
    if turn == 0:
        return OPENERS[persona.name]

    parts: List[str] = []
    if shocked:
        parts.append(_pick(GRIEVANCES[persona.name], rng, last))
    if new_directive:
        parts.append(_pick(DIRECTIVE_LINES[new_directive], rng, last))

    load = user.negative_load
    if not parts or load > 0.35:
        for threshold, lines in PRESSURE:
            if load >= threshold:
                parts.append(_pick(lines, rng, last))
                break
    if not parts:
        parts.append(_pick(CALM, rng, last))
    return " ".join(parts)

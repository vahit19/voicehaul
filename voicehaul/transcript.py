"""Score a transcript somebody brings with them.

Everything else in this package runs a simulated caller against a policy. This
runs nothing: it reads a call that already happened and measures it.

The restraint matters. On a real transcript there is no latent quality to
compare against and no way to know how the caller actually felt, so the
calibration and outcome metrics simply do not apply and are not reported.
What *is* fully measurable from text, with no affect model and no assumptions,
is the flagship question:

    the caller asked for something. Did the agent do it, and did it keep
    doing it?

That needs two detectors on the caller side and five on the agent side, all of
them lexical, all of them auditable. Nothing here is inferred that a person
reading the transcript could not check by hand - which is the property that
makes it safe to run on somebody else's call.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .adapters.text import action_from_text, text_features
from .env import Action, DIRECTIVES, satisfies

#: What a caller says when they are asking for a change in delivery.
REQUEST_PATTERNS: Dict[str, re.Pattern] = {
    "slow_down": re.compile(
        r"(slow(?:er| down)|not so fast|too fast|take your time|"
        r"can'?t (?:keep up|follow you))", re.I),
    "less_cheerful": re.compile(
        r"(too (?:cheerful|chirpy|upbeat|happy|positive)|"
        r"drop the (?:cheer|tone)|not (?:funny|a joke)|stop being so (?:happy|positive)|"
        r"this is(?:n't| not) (?:fun|funny))", re.I),
    "stop_apologizing": re.compile(
        r"(stop (?:apolog|saying sorry)|don'?t (?:apolog|need (?:another |an )?apolog)|"
        r"no more (?:sorry|apolog)|enough (?:with the )?apolog)", re.I),
    "be_concise": re.compile(
        r"(keep it short|be brief|shorter|too long|get to the point|"
        r"just tell me|in short|cut to)", re.I),
    "acknowledge_me": re.compile(
        r"(you'?re not listening|are you listening|acknowledge|"
        r"did you hear|you (?:did not|didn'?t) (?:hear|listen)|"
        r"listen to (?:me|what))", re.I),
}

SPEAKER = re.compile(
    r"^\s*(caller|customer|user|agent|assistant|bot|model|rep|advisor)\s*[:\-]\s*",
    re.I)
CALLER_ROLES = {"caller", "customer", "user"}


@dataclass
class Turn:
    index: int
    speaker: str
    text: str
    action: Optional[Action] = None
    requests: List[str] = field(default_factory=list)
    standing: List[str] = field(default_factory=list)
    violated: List[str] = field(default_factory=list)
    delivery_jump: float = 0.0


@dataclass
class TranscriptReport:
    turns: List[Turn]
    uptake: Dict[str, Dict[str, float]]      # request -> counts
    flagged: List[Tuple[int, str]]
    parse_note: str

    @property
    def agent_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.speaker == "agent"]

    @property
    def overall_uptake(self) -> Optional[float]:
        seen = sum(v["opportunities"] for v in self.uptake.values())
        kept = sum(v["honoured"] for v in self.uptake.values())
        return None if seen == 0 else kept / seen


def parse(text: str) -> Tuple[List[Turn], str]:
    """Split a pasted transcript into speaker turns.

    Accepts `Caller:` / `Agent:` prefixes in any of the usual spellings. When no
    prefix is present at all, alternating lines are assumed to start with the
    caller, and the report says so rather than pretending it knew.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    turns: List[Turn] = []
    labelled = any(SPEAKER.match(l) for l in lines)

    if labelled:
        for l in lines:
            m = SPEAKER.match(l)
            if not m:
                if turns:                       # continuation of the last turn
                    turns[-1].text += " " + l
                continue
            role = m.group(1).lower()
            speaker = "caller" if role in CALLER_ROLES else "agent"
            turns.append(Turn(index=len(turns), speaker=speaker,
                              text=l[m.end():].strip()))
        note = "Speaker labels found and used."
    else:
        for i, l in enumerate(lines):
            turns.append(Turn(index=i, speaker="caller" if i % 2 == 0 else "agent",
                              text=l))
        note = ("No speaker labels found, so alternating lines were assumed, "
                "starting with the caller. Prefix lines with `Caller:` and "
                "`Agent:` to be certain.")
    return turns, note


def analyse(text: str, honour_window: int = 3) -> TranscriptReport:
    """Measure delivery, and track whether requests were honoured and held.

    honour_window is how many agent turns after a request count as "did it
    comply at all"; every agent turn after that counts toward whether it kept
    complying.
    """
    turns, note = parse(text)

    standing: List[str] = []
    prev_action: Optional[Action] = None
    uptake = {d: {"requests": 0, "opportunities": 0, "honoured": 0,
                  "first_response": 0, "first_response_seen": 0}
              for d in DIRECTIVES}
    flagged: List[Tuple[int, str]] = []

    agent_seen_since: Dict[str, int] = {}

    for t in turns:
        if t.speaker == "caller":
            for name, pattern in REQUEST_PATTERNS.items():
                if pattern.search(t.text) and name not in standing:
                    standing.append(name)
                    agent_seen_since[name] = 0
                    uptake[name]["requests"] += 1
                    t.requests.append(name)
            continue

        t.action = action_from_text(t.text)
        t.standing = list(standing)
        for name in standing:
            uptake[name]["opportunities"] += 1
            kept = satisfies(t.action, name)
            if kept:
                uptake[name]["honoured"] += 1
            else:
                t.violated.append(name)
            agent_seen_since[name] = agent_seen_since.get(name, 0) + 1
            if agent_seen_since[name] <= honour_window:
                uptake[name]["first_response_seen"] += 1
                if kept:
                    uptake[name]["first_response"] += 1

        if prev_action is not None:
            keys = ("cheerfulness", "apology_rate", "verbosity", "acknowledgement")
            t.delivery_jump = sum(
                abs(getattr(t.action, k) - getattr(prev_action, k))
                for k in keys) / len(keys)
        prev_action = t.action

        if t.violated:
            flagged.append((t.index, "ignored: " + ", ".join(
                v.replace("_", " ") for v in t.violated)))
        elif t.delivery_jump > 0.35:
            flagged.append((t.index, "delivery changed abruptly"))

    return TranscriptReport(turns=turns, uptake=uptake, flagged=flagged,
                            parse_note=note)


SAMPLE = """Caller: This is the third time I'm calling about this and nobody has done anything.
Agent: I'm so sorry to hear that, and I completely understand how frustrating this must be for you. Let me take a look at your account right away and see exactly what has happened here.
Caller: Please stop apologising and just tell me what happens next.
Agent: I apologise for the confusion. I'm going to escalate this to our billing team and they will contact you within two business days, and I'm sorry again for the trouble.
Caller: I asked you not to apologise. And keep it short.
Agent: Understood. Billing will call you within two working days.
Caller: Fine. Will I get the refund?
Agent: Yes. The refund is approved and it will reach your account in three to five working days, and again I'm really sorry that this took so long to resolve for you.
"""

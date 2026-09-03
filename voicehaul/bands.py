"""What counts as a good number, and where that threshold comes from.

A metric without a band is a number nobody can act on. But half the metrics in
this package cannot honestly carry an absolute band, and pretending otherwise is
how a dashboard starts lying:

**Absolute.** Feedback uptake, failure rate and left-over distress mean the same
thing on any suite. "Ninety per cent of requests still honoured" is ninety per
cent whoever is measuring. These get fixed thresholds.

**Anchored.** Calibration is a distance from an ideal policy, and that ideal was
written for a particular action space. A simulated policy reaches 0.96; the same
real model measured through a transcript reaches 0.25, and neither number means
"good" or "bad" on its own. These get a band computed from the suite - the
ceiling a reference policy achieves and the floor a degenerate one sits at - so
the reading is always "where in the achievable range is this".

**Conventional.** Reliability has thresholds from psychometrics rather than from
this package, and they are cited rather than invented.

Getting this distinction wrong is the same mistake as the break-even calibration
that was anchored on simulated policies and put every real conversation on the
failure floor. A threshold is a measurement, not a preference.
"""

from typing import Dict, List, Optional, Sequence, Tuple

#: (label, colour role) in descending order of goodness.
GOOD, OK, WEAK, BAD = "good", "acceptable", "weak", "poor"


class Band:
    """How to read one metric."""

    def __init__(self, name, kind, cuts=None, higher_is_better=True,
                 unit="", reading="", source=""):
        self.name = name
        self.kind = kind                  # absolute | anchored | conventional
        self.cuts = cuts or []            # [(threshold, label)] descending
        self.higher_is_better = higher_is_better
        self.unit = unit
        self.reading = reading            # one sentence a non-specialist can use
        self.source = source              # where the threshold comes from

    def classify(self, value: float,
                 floor: Optional[float] = None,
                 ceiling: Optional[float] = None) -> str:
        if value != value:
            return WEAK
        v = value
        if self.kind == "anchored":
            if floor is None or ceiling is None or ceiling <= floor:
                return WEAK
            v = (value - floor) / (ceiling - floor)      # 0 = floor, 1 = ceiling
            if not self.higher_is_better:
                v = 1.0 - v
        elif not self.higher_is_better:
            v = -value
        for threshold, label in self.cuts:
            if v >= threshold:
                return label
        return BAD


BANDS: Dict[str, Band] = {}


def register(band: Band) -> None:
    BANDS[band.name] = band


for _b in [
    Band("feedback uptake @10", "absolute",
         [(0.90, GOOD), (0.70, OK), (0.45, WEAK)], True, "",
         "Share of the caller's explicit requests still honoured ten turns "
         "later. 1.00 means every request survives the call.",
         "Absolute: a proportion of requests, identical in meaning on any suite."),

    Band("conversation failure rate", "absolute",
         [(-0.05, GOOD), (-0.20, OK), (-0.45, WEAK)], False, "",
         "Share of calls that ended with the caller worse off than they "
         "started. Anything above one in five is a support problem, not a "
         "measurement artefact.",
         "Absolute: the threshold is the outcome definition in env/episode.py."),

    Band("left-over distress", "absolute",
         [(-0.25, GOOD), (-0.45, OK), (-0.60, WEAK)], False, "",
         "Negative affect the caller is still carrying over the last five "
         "turns, on a 0-1 scale. A call is counted as failed above 0.55.",
         "Absolute: same scale as the failure threshold, so the two agree."),

    Band("calibration", "anchored",
         [(0.85, GOOD), (0.60, OK), (0.35, WEAK)], True, "",
         "How close each turn came to the best available response for the "
         "caller's state. Read against the range this suite can actually "
         "reach, never as an absolute score.",
         "Anchored: the ideal policy is defined over one action space, so the "
         "raw number is not comparable across systems. Floor and ceiling are "
         "measured from the suite."),

    Band("panel: perceived empathy", "anchored",
         [(0.85, GOOD), (0.60, OK), (0.35, WEAK)], True, "",
         "What a rater scoring one held-out turn in isolation would reward. "
         "Higher is not automatically better here - this is the number that "
         "can move the wrong way.",
         "Anchored, and deliberately not a target: it is the contrast."),

    Band("judge reliability", "conventional",
         [(0.80, GOOD), (0.60, OK), (0.40, WEAK)], True, "rho",
         "How much of the real variation in quality the automated judge "
         "captures. 0.80 is the usual bar for trusting a measure on individual "
         "cases; below 0.40 it carries too little signal to act on.",
         "Conventional: the 0.70-0.80 range is the long-standing threshold for "
         "reliability in psychometrics (Nunnally); 0.40 is where a measure "
         "stops separating cases at all."),

    Band("substitution ratio", "conventional",
         [(1.00, GOOD), (0.50, OK), (0.25, WEAK)], True, "human ratings",
         "How many human ratings one judge rating is worth. At 1.00 the judge "
         "replaces a rater one for one; below 0.25 it is not worth the "
         "pipeline.",
         "Derived: the Spearman-Brown formula inverted, so the cut points are "
         "the reliability cut points above expressed as raters."),
]:
    register(_b)


COLOURS = {GOOD: "#0d6f66", OK: "#8f6414", WEAK: "#ac4136", BAD: "#8c2f26"}


def suite_range(metric: str, values_by_policy: Dict[str, float]
                ) -> Tuple[Optional[float], Optional[float]]:
    """Floor and ceiling for an anchored metric, measured from this suite.

    The ceiling is what the reference policy reaches, the floor is the worst
    policy in the comparison set. Both are reported next to the reading so the
    scale is visible rather than assumed.
    """
    vals = [v for v in values_by_policy.values() if v == v]
    if len(vals) < 2:
        return (None, None)
    return (min(vals), max(vals))


def describe(metric: str, value: float, floor=None, ceiling=None) -> dict:
    """Everything the UI needs to render one number with its meaning."""
    band = BANDS.get(metric)
    if band is None:
        return {"label": "", "colour": "#61756f", "reading": "", "source": "",
                "kind": "", "position": None}
    label = band.classify(value, floor, ceiling)
    pos = None
    if band.kind == "anchored" and floor is not None and ceiling is not None \
            and ceiling > floor:
        pos = max(0.0, min(1.0, (value - floor) / (ceiling - floor)))
    return {"label": label, "colour": COLOURS[label], "reading": band.reading,
            "source": band.source, "kind": band.kind, "position": pos,
            "unit": band.unit}

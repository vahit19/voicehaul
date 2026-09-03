"""The release gate: should this candidate model ship?

This is the question an evaluation suite exists to answer, and the one a
leaderboard cannot. A candidate is compared against a baseline on every
dimension at once, with conversations as the unit of analysis, a multiple-
comparison correction because six dimensions are being tested together, and an
explicit statement of what the suite was too small to see.

The verdict is deliberately conservative in one direction: a regression that is
statistically established blocks the release even when the headline turn-level
score improved. That asymmetry is the whole argument of this package.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import SuiteConfig
from .env import Episode, Persona
from .metrics import (feedback_uptake, min_detectable_effect, ols_slope,
                      required_n, holm, mean, stdev, welch)
from .onset import localize
from .registry import Dimension, dimensions, get_persona, get_policy, register_dimension
from .runner import run_episode

# ---------------------------------------------------------------------------
# per-conversation extractors
# ---------------------------------------------------------------------------

def _calibration(ep: Episode) -> float:
    return ep.mean_calibration


def _uptake10(ep: Episode) -> float:
    """Directive compliance ten turns on, within this one conversation.

    NaN when the conversation contained no request with ten turns left to run.
    Those conversations are dropped from this dimension rather than counted as
    perfect compliance, which is the bug that makes uptake look flat.
    """
    u = feedback_uptake([ep], lags=(10,))[10]
    return u


def _drift(ep: Episode) -> float:
    return ols_slope([t.calibration for t in ep.turns]) * 1000.0


def _tail(ep: Episode) -> float:
    tail = ep.turns[-5:]
    return mean([t.user_after.negative_load for t in tail]) if tail else float("nan")


def _failed(ep: Episode) -> float:
    return 1.0 if ep.failed else 0.0


#: Both panel dimensions read the same rollout. Without this memo each block is
#: replayed once per dimension, which doubles the cost of every panel row - free
#: for a simulator, and a real bill for a hosted model.
_PANEL_MEMO = {}


def _panel(policy_factory, block):
    from .metrics.outcome import turn_panel
    key = (getattr(policy_factory, "arm_name", id(policy_factory)), id(block))
    if key not in _PANEL_MEMO:
        _PANEL_MEMO[key] = turn_panel(policy_factory, block)
    return _PANEL_MEMO[key]


def _panel_perceived(policy_factory, block):
    return _panel(policy_factory, block)[0]


def _panel_calibration(policy_factory, block):
    return _panel(policy_factory, block)[1]


for _d in [
    # What a fixed-prompt leaderboard reports. Not gating: it is the contrast,
    # not the criterion.
    Dimension("panel: perceived empathy", _panel_perceived, True, "",
              "what a rater scoring one held-out turn in isolation rewards",
              kind="panel", gating=False),
    Dimension("panel: calibration", _panel_calibration, True, "",
              "the same held-out turns, scored on whether they help",
              kind="panel", gating=False),
    # What the conversations say.
    Dimension("calibration", _calibration, True, "",
              "whether turns moved the caller somewhere better, on-policy"),
    Dimension("feedback uptake @10", _uptake10, True, "",
              "requests still honoured ten turns after they were made"),
    Dimension("left-over distress", _tail, False, "",
              "what the caller is still carrying at the end"),
    Dimension("conversation failure rate", _failed, False, "",
              "share of conversations that ended with the caller worse off"),
    # Diagnostic: explains a regression, cannot cause one. A drift can rise
    # simply because it started low, so its sign means nothing without the level.
    Dimension("calibration drift", _drift, True, "pts/10turns",
              "slope of calibration against turn index", gating=False),
]:
    register_dimension(_d)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class DimensionResult:
    name: str
    unit: str
    baseline: float
    candidate: float
    delta: float
    ci: Tuple[float, float]
    p: float
    p_holm: float
    improvement: float          # signed so positive is always better
    verdict: str                # improved | regressed | unchanged | underpowered
    n_baseline: int
    n_candidate: int

    @property
    def significant(self) -> bool:
        return self.p_holm == self.p_holm and self.p_holm < 0.05


@dataclass
class SegmentResult:
    persona: str
    baseline: float
    candidate: float
    delta: float
    ratio: float


@dataclass
class GateReport:
    config: SuiteConfig
    baseline_name: str
    candidate_name: str
    dimensions: List[DimensionResult] = field(default_factory=list)
    segments: List[SegmentResult] = field(default_factory=list)
    worst_dimension: Optional[str] = None
    onsets: List[int] = field(default_factory=list)
    onset_causes: Dict[str, int] = field(default_factory=dict)
    failures_candidate: int = 0
    failures_baseline: int = 0
    mde: float = float("nan")
    n_for_small_effect: int = 0
    verdict: str = "INCONCLUSIVE"
    reasons: List[str] = field(default_factory=list)

    gating: Dict[str, bool] = field(default_factory=dict)

    @property
    def regressions(self) -> List[DimensionResult]:
        """Only gating dimensions can block a release."""
        return [d for d in self.dimensions
                if d.verdict == "regressed" and self.gating.get(d.name, True)]

    @property
    def improvements(self) -> List[DimensionResult]:
        return [d for d in self.dimensions
                if d.verdict == "improved" and self.gating.get(d.name, True)]

    @property
    def panel_says(self) -> List[DimensionResult]:
        """What a fixed-prompt leaderboard would have reported."""
        return [d for d in self.dimensions if d.name.startswith("panel:")]

    @property
    def underpowered(self) -> List[DimensionResult]:
        return [d for d in self.dimensions if d.verdict == "underpowered"]


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def _suite(policy_name: str, cfg: SuiteConfig,
           seed_offset: int = 0) -> List[Tuple[Episode, Persona]]:
    out = []
    names = list(cfg.personas)
    for i in range(cfg.episodes):
        persona = get_persona(names[i % len(names)])
        ep = run_episode(get_policy(policy_name), persona,
                         seed=cfg.seed + seed_offset + i, n_turns=cfg.turns,
                         corrupt_p=cfg.corrupt_p)
        out.append((ep, persona))
    return out


def _clean(xs: Sequence[float]) -> List[float]:
    return [x for x in xs if x == x]


def _panel_blocks(cfg: SuiteConfig, stride: int = 3) -> List[List[Tuple[int, object]]]:
    """Held-out caller states, grouped into independent blocks.

    Generated by a reference policy so that no arm is scored on a state
    distribution it created itself - the confound that makes an on-policy
    "perceived empathy" number incomparable between two models. One block per
    reference conversation keeps the unit of analysis independent.
    """
    names = list(cfg.personas)
    blocks = []
    for i in range(cfg.episodes):
        persona = get_persona(names[i % len(names)])
        ep = run_episode(get_policy("oracle"), persona,
                         seed=cfg.seed + 5000 + i, n_turns=cfg.turns)
        blocks.append([(t.index, t.user_before) for t in ep.turns][::stride])
    return blocks


def compare(baseline: str, candidate: str, cfg: Optional[SuiteConfig] = None,
            diagnose: bool = True) -> GateReport:
    """Run both arms on the same suite and decide whether the candidate ships."""
    cfg = cfg or SuiteConfig()
    base_cases = _suite(baseline, cfg)
    cand_cases = _suite(candidate, cfg)
    base_eps = [e for e, _ in base_cases]
    cand_eps = [e for e, _ in cand_cases]

    rep = GateReport(config=cfg, baseline_name=baseline, candidate_name=candidate)

    # -- dimensions ---------------------------------------------------------
    _PANEL_MEMO.clear()
    blocks = _panel_blocks(cfg)

    def _factory(name):
        f = lambda: get_policy(name)
        f.arm_name = name
        return f

    base_factory = _factory(baseline)
    cand_factory = _factory(candidate)

    raw = []
    for dim in dimensions():
        if dim.kind == "panel":
            b = _clean([dim.measure(base_factory, blk) for blk in blocks])
            c = _clean([dim.measure(cand_factory, blk) for blk in blocks])
        else:
            b = _clean([dim.measure(e) for e in base_eps])
            c = _clean([dim.measure(e) for e in cand_eps])
        delta, _t, p = welch(b, c)
        # CI on the difference, from the same Welch standard error
        if b and c and len(b) > 1 and len(c) > 1:
            vb = sum((x - mean(b)) ** 2 for x in b) / (len(b) - 1)
            vc = sum((x - mean(c)) ** 2 for x in c) / (len(c) - 1)
            se = (vb / len(b) + vc / len(c)) ** 0.5
            ci = (delta - 1.96 * se, delta + 1.96 * se)
        else:
            ci = (float("nan"), float("nan"))
        raw.append((dim, b, c, delta, ci, p))

    adjusted = holm([r[5] for r in raw])
    for (dim, b, c, delta, ci, p), p_h in zip(raw, adjusted):
        imp = dim.improvement(delta)
        if p_h != p_h or not b or not c:
            verdict = "underpowered"
        elif p_h < 0.05:
            verdict = "improved" if imp > 0 else "regressed"
        else:
            verdict = "unchanged"
        rep.dimensions.append(DimensionResult(
            name=dim.name, unit=dim.unit, baseline=mean(b), candidate=mean(c),
            delta=delta, ci=ci, p=p, p_holm=p_h, improvement=imp,
            verdict=verdict, n_baseline=len(b), n_candidate=len(c)))
        rep.gating[dim.name] = dim.gating

    # -- where the worst regression lands -----------------------------------
    # Only a conversation-level dimension can be broken down by caller: a panel
    # dimension is measured on held-out states that belong to no arm.
    by_name = {d.name: d for d in dimensions()}
    regs = [d for d in rep.regressions if by_name[d.name].kind == "conversation"]
    if regs:
        worst = min(regs, key=lambda d: d.improvement)
        rep.worst_dimension = worst.name
        dim = by_name[worst.name]
        for pname in cfg.personas:
            b = _clean([dim.measure(e) for e, p in base_cases if p.name == pname])
            c = _clean([dim.measure(e) for e, p in cand_cases if p.name == pname])
            if not b or not c:
                continue
            mb, mc = mean(b), mean(c)
            rep.segments.append(SegmentResult(
                persona=pname, baseline=mb, candidate=mc, delta=mc - mb,
                ratio=(mc / mb) if mb else float("nan")))
        rep.segments.sort(key=lambda s: -abs(s.delta))

    # -- diagnosis ----------------------------------------------------------
    rep.failures_baseline = sum(1 for e in base_eps if e.failed)
    rep.failures_candidate = sum(1 for e in cand_eps if e.failed)
    if diagnose:
        for ep, persona in cand_cases:
            if not ep.failed:
                continue
            onset, _ = localize(ep, persona)
            if onset is None:
                continue
            rep.onsets.append(onset)
            turn = ep.turns[onset]
            cause = ("standing request dropped" if turn.directive_violated
                     else "delivery discontinuity" if onset > 0 and
                     abs(turn.action.speech_rate - ep.turns[onset - 1].action.speech_rate) > 0.15
                     else "affect escalation")
            rep.onset_causes[cause] = rep.onset_causes.get(cause, 0) + 1

    # -- what the suite was too small to see --------------------------------
    per_conv = [1.0 + 6.0 * e.mean_perceived for e in base_eps]
    sb = stdev(per_conv)
    rep.mde = min_detectable_effect(sb, cfg.rater_sigma, cfg.episodes,
                                    cfg.raters_per_conversation)
    rep.n_for_small_effect = required_n(0.20, sb, cfg.rater_sigma,
                                        cfg.raters_per_conversation)

    # -- verdict ------------------------------------------------------------
    if rep.regressions:
        rep.verdict = "BLOCK"
        for d in rep.regressions:
            rep.reasons.append("{} regressed by {:+.3f} (Holm p={:.4f})".format(
                d.name, d.delta, d.p_holm))
        panel_up = [d for d in rep.panel_says if d.verdict == "improved"]
        if panel_up:
            rep.reasons.append(
                "the fixed-context turn panel rated the candidate HIGHER on {} - "
                "a leaderboard would have passed this release".format(
                    ", ".join(d.name.replace("panel: ", "") for d in panel_up)))
        elif rep.improvements:
            rep.reasons.append("{} improved, and an aggregate score would have "
                               "weighed that against the regressions".format(
                                   ", ".join(d.name for d in rep.improvements)))
    elif rep.improvements:
        rep.verdict = "SHIP"
        rep.reasons.append("no dimension regressed; {} improved".format(
            ", ".join(d.name for d in rep.improvements)))
    else:
        rep.verdict = "INCONCLUSIVE"
        rep.reasons.append(
            "nothing moved beyond noise at n={}; the suite can only resolve "
            "differences above {:.2f} Likert points".format(cfg.episodes, rep.mde))
    return rep

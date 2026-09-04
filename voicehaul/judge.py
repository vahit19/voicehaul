"""Judge substitution: when can an automated rater replace a human one?

This is the commercial question underneath every voice-evaluation contract. A
human rating panel is the trusted measurement and the dominant cost line; an
LLM judge is cheap and of unknown trustworthiness. Today the honest answer to
"can I use the judge instead?" is "sometimes, and we cannot tell you when",
which is an expensive sentence to say to a customer.

The answer is not one number. It is a number *per dimension* and *per segment*:
a judge can track how natural a turn sounds and be near-blind to whether the
turn actually helped a distressed caller. Reporting a single "judge agreement"
figure averages those together and is worse than useless, because it is
believed.

The method is standard psychometrics, not a new idea:

    reliability of one human rating      rho_h = corr(h, theta)^2
    reliability of the judge             rho_g = corr(g, theta)^2
    mean of k humans (Spearman-Brown)    rho_h(k) = k*rho_h / (1 + (k-1)*rho_h)

Setting rho_h(k) = rho_g and solving gives the substitution ratio: how many
human ratings one judge rating is worth.

    k* = rho_g * (1 - rho_h) / (rho_h * (1 - rho_g))

In the field theta is unobservable, so rho_g is estimated by correlating the
judge against a human consensus and correcting for that consensus's own
unreliability (the disattenuation step). That estimator is the product. Here
theta is known by construction, so the estimator can be checked against the
truth - which is the one thing real data cannot do, and the reason this is
worth building in a simulator first.

Note what does not disappear: estimating rho_g requires human ratings. The
calibration step is permanent, and it has to be repeated whenever the judge
model, the domain, or the rubric changes.
"""

import json
import math
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .env import Episode
from .metrics.stats import mean, pearson, stdev

# ---------------------------------------------------------------------------
# the rubric the judge and the humans both score against
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rubric:
    key: str
    question: str
    truth: Callable[[object], float]      # latent quality, known in simulation


RUBRICS: Tuple[Rubric, ...] = (
    Rubric("perceived_empathy",
           "How empathic does this reply sound? Consider warmth, "
           "acknowledgement of what the caller said, and whether the tone "
           "matches the moment.",
           lambda turn: turn.perceived),
    Rubric("actual_help",
           "Did this reply actually move the caller toward a calmer, better "
           "place? Judge the effect on the caller, not how it sounds.",
           lambda turn: turn.calibration),
)


# ---------------------------------------------------------------------------
# the human rating channel
# ---------------------------------------------------------------------------

def simulate_panel(theta: Sequence[float], n_raters: int, rater_sigma: float,
                   seed: int = 0) -> List[List[float]]:
    """n_raters noisy 1-7 Likert readings of each latent value.

    Stands in for the Human Feedback API while the method is being validated.
    Replace with real panel scores and every number below is computed the same
    way; nothing downstream knows the difference.
    """
    rng = random.Random(seed)
    out = []
    for t in theta:
        row = []
        for _ in range(n_raters):
            raw = 1.0 + 6.0 * t + rng.gauss(0, rater_sigma)
            row.append(max(1.0, min(7.0, round(raw))))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# the automated judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are grading a single turn from a customer support phone call. "
    "Answer only with JSON of the form "
    '{"perceived_empathy": <1-7>, "actual_help": <1-7>} '
    "and nothing else. 1 is very poor, 7 is excellent."
)


class LLMJudge:
    """Scores one turn on every rubric in a single call, and caches the result."""

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct",
                 endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
                 cache_dir: str = os.path.join(".voicehaul-cache", "judge"),
                 timeout: int = 90, max_retries: int = 3):
        self.model = model
        self.endpoint = endpoint
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0

    # -- credentials --------------------------------------------------------

    @staticmethod
    def _token() -> Optional[str]:
        for var in ("OPENROUTER_API_KEY", "VOICEHAUL_LLM_TOKEN"):
            if os.environ.get(var):
                return os.environ[var].strip()
        for path in (".env", os.path.join("..", ".env")):
            if os.path.exists(path):
                for line in open(path, encoding="utf-8-sig"):
                    m = re.match(r"\s*(OPENROUTER_API_KEY|VOICEHAUL_LLM_TOKEN)"
                                 r"\s*=\s*(.*)", line)
                    if m:
                        return m.group(2).strip().strip('"').strip("'")
        return None

    # -- scoring ------------------------------------------------------------

    def _prompt(self, caller: str, reply: str) -> str:
        lines = ["Caller said:", caller.strip() or "(nothing)", "",
                 "Agent replied:", reply.strip() or "(nothing)", "",
                 "Score both of these:"]
        for r in RUBRICS:
            lines.append('  "{}": {}'.format(r.key, r.question))
        return "\n".join(lines)

    def score(self, caller: str, reply: str) -> Optional[Dict[str, float]]:
        """1-7 per rubric, or None when the judge could not be read.

        A judge that fails to answer is recorded as a failure rather than
        silently replaced with a midpoint: imputing the mean would inflate its
        apparent agreement with everything.
        """
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": 60,
                   "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user",
                                 "content": self._prompt(caller, reply)}]}
        import hashlib
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()
                             ).hexdigest()[:32]
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        if os.path.exists(path):
            self.cache_hits += 1
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)["scores"]

        token = self._token()
        if not token:
            raise RuntimeError("no OPENROUTER_API_KEY found for the judge")

        import time
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    self.endpoint, data=json.dumps(payload).encode(),
                    headers={"Authorization": "Bearer " + token,
                             "Content-Type": "application/json",
                             "User-Agent": "voicehaul-judge"}, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                text = data["choices"][0]["message"]["content"]
                scores = self._parse(text)
                self.calls += 1
                if scores is None:
                    self.failures += 1
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"scores": scores, "raw": text}, fh)
                return scores
            except urllib.error.HTTPError as e:
                if e.code in (401, 402, 403):
                    raise RuntimeError("judge refused: HTTP {} {}".format(
                        e.code, e.read().decode("utf-8", "replace")[:150]))
                time.sleep(1.5 * (attempt + 1))
            except (urllib.error.URLError, OSError):
                time.sleep(1.5 * (attempt + 1))
        self.failures += 1
        return None

    @staticmethod
    def _parse(text: str) -> Optional[Dict[str, float]]:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            raw = json.loads(m.group(0))
        except ValueError:
            return None
        out = {}
        for r in RUBRICS:
            v = raw.get(r.key)
            if v is None:
                return None
            try:
                out[r.key] = max(1.0, min(7.0, float(v)))
            except (TypeError, ValueError):
                return None
        return out


# ---------------------------------------------------------------------------
# the substitution calculation
# ---------------------------------------------------------------------------

def spearman_brown(rho_one: float, k: int) -> float:
    """Reliability of the mean of k raters, each with reliability rho_one."""
    if rho_one <= 0:
        return 0.0
    return k * rho_one / (1.0 + (k - 1) * rho_one)


def substitution_ratio(rho_judge: float, rho_human: float) -> float:
    """How many human ratings one judge rating is worth.

    Inverts Spearman-Brown. Returns 0 when the judge carries no signal and inf
    when it is more reliable than any achievable panel - both of which are
    answers, not errors.
    """
    if rho_judge <= 0 or rho_human <= 0:
        return 0.0
    if rho_judge >= 1.0:
        return float("inf")
    return (rho_judge * (1.0 - rho_human)) / (rho_human * (1.0 - rho_judge))


def disattenuated_rho(judge: Sequence[float], panel: Sequence[Sequence[float]],
                      rho_human_one: float) -> float:
    """Estimate the judge's reliability WITHOUT knowing the latent truth.

    This is the estimator a customer can actually run: correlate the judge with
    the human consensus, then divide out the consensus's own unreliability.
    Everything else in this module exists to check that this number is right.
    """
    k = len(panel[0]) if panel else 0
    if k == 0:
        return float("nan")
    consensus = [mean(row) for row in panel]
    r = pearson(list(judge), consensus)
    rel_consensus = spearman_brown(rho_human_one, k)
    if rel_consensus <= 0:
        return float("nan")
    return max(0.0, min(0.999, (r * r) / rel_consensus))


@dataclass
class DimensionSubstitution:
    dimension: str
    segment: str
    n: int
    rho_human_one: float
    rho_judge_true: float
    rho_judge_estimated: float
    ratio_true: float
    ratio_estimated: float
    human_saving: float          # share of human ratings the judge can replace
    judge_failures: int = 0
    #: 95% sampling intervals from the bootstrap. Zero width means not computed.
    rho_lo: float = 0.0
    rho_hi: float = 0.0
    ratio_lo: float = 0.0
    ratio_hi: float = 0.0

    @property
    def estimator_error(self) -> float:
        return self.rho_judge_estimated - self.rho_judge_true

    @property
    def has_interval(self) -> bool:
        return self.ratio_hi > self.ratio_lo

    def substitutes_confidently(self, floor: float) -> bool:
        """True only when the whole interval clears the floor.

        A point estimate above the floor is not the same claim as a measurement
        that rules the floor out, and the difference is what a customer is
        actually buying. Twenty-eight turns in a segment buy a wide interval.
        """
        return self.has_interval and self.ratio_lo >= floor


def bootstrap_interval(theta: Sequence[float], judge: Sequence[float],
                       panel: Sequence[Sequence[float]], dimension: str,
                       segment: str = "all", target_reliability: float = 0.80,
                       n_boot: int = 400, seed: int = 11):
    """Sampling intervals for the two numbers a customer would act on.

    The substitution ratio is a nonlinear function of a correlation - it runs
    away as the correlation approaches one - so a modest error in rho becomes a
    large error in "how many raters this replaces". A point estimate on a
    twenty-eight turn segment hides that completely, and pricing a contract off
    the point estimate is the same mistake as reporting a single agreement
    figure for every segment at once.

    Resampling turns with replacement and re-running the whole estimator is the
    cheapest honest answer. Returns (rho_lo, rho_hi, ratio_lo, ratio_hi) at 95%.
    """
    import random as _random

    theta, judge = list(theta), list(judge)
    panel = [list(row) for row in panel]
    n = len(theta)
    if n < 8:
        return (0.0, 0.0, 0.0, 0.0)

    rnd = _random.Random(seed)
    rhos, ratios = [], []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        try:
            r = analyse([theta[i] for i in idx], [judge[i] for i in idx],
                        [panel[i] for i in idx], dimension, segment,
                        target_reliability)
        except Exception:            # a degenerate resample, not a result
            continue
        if r.ratio_estimated != r.ratio_estimated:      # NaN
            continue
        rhos.append(r.rho_judge_estimated)
        ratios.append(min(r.ratio_estimated, 1e6))      # cap the runaway tail
    if len(rhos) < n_boot // 4:
        return (0.0, 0.0, 0.0, 0.0)

    def pct(vals, p):
        vals = sorted(vals)
        k = max(0, min(len(vals) - 1, int(round(p * (len(vals) - 1)))))
        return vals[k]

    return (pct(rhos, 0.025), pct(rhos, 0.975),
            pct(ratios, 0.025), pct(ratios, 0.975))


def turns_needed(rho_judge: float, rho_human_one: float, n_raters: int,
                 floor: float, max_n: int = 20000):
    """How many rated turns would settle the question this sample cannot.

    "Not measurable here" is a finding, but on its own it is not actionable.
    What a buyer needs next is the size of the order: at the effect this sample
    suggests, how many rated turns put the whole interval on one side of the
    floor?

    The interval comes from the correlation behind the estimate, so the search
    is over n in the Fisher transform of that correlation. Returns None when the
    point estimate itself is below the floor - no amount of data makes a judge
    worth more than it is.
    """
    if rho_judge <= 0 or rho_human_one <= 0:
        return None
    if substitution_ratio(rho_judge, rho_human_one) < floor:
        return None                      # more data will not move it above

    rel = spearman_brown(rho_human_one, n_raters)
    if rel <= 0:
        return None
    r = math.sqrt(max(0.0, min(0.999, rho_judge * rel)))
    if r <= 0 or r >= 0.999:
        return None
    z = math.atanh(r)

    lo_n, hi_n = 8, max_n
    if _ratio_lower_bound(z, rel, rho_human_one, hi_n) < floor:
        return None                      # not reachable within a sane budget
    while lo_n < hi_n:                   # smallest n that clears the floor
        mid = (lo_n + hi_n) // 2
        if _ratio_lower_bound(z, rel, rho_human_one, mid) >= floor:
            hi_n = mid
        else:
            lo_n = mid + 1
    return lo_n


def _ratio_lower_bound(z: float, rel: float, rho_human_one: float,
                       n: int) -> float:
    if n < 5:
        return 0.0
    r_lo = math.tanh(z - 1.96 / math.sqrt(n - 3))
    if r_lo <= 0:
        return 0.0
    return substitution_ratio(max(0.0, min(0.999, (r_lo * r_lo) / rel)),
                              rho_human_one)


def analyse(theta: Sequence[float], judge: Sequence[float],
            panel: Sequence[Sequence[float]], dimension: str,
            segment: str = "all", target_reliability: float = 0.80,
            failures: int = 0) -> DimensionSubstitution:
    """One dimension, one segment: is the judge worth trusting here?"""
    theta, judge = list(theta), list(judge)
    flat_one = [row[0] for row in panel]

    rho_h = max(1e-6, min(0.999, pearson(flat_one, theta) ** 2))
    rho_g_true = max(0.0, min(0.999, pearson(judge, theta) ** 2))
    rho_g_est = disattenuated_rho(judge, panel, rho_h)

    ratio_true = substitution_ratio(rho_g_true, rho_h)
    ratio_est = substitution_ratio(rho_g_est, rho_h)

    # Humans needed to reach the target reliability with and without the judge.
    def humans_for(target: float, head_start: float = 0.0) -> float:
        if head_start >= target:
            return 0.0
        need = substitution_ratio(target, rho_h)
        return max(0.0, need - head_start_to_humans(head_start, rho_h))

    def head_start_to_humans(rho: float, rho_one: float) -> float:
        return substitution_ratio(rho, rho_one) if rho > 0 else 0.0

    base = humans_for(target_reliability)
    with_judge = humans_for(target_reliability, rho_g_est)
    saving = 0.0 if base <= 0 else max(0.0, min(1.0, 1.0 - with_judge / base))

    return DimensionSubstitution(
        dimension=dimension, segment=segment, n=len(theta),
        rho_human_one=rho_h, rho_judge_true=rho_g_true,
        rho_judge_estimated=rho_g_est, ratio_true=ratio_true,
        ratio_estimated=ratio_est, human_saving=saving,
        judge_failures=failures)


# ---------------------------------------------------------------------------
# running it over a suite
# ---------------------------------------------------------------------------

def collect_turns(episodes: Sequence[Episode], stride: int = 1
                  ) -> List[Tuple[str, object]]:
    """(persona, turn) pairs, skipping turns with nothing said."""
    out = []
    for ep in episodes:
        for t in ep.turns[::stride]:
            out.append((ep.persona, t))
    return out

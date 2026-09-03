"""What a regression suite costs, and what the measurements above save.

Every number elsewhere in this package is a statistic. A statistic is not a
decision until it has a price on it: "n = 168 conversations" is not actionable,
"$2,016 per release, $24,192 a year" is.

The model is deliberately thin. It multiplies three things this package already
measures - how many conversations a target sensitivity needs, how many of those
a diversity-selected subset can replace, and how many human ratings one judge
rating is worth - by two prices the customer supplies. Nothing is estimated
here that is not measured elsewhere, and the one place a guess would be
tempting (what a rating costs) is an input rather than a default anyone should
trust.

Two things it refuses to do:

* It never lets the judge replace the whole panel. Estimating the substitution
  ratio *requires* human ratings, and the estimate expires when the judge model,
  the domain or the rubric changes. A calibration sample is a permanent line
  item, and pricing it at zero would be the kind of arithmetic that makes a
  procurement decision look better than it is.
* It never reports a saving on a dimension where the judge is unreliable. Below
  the substitution floor the honest answer is that there is nothing to save.
"""

from dataclasses import dataclass
from typing import Optional

from .metrics.power import required_n

#: A judge rating is worth less than this many human ratings -> no substitution.
SUBSTITUTION_FLOOR = 0.25

#: Share of conversations that stay on the human panel to keep the judge
#: calibrated, whatever the substitution ratio says.
CALIBRATION_SHARE = 0.15


@dataclass
class CostModel:
    cost_per_human_rating: float = 3.00
    cost_per_judge_rating: float = 0.01
    releases_per_year: int = 12
    raters_per_conversation: int = 3


@dataclass
class CostResult:
    conversations: int
    human_ratings_baseline: int
    cost_baseline: float
    human_ratings_with_judge: int
    judge_ratings: int
    cost_with_judge: float
    calibration_conversations: int
    saving_per_release: float
    saving_per_year: float
    substitution_ratio: float
    usable: bool
    note: str

    @property
    def saving_share(self) -> float:
        return 0.0 if self.cost_baseline <= 0 else (
            self.saving_per_release / self.cost_baseline)


def estimate(target_effect: float, sigma_between: float, rater_sigma: float,
             substitution_ratio: float, model: Optional[CostModel] = None,
             coverage_factor: float = 1.0) -> CostResult:
    """Price one release cycle, with and without an automated judge.

    coverage_factor < 1 is the fraction of conversations a diversity-selected
    subset needs for the same coverage as random sampling; it comes from
    select.equivalent_budget, not from an assumption.
    """
    m = model or CostModel()
    n = required_n(target_effect, sigma_between, rater_sigma,
                   m.raters_per_conversation)
    n = max(1, int(round(n * max(0.05, min(1.0, coverage_factor)))))

    human_baseline = n * m.raters_per_conversation
    cost_baseline = human_baseline * m.cost_per_human_rating

    if substitution_ratio < SUBSTITUTION_FLOOR:
        return CostResult(
            conversations=n, human_ratings_baseline=human_baseline,
            cost_baseline=cost_baseline,
            human_ratings_with_judge=human_baseline, judge_ratings=0,
            cost_with_judge=cost_baseline, calibration_conversations=n,
            saving_per_release=0.0, saving_per_year=0.0,
            substitution_ratio=substitution_ratio, usable=False,
            note=("The judge is worth {:.2f} human ratings here, below the {:.2f} "
                  "floor. There is nothing to save on this dimension: it stays on "
                  "the panel.").format(substitution_ratio, SUBSTITUTION_FLOOR))

    # Every conversation gets one judge rating; the judge contributes
    # `substitution_ratio` human-equivalents, and humans make up the rest.
    per_conv_from_judge = min(m.raters_per_conversation - 1.0,
                              substitution_ratio)
    humans_per_conv = max(1.0, m.raters_per_conversation - per_conv_from_judge)

    calib = max(1, int(round(n * CALIBRATION_SHARE)))
    human_with_judge = int(round(
        (n - calib) * humans_per_conv + calib * m.raters_per_conversation))
    judge_ratings = n
    cost_with_judge = (human_with_judge * m.cost_per_human_rating
                       + judge_ratings * m.cost_per_judge_rating)

    saving = max(0.0, cost_baseline - cost_with_judge)
    return CostResult(
        conversations=n, human_ratings_baseline=human_baseline,
        cost_baseline=cost_baseline, human_ratings_with_judge=human_with_judge,
        judge_ratings=judge_ratings, cost_with_judge=cost_with_judge,
        calibration_conversations=calib,
        saving_per_release=saving, saving_per_year=saving * m.releases_per_year,
        substitution_ratio=substitution_ratio, usable=True,
        note=("{} of the {} conversations stay fully on the panel to keep the "
              "judge calibrated. That sample is not optional and it does not "
              "shrink: the ratio expires whenever the judge model, the domain or "
              "the rubric changes.").format(calib, n))

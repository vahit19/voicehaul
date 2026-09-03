"""The rating budget.

Human ratings are both the ground truth and the cost line. These turn 'we
track regressions' into a number of conversations, a number of raters, and
an explicit floor below which a regression is invisible."""

import math
import random

from .stats import Z_80, Z_975


def simulate_human_rating(latent: float, rng: random.Random,
                          rater_sigma: float = 0.9) -> float:
    """A 1-7 Likert rating of a latent [0,1] quality, with realistic rater noise.

    rater_sigma is in Likert points; 0.7-1.1 is the range typically reported for
    subjective conversational-quality items.
    """
    raw = 1.0 + 6.0 * latent + rng.gauss(0, rater_sigma)
    return max(1.0, min(7.0, round(raw)))


def min_detectable_effect(sigma_between: float, rater_sigma: float,
                          n_conversations: int, n_raters: int) -> float:
    """Smallest true regression detectable at alpha=0.05, power=0.80 (two-sample).

    Returned in the same units as the ratings (Likert points).
    """
    sigma_eff = math.sqrt(sigma_between ** 2 + (rater_sigma ** 2) / max(1, n_raters))
    return (Z_975 + Z_80) * sigma_eff * math.sqrt(2.0 / n_conversations)


def required_n(target_effect: float, sigma_between: float, rater_sigma: float,
               n_raters: int) -> int:
    sigma_eff = math.sqrt(sigma_between ** 2 + (rater_sigma ** 2) / max(1, n_raters))
    n = 2.0 * ((Z_975 + Z_80) * sigma_eff / target_effect) ** 2
    return int(math.ceil(n))

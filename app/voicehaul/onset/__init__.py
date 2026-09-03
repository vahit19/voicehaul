"""Failure-onset localization.

A conversation that ends badly usually went wrong long before it ended. Knowing
which turn broke it is what turns "this model scores 3.8 on empathy" into "this
model drops the caller's slow-down request around turn 17, and everything after
is downstream of it".

Layered after Runopsy: deterministic detectors propose, a segment walk-back
finds where the anomalous stretch begins, counterfactual replay gates.
"""

from .localize import false_positive_rate, localize, score_localization
from .signals import (COMPONENTS, anomaly_components, anomaly_scores,
                      dominant_cause, segment_start)

__all__ = ["localize", "score_localization", "false_positive_rate",
           "anomaly_scores", "anomaly_components", "dominant_cause",
           "COMPONENTS", "segment_start"]

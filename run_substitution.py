"""Judge substitution: when can an automated rater replace a human one?

    python run_substitution.py --turns 160

Scores the same turns three ways - the latent truth (known here by
construction), a simulated human panel, and a real LLM judge - then reports how
many human ratings one judge rating is worth, per dimension and per caller
segment.

The point of running it in a simulator first: theta is observable, so the
field estimator can be checked against the true answer. On real data you can
only ever compare two noisy measurements to each other.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

from voicehaul.config import SuiteConfig
from voicehaul.env import PERSONAS
from voicehaul.judge import (RUBRICS, LLMJudge, analyse, bootstrap_interval,
                             collect_turns,
                             simulate_panel, spearman_brown,
                             substitution_ratio)
from voicehaul.policies import (CalibratedPolicy, DrifterPolicy, FlatPolicy,
                                MirrorPolicy)
from voicehaul.runner import run_episode

RULE = "-" * 84


def build_turns(n_per_policy: int, n_turns: int, seed: int, source: str,
                neutral: float = 0.58):
    """A spread of quality, so agreement is not measured on one narrow band.

    source="llm" grades real model utterances, which is the only valid input
    for a judge: a simulated policy emits delivery parameters, not words, and
    asking a language model to grade a bracketed parameter dump measures its
    tolerance for a strange prompt rather than its agreement with a rater.
    """
    eps = []
    if source == "llm":
        from voicehaul.policies.llm import calm_regulating, warm_mirroring
        for factory in (calm_regulating, warm_mirroring):
            for i in range(n_per_policy * 2):
                eps.append(run_episode(factory(), PERSONAS[i % len(PERSONAS)],
                                       seed=seed + i, n_turns=n_turns,
                                       neutral=neutral))
        return collect_turns(eps, stride=1)
    for cls in (CalibratedPolicy, MirrorPolicy, DrifterPolicy, FlatPolicy):
        for i in range(n_per_policy):
            eps.append(run_episode(cls(), PERSONAS[i % len(PERSONAS)],
                                   seed=seed + i, n_turns=n_turns))
    return collect_turns(eps, stride=3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-policy", type=int, default=3)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--rater-sigma", type=float, default=0.9)
    ap.add_argument("--target", type=float, default=0.80,
                    help="panel reliability a customer is buying")
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--source", choices=("sim", "llm"), default="llm",
                    help="grade real model utterances, or simulated policies")
    ap.add_argument("--neutral", type=float, default=0.32)
    args = ap.parse_args()

    pairs = build_turns(args.episodes_per_policy, args.turns, seed=0,
                        source=args.source, neutral=args.neutral)[:args.limit]
    print(RULE)
    print("JUDGE SUBSTITUTION   how much of the human panel can be automated?")
    print(RULE)
    print("turns scored   : {}".format(len(pairs)))
    print("human panel    : {} raters, sd {:.2f} Likert points".format(
        args.raters, args.rater_sigma))
    print("judge          : {}".format(args.model))
    print("graded text    : {}".format(
        "real model utterances" if args.source == "llm"
        else "simulated delivery parameters (weak judge input)"))
    print("target          : panel reliability {:.2f}".format(args.target))
    print()
    sys.stdout.flush()

    judge = LLMJudge(model=args.model)
    scores: List[Dict[str, float]] = []
    t0 = time.time()
    for i, (_persona, turn) in enumerate(pairs):
        try:
            scores.append(judge.score(turn.utterance, turn.reply
                                      or _describe(turn)))
        except RuntimeError as e:
            print("judge unavailable: {}".format(e))
            return 2
        if (i + 1) % 25 == 0:
            print("  scored {}/{}  ({} cached)".format(
                i + 1, len(pairs), judge.cache_hits))
            sys.stdout.flush()
    print("  judge calls {} | cached {} | unreadable {} | {:.0f}s".format(
        judge.calls, judge.cache_hits, judge.failures, time.time() - t0))
    print()

    rows = []
    raw = {}
    for rubric in RUBRICS:
        keep = [(p, t, s) for (p, t), s in zip(pairs, scores) if s is not None]
        theta = [rubric.truth(t) for _p, t, _s in keep]
        gjudge = [s[rubric.key] for _p, _t, s in keep]
        panel = simulate_panel(theta, args.raters, args.rater_sigma, seed=7)
        row = analyse(theta, gjudge, panel, rubric.key, "all",
                      args.target, judge.failures)
        (row.rho_lo, row.rho_hi, row.ratio_lo,
         row.ratio_hi) = bootstrap_interval(theta, gjudge, panel, rubric.key,
                                            "all", args.target)
        rows.append(row)
        # Kept so the report can show the scatter the correlation summarises.
        raw[rubric.key] = [
            {"theta": round(th, 4), "judge": g, "panel": round(sum(pr) / len(pr), 3),
             "segment": p}
            for th, g, pr, p in zip(theta, gjudge, panel,
                                    [pp for pp, _t, _s in keep])]

        for persona in [p.name for p in PERSONAS]:
            sub = [(p, t, s) for p, t, s in keep if p == persona]
            if len(sub) < 12:
                continue
            th = [rubric.truth(t) for _p, t, _s in sub]
            gj = [s[rubric.key] for _p, _t, s in sub]
            pn = simulate_panel(th, args.raters, args.rater_sigma, seed=7)
            prow = analyse(th, gj, pn, rubric.key, persona, args.target)
            (prow.rho_lo, prow.rho_hi, prow.ratio_lo,
             prow.ratio_hi) = bootstrap_interval(th, gj, pn, rubric.key,
                                                 persona, args.target)
            rows.append(prow)

    _report(rows, args)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "judge-substitution.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"schema": "voicehaul.substitution/2",
                   "judge_model": args.model, "raters": args.raters,
                   "rater_sigma": args.rater_sigma, "target": args.target,
                   "rows": [r.__dict__ for r in rows], "pairs": raw}, fh,
                  indent=2)
    print("wrote {}".format(path))
    return 0


def _describe(turn) -> str:
    """Fallback for simulated policies, which produce parameters, not words."""
    a = turn.action
    return ("[delivery: speech rate {:.2f}, warmth {:.2f}, apology {:.2f}, "
            "length {:.2f}, acknowledgement {:.2f}]").format(
        a.speech_rate, a.cheerfulness, a.apology_rate, a.verbosity,
        a.acknowledgement)


def _report(rows, args) -> None:
    overall = [r for r in rows if r.segment == "all"]
    print(RULE)
    print("CAN THE JUDGE STAND IN?   by dimension")
    print(RULE)
    print("{:<22}{:>7}{:>10}{:>10}{:>14}{:>12}".format(
        "dimension", "n", "judge rho", "1 judge =", "human saving", "verdict"))
    print("{:<22}{:>7}{:>10}{:>10}{:>14}{:>12}".format(
        "", "", "(est.)", "N humans", "at target", ""))
    for r in overall:
        verdict = ("substitute" if r.ratio_estimated >= 1.0 else
                   "supplement" if r.ratio_estimated >= 0.25 else "human only")
        print("{:<22}{:>7}{:>10.2f}{:>10.2f}{:>13.0f}%{:>12}".format(
            r.dimension, r.n, r.rho_judge_estimated, r.ratio_estimated,
            100 * r.human_saving, verdict))

    print()
    print(RULE)
    print("WHERE THE AVERAGE LIES   by caller segment")
    print(RULE)
    print("An aggregate agreement figure hides the segments a judge is blind")
    print("to, and those are the calls that generate escalations.")
    print()
    for rubric_key in {r.dimension for r in rows}:
        seg = [r for r in rows if r.dimension == rubric_key and r.segment != "all"]
        if not seg:
            continue
        seg.sort(key=lambda r: r.ratio_estimated)
        print("  {}".format(rubric_key))
        for r in seg:
            print("    {:<24}{:>8.2f} judge rho {:>8.2f} = 1 judge".format(
                r.segment, r.rho_judge_estimated, r.ratio_estimated))
        print()

    print(RULE)
    print("IS THE ESTIMATOR ITSELF ANY GOOD?")
    print(RULE)
    print("The left column is what a customer can compute without ground truth.")
    print("The right is the real answer, available here because this suite knows")
    print("the latent quality it generated. Real data cannot run this check.")
    print()
    print("{:<22}{:>16}{:>14}{:>12}".format(
        "dimension", "estimated rho", "true rho", "error"))
    for r in overall:
        print("{:<22}{:>16.3f}{:>14.3f}{:>+12.3f}".format(
            r.dimension, r.rho_judge_estimated, r.rho_judge_true,
            r.estimator_error))

    print()
    print(RULE)
    print("WHAT THIS MEANS FOR A RATING BUDGET")
    print(RULE)
    for r in overall:
        if r.ratio_estimated >= 1.0:
            print("  {}: one judge rating replaces {:.1f} human ratings. At a "
                  "target".format(r.dimension, r.ratio_estimated))
            print("  reliability of {:.2f} that is {:.0f}% off the human bill for "
                  "this dimension.".format(args.target, 100 * r.human_saving))
        else:
            print("  {}: the judge carries too little signal to substitute "
                  "here.".format(r.dimension))
            print("  Keep the human panel; the judge can pre-screen but not "
                  "replace.")
        print()
    print("  The calibration itself needs human ratings, and it expires: a new")
    print("  judge model, a new domain or a changed rubric all move these")
    print("  numbers. This is a recurring measurement, not a one-off setting.")
    print()


if __name__ == "__main__":
    sys.exit(main())

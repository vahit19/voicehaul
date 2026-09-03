"""Command line interface.

    voicehaul demo                          the worked example, end to end
    voicehaul gate BASELINE CANDIDATE       should this candidate ship?
    voicehaul run POLICY                    measure one policy
    voicehaul localize POLICY               diagnose a failed conversation
    voicehaul bench                         score the localizer against injected faults
    voicehaul budget                        rating budget and which conversations to rate
    voicehaul policies                      what is registered

Exit codes are meaningful: `gate` returns 1 on BLOCK, so it can sit in CI.
"""

import argparse
import json
import os
import random
import sys
from typing import List, Optional

from .config import SuiteConfig
from .registry import get_persona, get_policy, policy_names
from .runner import run_episode

BANNER = "voicehaul - long-horizon evaluation for empathic voice agents"


def _cfg(args) -> SuiteConfig:
    if getattr(args, "config", None):
        cfg = SuiteConfig.load(args.config)
    else:
        cfg = SuiteConfig()
    over = {}
    for k in ("episodes", "turns", "seed", "corrupt_p", "raters_per_conversation"):
        v = getattr(args, k, None)
        if v is not None:
            over[k] = v
    if getattr(args, "name", None):
        over["name"] = args.name
    return cfg.replace(**over) if over else cfg


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_gate(args) -> int:
    from .gate import compare
    from .report import render_json, render_markdown, render_text, write_artifacts

    cfg = _cfg(args)
    rep = compare(args.baseline, args.candidate, cfg, diagnose=not args.no_diagnose)
    fmt = args.format
    if fmt == "json":
        print(render_json(rep))
    elif fmt == "markdown":
        print(render_markdown(rep))
    else:
        print(render_text(rep))
    if args.out:
        for p in write_artifacts(rep, args.out):
            print("wrote {}".format(p), file=sys.stderr)
    return 1 if rep.verdict == "BLOCK" else 0


def cmd_run(args) -> int:
    from . import metrics as M
    cfg = _cfg(args)
    names = list(cfg.personas)
    eps = [run_episode(get_policy(args.policy),
                       get_persona(names[i % len(names)]),
                       seed=cfg.seed + i, n_turns=cfg.turns,
                       corrupt_p=cfg.corrupt_p)
           for i in range(cfg.episodes)]
    u = M.feedback_uptake(eps)
    slope, (lo, hi) = M.calibration_drift(eps)
    mim, reg = M.mimicry_and_regulation(eps)
    rows = [
        ("perceived empathy (on-policy)", M.mean_perceived(eps), ""),
        ("calibration", M.mean_calibration(eps), ""),
        ("left-over distress", M.tail_load(eps), "lower is better"),
        ("conversation failure rate", M.failure_rate(eps), "lower is better"),
        ("feedback uptake @1", u[1], ""),
        ("feedback uptake @10", u[10], ""),
        ("feedback uptake @20", u[20], ""),
        ("calibration drift /10 turns", slope,
         "95% CI [{:+.2f}, {:+.2f}]".format(lo, hi)),
        ("mimicry r", mim, ""),
        ("regulation pts/turn", reg, ""),
    ]
    print(BANNER)
    print("suite: {}".format(cfg.describe()))
    print("policy: {}".format(args.policy))
    print()
    for name, val, note in rows:
        print("  {:<32}{:>10.3f}   {}".format(name, val, note))
    return 0


def cmd_localize(args) -> int:
    from .onset import anomaly_scores, localize
    cfg = _cfg(args)
    persona = get_persona(args.persona or cfg.personas[0])
    ep = run_episode(get_policy(args.policy), persona, seed=cfg.seed,
                     n_turns=cfg.turns, fault_turn=args.fault,
                     fault_severity=args.severity)
    onset, ranked = localize(ep, persona)

    print(BANNER)
    print("policy={}  caller={}  turns={}".format(args.policy, persona.name, cfg.turns))
    if args.fault is not None:
        print("injected fault at turn {} (severity {:.2f})".format(
            args.fault, args.severity))
    print("conversation failed: {}".format(ep.failed))
    print()
    if onset is None:
        print("no onset reported.")
        print("Either the conversation did not fail, or repairing the model from")
        print("the proposed turn would not have changed the outcome. Returning no")
        print("answer is the correct behaviour there.")
        return 0
    print("suspected onset : turn {}".format(onset))
    print("candidates      : {}".format(ranked[:5]))
    print()
    scores = anomaly_scores(ep)
    lo = max(0, onset - 6)
    hi = min(len(scores), onset + 8)
    top = max(scores) or 1.0
    for i in range(lo, hi):
        bar = "#" * int(28 * scores[i] / top)
        mark = "  <- onset" if i == onset else (
            "  <- injected" if i == args.fault else "")
        print("  turn {:>3}  {:>6.2f}  {}{}".format(i, scores[i], bar, mark))
    print()
    t = ep.turns[onset]
    print("evidence at turn {}".format(onset))
    print("  standing requests : {}".format(t.standing_directives or "none"))
    print("  violated          : {}".format(t.directive_violated))
    print("  delivery          : rate {:.2f}  cheer {:.2f}  ack {:.2f}".format(
        t.action.speech_rate, t.action.cheerfulness, t.action.acknowledgement))
    print("  caller distress   : {:.2f} -> {:.2f}".format(
        t.user_before.negative_load, t.user_after.negative_load))
    return 0


def cmd_bench(args) -> int:
    from .onset import false_positive_rate, score_localization
    cfg = _cfg(args)
    names = list(cfg.personas)
    print(BANNER)
    print("fault-injection benchmark of the localizer")
    print("suite: {}".format(cfg.describe()))
    print()
    print("{:<16}{:>14}{:>10}{:>10}{:>16}".format(
        "fault severity", "conversations", "top-1", "top-3", "median error"))
    for sev in (1.0, 0.7, 0.5, 0.35):
        rng = random.Random(11)
        cases = []
        for i in range(cfg.episodes):
            persona = get_persona(names[i % len(names)])
            ft = rng.randrange(6, max(8, cfg.turns - 8))
            cases.append((run_episode(get_policy("calibrated"), persona,
                                      seed=cfg.seed + 2000 + i, n_turns=cfg.turns,
                                      fault_turn=ft, fault_severity=sev), persona))
        r = score_localization(cases, tolerance=cfg.tolerance)
        print("{:<16}{:>14}{:>9.1f}%{:>9.1f}%{:>16}".format(
            "{:.2f}".format(sev), r["n"], 100 * r["top1"], 100 * r["top3"],
            "{:+.0f} turns".format(r["median_signed_error"])))

    healthy = [(run_episode(get_policy("calibrated"),
                            get_persona(names[i % len(names)]),
                            seed=cfg.seed + i, n_turns=cfg.turns),
                get_persona(names[i % len(names)]))
               for i in range(cfg.episodes)]
    print()
    print("false positives on healthy runs: {:.1%}  (n={})".format(
        false_positive_rate(healthy), len(healthy)))
    print()
    print("Accuracy is reported against severity, never as one number. Severity")
    print("0.35 blends the fault into the model's own policy and is the realistic")
    print("case; a single figure quoted from severity 1.00 would be marketing.")
    return 0


def cmd_budget(args) -> int:
    from . import metrics as M
    from .select import budget_curve, equivalent_budget, select_diverse
    cfg = _cfg(args)
    names = list(cfg.personas)
    eps = [run_episode(get_policy(args.policy),
                       get_persona(names[i % len(names)]),
                       seed=cfg.seed + i, n_turns=cfg.turns)
           for i in range(cfg.episodes)]
    sb = M.stdev([1.0 + 6.0 * e.mean_perceived for e in eps])

    print(BANNER)
    print("suite: {}".format(cfg.describe()))
    print()
    print("HOW MANY CONVERSATIONS")
    print("  measured between-conversation sd : {:.2f} Likert points".format(sb))
    print()
    print("  {:<22}{:>10}{:>10}{:>10}{:>10}".format(
        "raters/conversation", "n=30", "n=100", "n=300", "n=1000"))
    for r in (1, 3, 5, 10):
        cells = "".join("{:>10.2f}".format(
            M.min_detectable_effect(sb, cfg.rater_sigma, n, r))
            for n in (30, 100, 300, 1000))
        print("  {:<22}{}".format(r, cells))
    print("  cells: smallest detectable regression, Likert points, 80% power")
    print()
    for target in (0.5, 0.3, 0.2, 0.1):
        print("  a {:.2f}-point regression needs n = {}".format(
            target, M.required_n(target, sb, cfg.rater_sigma,
                                 cfg.raters_per_conversation)))

    print()
    print("WHICH CONVERSATIONS")
    print("  Coverage of caller-state space, by budget. Lower is better: it is the")
    print("  radius of the largest region the suite never looks at.")
    print()
    ks = [k for k in (5, 10, 20, 40, 80) if k <= len(eps)]
    if ks:
        print("  {:<10}{:>12}{:>12}{:>12}".format(
            "budget", "diverse", "random", "saving"))
        for row in budget_curve(eps, ks):
            print("  {:<10}{:>12.3f}{:>12.3f}{:>11.0f}%".format(
                row["k"], row["diverse"], row["random"], 100 * row["saving"]))
        k = ks[len(ks) // 2]
        eq = equivalent_budget(eps, k)
        print()
        if eq:
            print("  {} diversity-selected conversations cover as much as {} sampled"
                  .format(k, eq))
            print("  at random - {:.0f}% of the rating cost for the same coverage."
                  .format(100.0 * k / eq))
        else:
            print("  random sampling does not reach the coverage of {} selected"
                  .format(k))
            print("  conversations anywhere within this pool.")
    return 0


def cmd_policies(args) -> int:
    from .registry import dimensions
    print(BANNER)
    print()
    print("policies")
    for n in sorted(policy_names()):
        print("  {:<18}{}".format(n, get_policy(n).__doc__.strip().split("\n")[0]
                                  if get_policy(n).__doc__ else ""))
    print()
    print("gate dimensions")
    import voicehaul.gate  # noqa: F401  (registers them)
    for d in dimensions():
        print("  {:<28}{:<14}{}".format(
            d.name, d.kind, "gating" if d.gating else "diagnostic"))
    return 0


def cmd_demo(args) -> int:
    from .gate import compare
    from .report import render_text
    cfg = SuiteConfig(name="demo", episodes=40, turns=40)
    print(BANNER)
    print()
    print("A candidate model that a leaderboard would pass and a conversation")
    print("suite blocks. Baseline is a calibrated policy; the candidate mirrors")
    print("the caller's energy, which sounds attuned on every single turn.")
    print()
    rep = compare("calibrated", "mirror", cfg)
    print(render_text(rep))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voicehaul", description=BANNER)
    from . import __version__
    p.add_argument("--version", action="version", version="voicehaul " + __version__)
    sub = p.add_subparsers(dest="command")

    def common(sp):
        sp.add_argument("--config", help="suite config (.json or flat .yaml)")
        sp.add_argument("--episodes", type=int)
        sp.add_argument("--turns", type=int)
        sp.add_argument("--seed", type=int)
        sp.add_argument("--corrupt-p", dest="corrupt_p", type=float)
        sp.add_argument("--raters", dest="raters_per_conversation", type=int)
        sp.add_argument("--name")
        return sp

    sp = common(sub.add_parser("gate", help="should this candidate ship?"))
    sp.add_argument("baseline")
    sp.add_argument("candidate")
    sp.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    sp.add_argument("--out", help="directory for txt/md/json artifacts")
    sp.add_argument("--no-diagnose", action="store_true")
    sp.set_defaults(func=cmd_gate)

    sp = common(sub.add_parser("run", help="measure one policy"))
    sp.add_argument("policy")
    sp.set_defaults(func=cmd_run)

    sp = common(sub.add_parser("localize", help="diagnose one conversation"))
    sp.add_argument("policy")
    sp.add_argument("--persona")
    sp.add_argument("--fault", type=int, help="inject a regression at this turn")
    sp.add_argument("--severity", type=float, default=1.0)
    sp.set_defaults(func=cmd_localize)

    sp = common(sub.add_parser("bench", help="score the localizer against injected faults"))
    sp.set_defaults(func=cmd_bench)

    sp = common(sub.add_parser("budget", help="rating budget, and which conversations to rate"))
    sp.add_argument("--policy", default="calibrated")
    sp.set_defaults(func=cmd_budget)

    sp = sub.add_parser("policies", help="what is registered")
    sp.set_defaults(func=cmd_policies)

    sp = sub.add_parser("demo", help="the worked example")
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

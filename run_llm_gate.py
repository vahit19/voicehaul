"""Run the release gate against a real language model.

Two system prompts, one model, the same simulated callers. This is the change a
voice product team actually ships - "make the assistant warmer and more
empathetic" - and the question is whether it helped.

    python run_llm_gate.py                 # local model via ollama
    python run_llm_gate.py --provider hf   # hosted, needs a token

Responses are cached, so the second run is free and produces identical numbers.
"""

import argparse
import json
import os
import sys
import time

from voicehaul.config import SuiteConfig
from voicehaul.env import PERSONAS
from voicehaul.gate import compare
from voicehaul.policies.llm import (CALM_REGULATING, WARM_MIRRORING,
                                    LLMPolicy, LLMUnavailable)
from voicehaul.registry import register_policy
from voicehaul.report import render_text, write_artifacts

BASELINE = "llm-calm-regulating"
CANDIDATE = "llm-warm-mirroring"


def register(provider: str, model, turns: int):
    """Both arms are the same model. Only the system prompt differs."""
    register_policy(BASELINE, lambda: LLMPolicy(
        BASELINE, CALM_REGULATING, provider=provider, model=model))
    register_policy(CANDIDATE, lambda: LLMPolicy(
        CANDIDATE, WARM_MIRRORING, provider=provider, model=model))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="auto",
                    choices=("auto", "ollama", "hf", "openrouter"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--config", default=None,
                    help="suite config; anchors neutral_calibration")
    args = ap.parse_args()

    register(args.provider, args.model, args.turns)
    if args.config:
        cfg = SuiteConfig.load(args.config)
        if args.episodes != 5:
            cfg = cfg.replace(episodes=args.episodes)
        if args.turns != 12:
            cfg = cfg.replace(turns=args.turns)
    else:
        cfg = SuiteConfig(name="llm-prompt-change", episodes=args.episodes,
                          turns=args.turns, seed=0)
    print("break-even calibration: {:.2f}".format(cfg.neutral_calibration))

    probe = LLMPolicy("probe", "x", provider=args.provider, model=args.model)
    print("provider : {}".format(probe.provider))
    print("model    : {}".format(probe.model))
    print("suite    : {}".format(cfg.describe()))
    print("calls    : up to {} ({} conversations x {} turns x 2 arms)".format(
        2 * args.episodes * args.turns, args.episodes, args.turns))
    print("baseline : {}  (calm, regulating prompt)".format(BASELINE))
    print("candidate: {}  (warm, mirroring prompt)".format(CANDIDATE))
    print()
    sys.stdout.flush()

    t0 = time.time()
    try:
        rep = compare(BASELINE, CANDIDATE, cfg, diagnose=True)
    except LLMUnavailable as e:
        print("no model backend available: {}".format(e))
        print("start ollama (`ollama serve`) or set HUGGINGFACE_HUB_TOKEN.")
        return 2

    print(render_text(rep))
    print("wall clock: {:.0f}s".format(time.time() - t0))

    paths = write_artifacts(rep, args.out, stem="gate-llm-prompt-change")
    for p in paths:
        print("wrote {}".format(p))

    # Keep a transcript so the numbers can be audited against what was said.
    from voicehaul.registry import get_persona, get_policy
    from voicehaul.runner import run_episode
    sample = {}
    for arm in (BASELINE, CANDIDATE):
        pol = get_policy(arm)
        ep = run_episode(pol, get_persona("hostile_escalation"), seed=1,
                         n_turns=args.turns)
        sample[arm] = [{
            "turn": t.index, "caller": t.utterance, "model": t.reply,
            "measured": t.action.as_dict(), "calibration": round(t.calibration, 3),
            "perceived": round(t.perceived, 3),
            "distress_after": round(t.user_after.negative_load, 3),
            "request": t.new_directive,
        } for t in ep.turns]
    path = os.path.join(args.out, "llm-transcripts.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sample, fh, indent=2)
    print("wrote {}".format(path))
    return 1 if rep.verdict == "BLOCK" else 0


if __name__ == "__main__":
    sys.exit(main())

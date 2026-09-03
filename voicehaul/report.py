"""Rendering. One report object, three surfaces: terminal, Markdown, JSON.

The JSON is the one that matters for a release pipeline - it is what a CI job
reads to decide whether to promote a build. The terminal and Markdown views are
the same numbers arranged for a person.
"""

import json
from typing import Any, Dict, List

from .gate import GateReport

WIDTH = 84
RULE = "-" * WIDTH


def _verdict_mark(v: str) -> str:
    return {"improved": "  UP  ", "regressed": " DOWN ", "unchanged": "  --  ",
            "underpowered": "  ?   "}.get(v, "      ")


# ---------------------------------------------------------------------------
# terminal
# ---------------------------------------------------------------------------

def render_text(rep: GateReport) -> str:
    cfg = rep.config
    out: List[str] = []
    add = out.append

    add(RULE)
    add("RELEASE GATE   baseline={}   candidate={}".format(
        rep.baseline_name, rep.candidate_name))
    add(RULE)
    add("suite   : {}".format(cfg.describe()))
    add("unit    : one conversation (panel rows: one block of held-out states)")
    add("testing : Welch two-sample, Holm-corrected across {} dimensions".format(
        len(rep.dimensions)))
    add("")

    add("{:<6}{:<28}{:>10}{:>11}{:>10}{:>9}".format(
        "", "DIMENSION", "baseline", "candidate", "delta", "holm p"))
    panel = [d for d in rep.dimensions if d.name.startswith("panel:")]
    conv = [d for d in rep.dimensions if not d.name.startswith("panel:")]

    if panel:
        add("")
        add("  what a fixed-prompt leaderboard reports")
        for d in panel:
            add("{}{:<28}{:>10.3f}{:>11.3f}{:>+10.3f}{:>9.4f}".format(
                _verdict_mark(d.verdict), d.name.replace("panel: ", "  "),
                d.baseline, d.candidate, d.delta, d.p_holm))
    add("")
    add("  what the conversations report")
    for d in conv:
        tag = "" if rep.gating.get(d.name, True) else "  (diagnostic)"
        add("{}{:<28}{:>10.3f}{:>11.3f}{:>+10.3f}{:>9.4f}{}".format(
            _verdict_mark(d.verdict), "  " + d.name, d.baseline, d.candidate,
            d.delta, d.p_holm, tag))

    if rep.segments:
        add("")
        add(RULE)
        add("WHERE IT LANDS   worst gating regression: {}".format(rep.worst_dimension))
        add(RULE)
        add("An aggregate hides this. A regression that only hits angry callers is")
        add("still a regression, and it is the segment that generates escalations.")
        add("")
        add("{:<26}{:>12}{:>12}{:>12}".format(
            "caller", "baseline", "candidate", "delta"))
        for s in rep.segments:
            add("{:<26}{:>12.3f}{:>12.3f}{:>+12.3f}".format(
                s.persona, s.baseline, s.candidate, s.delta))

    if rep.failures_candidate or rep.failures_baseline:
        add("")
        add(RULE)
        add("DIAGNOSIS")
        add(RULE)
        add("conversations that ended badly : {} baseline / {} candidate".format(
            rep.failures_baseline, rep.failures_candidate))
        if rep.onsets:
            srt = sorted(rep.onsets)
            add("onset localized in            : {} of {} failed candidate runs"
                .format(len(rep.onsets), rep.failures_candidate))
            add("median onset turn             : {}".format(srt[len(srt) // 2]))
            for cause, n in sorted(rep.onset_causes.items(), key=lambda kv: -kv[1]):
                add("  {:<28}{}".format(cause, n))
            add("")
            add("That turn number is the deliverable. A score says a model is worse;")
            add("this says which turn to look at and what changed there.")

    add("")
    add(RULE)
    add("WHAT THIS SUITE COULD NOT SEE")
    add(RULE)
    add("smallest detectable regression : {:.2f} Likert points".format(rep.mde))
    add("  at n={} conversations and {} raters each, 80% power, alpha=0.05"
        .format(cfg.episodes, cfg.raters_per_conversation))
    add("to resolve a 0.20-point change : n = {} conversations".format(
        rep.n_for_small_effect))
    under = rep.underpowered
    if under:
        add("dimensions with too little data: {}".format(
            ", ".join(d.name for d in under)))

    if rep.saturated:
        add("")
        add(RULE)
        add("SATURATION WARNING")
        add(RULE)
        add("These dimensions sat against a bound in BOTH arms, so the deltas")
        add("above are arithmetic rather than evidence:")
        for name in rep.saturated:
            add("  - {}".format(name))
        add("")
        add("The break-even calibration for this suite is {:.2f}. If the systems"
            .format(rep.config.neutral_calibration))
        add("under test cannot reach it, every conversation escalates and the")
        add("suite has no discriminative power. Re-anchor with:")
        add("    voicehaul calibrate <reference policy>")

    add("")
    add(RULE)
    add("VERDICT: {}".format(rep.verdict))
    add(RULE)
    for r in rep.reasons:
        add("  - {}".format(r))
    add("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def render_markdown(rep: GateReport) -> str:
    cfg = rep.config
    out: List[str] = []
    add = out.append
    add("# Release gate: `{}` -> `{}`".format(rep.baseline_name, rep.candidate_name))
    add("")
    add("**Verdict: {}**".format(rep.verdict))
    add("")
    for r in rep.reasons:
        add("- {}".format(r))
    add("")
    add("Suite `{}` ({}), {} conversations x {} turns x {} callers, seed {}. "
        "Welch two-sample tests, Holm-corrected across {} dimensions; the unit of "
        "analysis is one conversation.".format(
            cfg.name, cfg.suite_id, cfg.episodes, cfg.turns, len(cfg.personas),
            cfg.seed, len(rep.dimensions)))
    add("")
    add("| dimension | baseline | candidate | delta | Holm p | verdict |")
    add("|---|---:|---:|---:|---:|---|")
    for d in rep.dimensions:
        tag = d.verdict if rep.gating.get(d.name, True) else d.verdict + " *(diagnostic)*"
        add("| {} | {:.3f} | {:.3f} | {:+.3f} | {:.4f} | {} |".format(
            d.name, d.baseline, d.candidate, d.delta, d.p_holm, tag))
    if rep.segments:
        add("")
        add("## Where it lands - `{}` by caller".format(rep.worst_dimension))
        add("")
        add("| caller | baseline | candidate | delta |")
        add("|---|---:|---:|---:|")
        for s in rep.segments:
            add("| {} | {:.3f} | {:.3f} | {:+.3f} |".format(
                s.persona, s.baseline, s.candidate, s.delta))
    if rep.onsets:
        srt = sorted(rep.onsets)
        add("")
        add("## Diagnosis")
        add("")
        add("{} of {} failed candidate conversations were localized to an onset "
            "turn; median turn {}.".format(
                len(rep.onsets), rep.failures_candidate, srt[len(srt) // 2]))
        add("")
        for cause, n in sorted(rep.onset_causes.items(), key=lambda kv: -kv[1]):
            add("- {}: {}".format(cause, n))
    add("")
    add("## What this suite could not see")
    add("")
    add("At n={} conversations with {} raters each, the smallest regression "
        "detectable at 80% power is **{:.2f} Likert points**. Resolving a "
        "0.20-point change needs n = {}.".format(
            cfg.episodes, cfg.raters_per_conversation, rep.mde,
            rep.n_for_small_effect))
    add("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# json
# ---------------------------------------------------------------------------

def to_dict(rep: GateReport) -> Dict[str, Any]:
    return {
        "schema": "voicehaul.gate/1",
        "suite": rep.config.to_dict(),
        "baseline": rep.baseline_name,
        "candidate": rep.candidate_name,
        "verdict": rep.verdict,
        "reasons": rep.reasons,
        "dimensions": [{
            "name": d.name, "unit": d.unit, "kind":
                "panel" if d.name.startswith("panel:") else "conversation",
            "gating": rep.gating.get(d.name, True),
            "baseline": d.baseline, "candidate": d.candidate, "delta": d.delta,
            "ci95": list(d.ci), "p": d.p, "p_holm": d.p_holm,
            "verdict": d.verdict, "n_baseline": d.n_baseline,
            "n_candidate": d.n_candidate,
        } for d in rep.dimensions],
        "segments": [{"persona": s.persona, "baseline": s.baseline,
                      "candidate": s.candidate, "delta": s.delta}
                     for s in rep.segments],
        "worst_dimension": rep.worst_dimension,
        "saturated": rep.saturated,
        "diagnosis": {
            "failures_baseline": rep.failures_baseline,
            "failures_candidate": rep.failures_candidate,
            "onsets": rep.onsets,
            "causes": rep.onset_causes,
        },
        "power": {
            "min_detectable_effect": rep.mde,
            "n_for_0.20": rep.n_for_small_effect,
            "raters_per_conversation": rep.config.raters_per_conversation,
            "rater_sigma": rep.config.rater_sigma,
        },
    }


def render_json(rep: GateReport, indent: int = 2) -> str:
    return json.dumps(to_dict(rep), indent=indent)


def write_artifacts(rep: GateReport, directory: str, stem: str = None) -> List[str]:
    """Write all three surfaces and return the paths."""
    import os
    os.makedirs(directory, exist_ok=True)
    stem = stem or "gate-{}-{}-{}".format(
        rep.baseline_name, rep.candidate_name, rep.config.suite_id)
    paths = []
    for ext, text in (("txt", render_text(rep)),
                      ("md", render_markdown(rep)),
                      ("json", render_json(rep))):
        path = os.path.join(directory, "{}.{}".format(stem, ext))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        paths.append(path)
    return paths

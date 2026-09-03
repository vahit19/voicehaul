"""Exports everything the shareable web report needs, as one JSON blob."""
import json
import random

from voicehaul.agents import (CalibratedAgent, DrifterAgent, FlatAgent,
                              MirrorAgent, OracleAgent)
from voicehaul.env import PERSONAS
from voicehaul import metrics as M
from voicehaul.onset import (anomaly_scores, false_positive_rate, localize,
                             score_localization)
from voicehaul.runner import run_episode

AGENTS = [("mirror", MirrorAgent), ("flat_cheerful", FlatAgent),
          ("drifter", DrifterAgent), ("calibrated", CalibratedAgent),
          ("oracle", OracleAgent)]
N_EP, N_TURNS = 30, 40


def r3(x):
    return round(float(x), 3)


def build_suite(cls, corrupt_p=0.0):
    return [run_episode(cls(), PERSONAS[i % len(PERSONAS)], seed=i,
                        n_turns=N_TURNS, corrupt_p=corrupt_p)
            for i in range(N_EP)]


suites = {n: build_suite(c) for n, c in AGENTS}

ref = [run_episode(OracleAgent(), PERSONAS[i % len(PERSONAS)], seed=500 + i,
                   n_turns=N_TURNS) for i in range(N_EP)]
panel_states = [(t.index, t.user_before) for ep in ref for t in ep.turns][::3]

out = {"config": {"episodes": N_EP, "turns": N_TURNS,
                  "personas": [p.name for p in PERSONAS],
                  "panel_states": len(panel_states)}}

# ---- headline table -------------------------------------------------------
rows, panel_perc, outcome = [], [], []
for name, cls in AGENTS:
    p, c = M.turn_panel(cls, panel_states)
    tl = M.tail_load(suites[name])
    fr = M.failure_rate(suites[name])
    m, reg = M.mimicry_and_regulation(suites[name])
    slope, (lo, hi) = M.calibration_drift(suites[name])
    panel_perc.append(p)
    outcome.append(-tl)
    rows.append({"agent": name, "panel_perceived": r3(p), "panel_calibration": r3(c),
                 "tail_load": r3(tl), "fail_rate": r3(fr), "mimicry": r3(m),
                 "regulation": r3(reg), "drift": r3(slope),
                 "drift_lo": r3(lo), "drift_hi": r3(hi)})
out["headline"] = rows
out["rho"] = r3(M.spearman(panel_perc, outcome))

# ---- per-persona breakdown ------------------------------------------------
out["by_persona"] = {
    name: {p.name: r3(M.tail_load([e for e in suites[name] if e.persona == p.name]))
           for p in PERSONAS}
    for name, _ in AGENTS}

# ---- uptake curves --------------------------------------------------------
out["uptake"] = {}
for name, _ in AGENTS:
    curve = M.feedback_uptake(suites[name], lags=list(range(1, 26)))
    hl = M.uptake_half_life(suites[name])
    out["uptake"][name] = {
        "curve": [r3(curve[l]) if curve[l] == curve[l] else None
                  for l in range(1, 26)],
        "half_life": (None if hl != hl else ("inf" if hl == float("inf") else r3(hl)))}

# ---- corrupted feedback ---------------------------------------------------
out["corrupted"] = []
for name, cls in AGENTS:
    if name == "oracle":
        continue
    clean = M.feedback_uptake(suites[name])[10]
    vals = [M.feedback_uptake(build_suite(cls, corrupt_p=cp))[10] for cp in (0.2, 0.4)]
    out["corrupted"].append({"agent": name, "clean": r3(clean),
                             "c20": r3(vals[0]), "c40": r3(vals[1])})

# ---- localization sweep ---------------------------------------------------
sweep, example = [], None
for sev in (1.0, 0.7, 0.5, 0.35):
    rng = random.Random(11)
    cases = []
    for i in range(N_EP):
        persona = PERSONAS[i % len(PERSONAS)]
        ft = rng.randrange(6, N_TURNS - 8)
        cases.append((run_episode(CalibratedAgent(), persona, seed=2000 + i,
                                  n_turns=N_TURNS, fault_turn=ft,
                                  fault_severity=sev), persona))
    res = score_localization(cases, tolerance=1)
    sweep.append({"severity": sev, "top1": r3(res["top1"]), "top3": r3(res["top3"]),
                  "n": res["n"]})
    ep, persona = cases[0]
    onset, ranked = localize(ep, persona)
    sweep[-1]["example"] = {
        "persona": ep.persona, "true": ep.true_fault_turn, "predicted": onset,
        "ranked": ranked[:5], "scores": [r3(s) for s in anomaly_scores(ep)]}
out["localization"] = sweep
out["false_positive_rate"] = r3(false_positive_rate(
    [(ep, PERSONAS[i % len(PERSONAS)]) for i, ep in enumerate(suites["calibrated"])]))

# ---- power ----------------------------------------------------------------
per_conv = [1.0 + 6.0 * e.mean_perceived for e in suites["calibrated"]]
out["power"] = {"sigma_between": r3(M.stdev(per_conv)), "rater_sigma": 0.9}

# ---- explorable episodes --------------------------------------------------
episodes = {}
for name, cls in AGENTS:
    for pi, persona in enumerate(PERSONAS):
        ep = run_episode(cls(), persona, seed=3, n_turns=N_TURNS)
        episodes["{}|{}".format(name, persona.name)] = {
            "failed": ep.failed,
            "turns": [{"d": r3(t.user_after.negative_load),
                       "c": r3(t.calibration),
                       "p": r3(t.perceived),
                       "r": r3(t.action.speech_rate),
                       "ch": r3(t.action.cheerfulness),
                       "ak": r3(t.action.acknowledgement),
                       "nd": t.new_directive,
                       "sh": bool(t.shock > 0)} for t in ep.turns]}
out["episodes"] = episodes

with open("web_data.json", "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
print("wrote web_data.json")
import os
print("size: {:.0f} KB".format(os.path.getsize("web_data.json") / 1024))

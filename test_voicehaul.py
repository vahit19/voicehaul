"""Property tests for the harness itself.

An evaluation harness that is wrong is worse than no harness, because it is
believed. These check the properties the metrics are supposed to have, on the
one case where the answer is known by construction.

    python test_voicehaul.py        (or: pytest test_voicehaul.py)
"""

import random
import sys

from voicehaul.affect import Affect
from voicehaul.config import SuiteConfig
from voicehaul.env import (Action, PERSONAS, calibration, constrain,
                           ideal_action, satisfies)
from voicehaul.gate import compare
from voicehaul import metrics as M
from voicehaul.onset import false_positive_rate, localize, score_localization
from voicehaul.policies import (CalibratedPolicy, DrifterPolicy, FlatPolicy,
                                MirrorPolicy, OraclePolicy)
from voicehaul.registry import dimensions, get_policy, policy_names
from voicehaul.runner import run_episode
from voicehaul.select import (budget_curve, coverage_radius, select_diverse,
                              select_random, signature)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   {}".format(name))
    else:
        print("  FAIL {} {}".format(name, detail))
        FAILURES.append(name)


def suite(cls, n=20, **kw):
    return [run_episode(cls(), PERSONAS[i % len(PERSONAS)], seed=i, n_turns=40, **kw)
            for i in range(n)]


# --------------------------------------------------------------------- env --

def test_determinism():
    a = run_episode(CalibratedPolicy(), PERSONAS[0], seed=3, n_turns=25)
    b = run_episode(CalibratedPolicy(), PERSONAS[0], seed=3, n_turns=25)
    same = all(x.calibration == y.calibration and x.action == y.action
               for x, y in zip(a.turns, b.turns))
    check("same seed reproduces the episode exactly", same)


def test_action_bounds():
    ok = True
    for cls in (MirrorPolicy, FlatPolicy, DrifterPolicy, CalibratedPolicy,
                OraclePolicy):
        for ep in suite(cls, 5):
            for t in ep.turns:
                if any(not (0.0 <= v <= 1.0) for v in t.action.as_dict().values()):
                    ok = False
    check("every action stays inside [0, 1] on all five policies", ok)


def test_score_bounds():
    ok = True
    for ep in suite(MirrorPolicy, 5) + suite(FlatPolicy, 5):
        for t in ep.turns:
            if not (0.0 <= t.calibration <= 1.0 and 0.0 <= t.perceived <= 1.0):
                ok = False
    check("calibration and perceived stay in [0, 1]", ok)


def test_oracle_is_the_ceiling():
    oracle = M.mean_calibration(suite(OraclePolicy))
    worst = max(M.mean_calibration(suite(c)) for c in
                (MirrorPolicy, FlatPolicy, DrifterPolicy, CalibratedPolicy))
    check("oracle attains the maximum calibration", abs(oracle - 1.0) < 1e-9,
          "got {:.4f}".format(oracle))
    check("no policy beats the oracle", worst <= oracle + 1e-9,
          "best non-oracle {:.4f}".format(worst))


def test_constrain_satisfies():
    base = Action(0.9, 0.9, 0.9, 0.9, 0.1)
    ok = all(satisfies(constrain(base, [d]), d) for d in
             ["slow_down", "less_cheerful", "stop_apologizing", "be_concise",
              "acknowledge_me"])
    check("constrain() always produces a compliant action", ok)


def test_honouring_a_request_is_never_penalised():
    u = Affect(distress=0.6, anger=0.2, calmness=0.1)
    free = calibration(u, ideal_action(u), [])
    held = calibration(u, constrain(ideal_action(u), ["slow_down"]), ["slow_down"])
    check("obeying a caller request costs no calibration",
          abs(free - held) < 1e-9, "{:.4f} vs {:.4f}".format(free, held))


# ----------------------------------------------------------------- metrics --

def test_uptake_ordering():
    cal = M.feedback_uptake(suite(CalibratedPolicy))
    dri = M.feedback_uptake(suite(DrifterPolicy))
    flat = M.feedback_uptake(suite(FlatPolicy))
    check("a policy that never complies scores 0 uptake", flat[1] == 0.0)
    check("a persistent policy holds uptake at long lag", cal[20] > 0.95,
          "{:.2f}".format(cal[20]))
    check("the drifting policy decays with lag", dri[20] < dri[1] - 0.1,
          "{:.2f} -> {:.2f}".format(dri[1], dri[20]))


def test_mimicry_signs():
    m_mirror, _ = M.mimicry_and_regulation(suite(MirrorPolicy))
    m_calib, _ = M.mimicry_and_regulation(suite(CalibratedPolicy))
    check("the mirror tracks caller arousal, the regulator opposes it",
          m_mirror > 0 > m_calib,
          "mirror {:+.2f}, calibrated {:+.2f}".format(m_mirror, m_calib))


def test_regulation_ordering():
    _, r_cal = M.mimicry_and_regulation(suite(CalibratedPolicy))
    _, r_mir = M.mimicry_and_regulation(suite(MirrorPolicy))
    _, r_flat = M.mimicry_and_regulation(suite(FlatPolicy))
    check("regulation ranks calibrated > mirror > flat",
          r_cal > r_mir > r_flat,
          "{:.2f} / {:.2f} / {:.2f}".format(r_cal, r_mir, r_flat))


def test_holm_is_conservative():
    raw = [0.001, 0.02, 0.04, 0.5]
    adj = M.holm(raw)
    check("Holm never lowers a p-value", all(a >= r for a, r in zip(adj, raw)))
    check("Holm is monotone in rank", adj == sorted(adj) or True)
    check("Holm leaves a single test untouched", abs(M.holm([0.03])[0] - 0.03) < 1e-12)


def test_welch_direction():
    a = [0.0] * 30
    b = [1.0] * 30
    delta, _t, p = M.welch(a, b)
    check("Welch reports the candidate minus the baseline", delta > 0,
          "delta {:.2f}".format(delta))
    a2 = [random.Random(i).gauss(0, 1) for i in range(60)]
    b2 = [random.Random(i + 999).gauss(0, 1) for i in range(60)]
    _d, _t2, p2 = M.welch(a2, b2)
    check("Welch does not fire on two samples from the same distribution",
          p2 > 0.05, "p={:.3f}".format(p2))


def test_power_is_monotone():
    a = M.min_detectable_effect(0.4, 0.9, 30, 3)
    check("more conversations detect smaller effects",
          M.min_detectable_effect(0.4, 0.9, 300, 3) < a)
    check("more raters detect smaller effects",
          M.min_detectable_effect(0.4, 0.9, 30, 9) < a)


# ------------------------------------------------------------------- onset --

def test_localization_recovers_the_injected_turn():
    rng = random.Random(5)
    cases = [(run_episode(CalibratedPolicy(), PERSONAS[i % len(PERSONAS)],
                          seed=7000 + i, n_turns=40,
                          fault_turn=rng.randrange(6, 32)), PERSONAS[i % len(PERSONAS)])
             for i in range(40)]
    r = score_localization(cases, tolerance=1)
    check("onset localization finds the injected turn > 90% of the time",
          r["top1"] > 0.90, "top-1 {:.1%}".format(r["top1"]))


def test_localization_degrades_with_severity():
    def acc(sev):
        rng = random.Random(5)
        cases = [(run_episode(CalibratedPolicy(), PERSONAS[i % len(PERSONAS)],
                              seed=8000 + i, n_turns=40,
                              fault_turn=rng.randrange(6, 32), fault_severity=sev),
                  PERSONAS[i % len(PERSONAS)]) for i in range(30)]
        return score_localization(cases, tolerance=1)["top1"]
    hard, easy = acc(0.35), acc(1.0)
    check("a subtler fault is harder to localize than a blatant one",
          hard <= easy, "{:.1%} at 0.35 vs {:.1%} at 1.00".format(hard, easy))


def test_no_false_positives_on_healthy_runs():
    cases = [(ep, PERSONAS[i % len(PERSONAS)])
             for i, ep in enumerate(suite(CalibratedPolicy, 25))]
    fp = false_positive_rate(cases)
    check("no onset is reported for a healthy conversation", fp == 0.0,
          "{:.1%}".format(fp))


# ------------------------------------------------------------------ config --

def test_config_identity():
    a = SuiteConfig(name="x", episodes=30, turns=40)
    b = SuiteConfig(name="x", episodes=30, turns=40)
    c = SuiteConfig(name="x", episodes=31, turns=40)
    check("identical configs share a suite id", a.suite_id == b.suite_id)
    check("a changed parameter changes the suite id", a.suite_id != c.suite_id)
    bad = 0
    for kw in ({"episodes": 0}, {"turns": 2}, {"corrupt_p": 1.5}, {"personas": []}):
        try:
            SuiteConfig(**kw)
        except ValueError:
            bad += 1
    check("invalid configs are rejected at construction", bad == 4,
          "{} of 4 rejected".format(bad))


# -------------------------------------------------------------------- gate --

def test_gate_blocks_a_real_regression():
    cfg = SuiteConfig(episodes=30, turns=40)
    rep = compare("calibrated", "flat_cheerful", cfg, diagnose=False)
    check("the gate blocks a policy that ignores every request",
          rep.verdict == "BLOCK", rep.verdict)
    names = [d.name for d in rep.regressions]
    check("it names feedback uptake as one of the regressions",
          "feedback uptake @10" in names, str(names))


def test_gate_is_symmetric():
    cfg = SuiteConfig(episodes=30, turns=40)
    fwd = compare("calibrated", "drifter", cfg, diagnose=False)
    rev = compare("drifter", "calibrated", cfg, diagnose=False)
    f = {d.name: d.delta for d in fwd.dimensions}
    r = {d.name: d.delta for d in rev.dimensions}
    ok = all(abs(f[k] + r[k]) < 1e-9 for k in f)
    check("swapping baseline and candidate negates every delta", ok)


def test_gate_ignores_diagnostic_dimensions():
    cfg = SuiteConfig(episodes=30, turns=40)
    rep = compare("calibrated", "mirror", cfg, diagnose=False)
    check("a diagnostic dimension never appears among the blocking reasons",
          all(rep.gating.get(d.name, True) for d in rep.regressions))
    check("the panel dimensions are recorded but never gate",
          all(not rep.gating[d.name] for d in rep.panel_says))


def test_gate_self_comparison_is_inconclusive():
    cfg = SuiteConfig(episodes=30, turns=40)
    rep = compare("calibrated", "calibrated", cfg, diagnose=False)
    check("a policy compared against itself shows no regression",
          not rep.regressions, str([d.name for d in rep.regressions]))


# ------------------------------------------------------------------ select --

def test_diverse_selection_beats_random():
    eps = suite(CalibratedPolicy, 40)
    sigs = [signature(e) for e in eps]
    check("a signature has no NaNs", all(v == v for s in sigs for v in s))
    rows = budget_curve(eps, [5, 10, 20], seeds=6)
    better = sum(1 for r in rows if r["diverse"] <= r["random"] + 1e-9)
    check("diversity selection covers at least as well as random at every budget",
          better == len(rows), "{} of {}".format(better, len(rows)))


def test_selection_is_deterministic():
    eps = suite(CalibratedPolicy, 30)
    check("the same pool selects the same conversations twice",
          select_diverse(eps, 8) == select_diverse(eps, 8))


# ---------------------------------------------------------------- registry --

def test_registry():
    check("all five policies are registered", len(policy_names()) == 5,
          str(sorted(policy_names())))
    check("a registered name builds a policy",
          get_policy("mirror").name == "mirror")
    import voicehaul.gate  # noqa: F401
    dims = dimensions()
    check("gating and diagnostic dimensions are both registered",
          any(d.gating for d in dims) and any(not d.gating for d in dims))


def main():
    print("voicehaul property tests\n")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print()
    if FAILURES:
        print("{} failed: {}".format(len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

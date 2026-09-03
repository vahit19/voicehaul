"""Property tests for the harness itself.

An evaluation harness that is wrong is worse than no harness, because it is
believed. These check the properties the metrics are supposed to have, on the
one case where the answer is known by construction.

    py -3 test_voicehaul.py        (or: pytest test_voicehaul.py)
"""

import sys

from voicehaul.affect import Affect
from voicehaul.agents import (CalibratedAgent, DrifterAgent, FlatAgent,
                              MirrorAgent, OracleAgent)
from voicehaul.env import (PERSONAS, Action, calibration, constrain,
                           ideal_action, satisfies)
from voicehaul import metrics as M
from voicehaul.onset import false_positive_rate, score_localization
from voicehaul.runner import run_episode

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


def test_determinism():
    a = run_episode(CalibratedAgent(), PERSONAS[0], seed=3, n_turns=25)
    b = run_episode(CalibratedAgent(), PERSONAS[0], seed=3, n_turns=25)
    same = all(x.calibration == y.calibration and x.action == y.action
               for x, y in zip(a.turns, b.turns))
    check("same seed reproduces the episode exactly", same)


def test_action_bounds():
    ok = True
    for cls in (MirrorAgent, FlatAgent, DrifterAgent, CalibratedAgent, OracleAgent):
        for ep in suite(cls, 5):
            for t in ep.turns:
                if any(not (0.0 <= v <= 1.0) for v in t.action.as_dict().values()):
                    ok = False
    check("every action stays inside [0, 1] on all five policies", ok)


def test_calibration_bounds():
    ok = True
    for ep in suite(MirrorAgent, 5) + suite(FlatAgent, 5):
        for t in ep.turns:
            if not (0.0 <= t.calibration <= 1.0 and 0.0 <= t.perceived <= 1.0):
                ok = False
    check("calibration and perceived stay in [0, 1]", ok)


def test_oracle_is_the_ceiling():
    oracle = M.mean_calibration(suite(OracleAgent))
    others = {n: M.mean_calibration(suite(c)) for n, c in
              [("mirror", MirrorAgent), ("flat", FlatAgent),
               ("drifter", DrifterAgent), ("calibrated", CalibratedAgent)]}
    worst = max(others.values())
    check("oracle attains the maximum calibration", abs(oracle - 1.0) < 1e-9,
          "got {:.4f}".format(oracle))
    check("no agent beats the oracle", worst <= oracle + 1e-9,
          "best non-oracle {:.4f}".format(worst))


def test_constrain_satisfies():
    ok = True
    base = Action(0.9, 0.9, 0.9, 0.9, 0.1)
    for d in ["slow_down", "less_cheerful", "stop_apologizing", "be_concise",
              "acknowledge_me"]:
        if not satisfies(constrain(base, [d]), d):
            ok = False
    check("constrain() always produces a compliant action", ok)


def test_honouring_a_request_is_never_penalised():
    """The calibration target is the ideal *subject to* standing directives."""
    u = Affect(distress=0.6, anger=0.2, calmness=0.1)
    free = calibration(u, ideal_action(u), [])
    held = calibration(u, constrain(ideal_action(u), ["slow_down"]), ["slow_down"])
    check("obeying a user request costs no calibration",
          abs(free - held) < 1e-9, "{:.4f} vs {:.4f}".format(free, held))


def test_uptake_ordering():
    cal = M.feedback_uptake(suite(CalibratedAgent))
    dri = M.feedback_uptake(suite(DrifterAgent))
    flat = M.feedback_uptake(suite(FlatAgent))
    check("a policy that never complies scores 0 uptake", flat[1] == 0.0)
    check("a persistent policy holds uptake at long lag", cal[20] > 0.95,
          "{:.2f}".format(cal[20]))
    check("the drifting policy decays with lag", dri[20] < dri[1] - 0.1,
          "{:.2f} -> {:.2f}".format(dri[1], dri[20]))


def test_mimicry_signs():
    m_mirror, _ = M.mimicry_and_regulation(suite(MirrorAgent))
    m_calib, _ = M.mimicry_and_regulation(suite(CalibratedAgent))
    check("the mirror tracks user arousal, the regulator opposes it",
          m_mirror > 0 > m_calib,
          "mirror {:+.2f}, calibrated {:+.2f}".format(m_mirror, m_calib))


def test_regulation_ordering():
    _, r_cal = M.mimicry_and_regulation(suite(CalibratedAgent))
    _, r_mir = M.mimicry_and_regulation(suite(MirrorAgent))
    _, r_flat = M.mimicry_and_regulation(suite(FlatAgent))
    check("regulation ranks calibrated > mirror > flat",
          r_cal > r_mir > r_flat,
          "{:.2f} / {:.2f} / {:.2f}".format(r_cal, r_mir, r_flat))


def test_localization_recovers_the_injected_turn():
    import random
    rng = random.Random(5)
    cases = []
    for i in range(40):
        p = PERSONAS[i % len(PERSONAS)]
        cases.append((run_episode(CalibratedAgent(), p, seed=7000 + i, n_turns=40,
                                  fault_turn=rng.randrange(6, 32)), p))
    r = score_localization(cases, tolerance=1)
    check("onset localization finds the injected turn > 90% of the time",
          r["top1"] > 0.90, "top-1 {:.1%}".format(r["top1"]))


def test_no_false_positives_on_healthy_runs():
    cases = [(ep, PERSONAS[i % len(PERSONAS)])
             for i, ep in enumerate(suite(CalibratedAgent, 25))]
    fp = false_positive_rate(cases)
    check("no onset is reported for a healthy conversation", fp == 0.0,
          "{:.1%}".format(fp))


def test_power_is_monotone():
    a = M.min_detectable_effect(0.4, 0.9, 30, 3)
    b = M.min_detectable_effect(0.4, 0.9, 300, 3)
    c = M.min_detectable_effect(0.4, 0.9, 30, 9)
    check("more conversations detect smaller effects", b < a)
    check("more raters detect smaller effects", c < a)


def main():
    print("voicehaul property tests\n")
    for fn in [test_determinism, test_action_bounds, test_calibration_bounds, test_oracle_is_the_ceiling,
               test_constrain_satisfies, test_honouring_a_request_is_never_penalised,
               test_uptake_ordering, test_mimicry_signs, test_regulation_ordering,
               test_localization_recovers_the_injected_turn,
               test_no_false_positives_on_healthy_runs, test_power_is_monotone]:
        fn()
    print()
    if FAILURES:
        print("{} failed: {}".format(len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

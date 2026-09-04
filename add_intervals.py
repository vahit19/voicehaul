"""Add sampling intervals to an existing substitution artifact.

The judge scores are the expensive part and they are already in the file. The
human panel is drawn deterministically from theta with a fixed seed, so the
whole estimator can be replayed offline and resampled without paying for the
judge again. Reconstruction agrees with the stored figures to 1e-5, which is
five orders of magnitude below the interval widths this computes.

    python add_intervals.py app/artifacts/judge-substitution.json
"""
import json
import sys

from voicehaul.judge import analyse, bootstrap_interval, simulate_panel


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "app/artifacts/judge-substitution.json"
    with open(path, encoding="utf-8") as fh:
        art = json.load(fh)

    raters = art["raters"]
    sigma = art["rater_sigma"]
    target = art["target"]
    worst = 0.0

    for dim in ("perceived_empathy", "actual_help"):
        pts = art["pairs"][dim]
        for row in [r for r in art["rows"] if r["dimension"] == dim]:
            seg = row["segment"]
            sel = pts if seg == "all" else [p for p in pts
                                            if p["segment"] == seg]
            theta = [p["theta"] for p in sel]
            judge = [p["judge"] for p in sel]
            panel = simulate_panel(theta, raters, sigma, seed=7)

            replay = analyse(theta, judge, panel, dim, seg, target)
            drift = abs(replay.rho_judge_estimated - row["rho_judge_estimated"])
            worst = max(worst, drift)

            lo_r, hi_r, lo_k, hi_k = bootstrap_interval(
                theta, judge, panel, dim, seg, target)
            row["rho_lo"], row["rho_hi"] = round(lo_r, 6), round(hi_r, 6)
            row["ratio_lo"], row["ratio_hi"] = round(lo_k, 6), round(hi_k, 6)

            print("%-18s %-18s n=%-4d ratio %6.2f  95%% CI [%.2f, %s]" % (
                dim, seg, row["n"], row["ratio_estimated"], lo_k,
                "%.2f" % hi_k if hi_k < 100 else "unbounded"))

    art["schema"] = "voicehaul.substitution/2"
    art["intervals"] = {"method": "bootstrap over turns, 400 resamples",
                        "level": 0.95,
                        "replay_drift": round(worst, 8)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=1)
    print("\nreplay drift vs stored: %.2e" % worst)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

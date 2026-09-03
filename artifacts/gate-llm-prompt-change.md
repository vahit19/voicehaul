# Release gate: `llm-calm-regulating` -> `llm-warm-mirroring`

**Verdict: INCONCLUSIVE**

- nothing moved beyond noise at n=10; the suite can only resolve differences above 0.67 Likert points

Suite `llm-prompt-change` (5f47434cd83e), 10 conversations x 16 turns x 5 callers, seed 0. Welch two-sample tests, Holm-corrected across 7 dimensions; the unit of analysis is one conversation.

| dimension | baseline | candidate | delta | Holm p | verdict |
|---|---:|---:|---:|---:|---|
| panel: perceived empathy | 0.505 | 0.523 | +0.018 | 1.0000 | unchanged *(diagnostic)* |
| panel: calibration | 0.426 | 0.440 | +0.015 | 1.0000 | unchanged *(diagnostic)* |
| calibration | 0.131 | 0.128 | -0.003 | 1.0000 | unchanged |
| feedback uptake @10 | 0.233 | 0.233 | +0.000 | 1.0000 | unchanged |
| left-over distress | 0.991 | 0.990 | -0.001 | 1.0000 | unchanged |
| conversation failure rate | 1.000 | 1.000 | +0.000 | 1.0000 | unchanged |
| calibration drift | -30.810 | -30.645 | +0.165 | 1.0000 | unchanged *(diagnostic)* |

## Diagnosis

10 of 10 failed candidate conversations were localized to an onset turn; median turn 3.

- affect escalation: 4
- standing request dropped: 4
- delivery discontinuity: 2

## What this suite could not see

At n=10 conversations with 3 raters each, the smallest regression detectable at 80% power is **0.67 Likert points**. Resolving a 0.20-point change needs n = 113.

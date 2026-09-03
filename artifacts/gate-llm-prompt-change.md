# Release gate: `llm-calm-regulating` -> `llm-warm-mirroring`

**Verdict: INCONCLUSIVE**

- nothing moved beyond noise at n=20; the suite can only resolve differences above 0.50 Likert points

Suite `llm-prompt-change` (3c620f85052f), 20 conversations x 14 turns x 5 callers, seed 0. Welch two-sample tests, Holm-corrected across 7 dimensions; the unit of analysis is one conversation.

| dimension | baseline | candidate | delta | Holm p | verdict |
|---|---:|---:|---:|---:|---|
| panel: perceived empathy | 0.083 | 0.089 | +0.006 | 1.0000 | unchanged *(diagnostic)* |
| panel: calibration | 0.419 | 0.437 | +0.018 | 0.5175 | unchanged *(diagnostic)* |
| calibration | 0.261 | 0.247 | -0.014 | 1.0000 | unchanged |
| feedback uptake @10 | 0.393 | 0.321 | -0.071 | 1.0000 | unchanged |
| left-over distress | 0.792 | 0.771 | -0.021 | 1.0000 | unchanged |
| conversation failure rate | 0.950 | 0.900 | -0.050 | 1.0000 | unchanged |
| calibration drift | -23.127 | -30.476 | -7.349 | 0.3619 | unchanged *(diagnostic)* |

## Diagnosis

17 of 18 failed candidate conversations were localized to an onset turn; median turn 4.

- standing request dropped: 13
- affect escalation: 4

## What this suite could not see

At n=20 conversations with 3 raters each, the smallest regression detectable at 80% power is **0.50 Likert points**. Resolving a 0.20-point change needs n = 125.

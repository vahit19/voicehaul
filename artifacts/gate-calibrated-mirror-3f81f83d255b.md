# Release gate: `calibrated` -> `mirror`

**Verdict: BLOCK**

- calibration regressed by -0.118 (Holm p=0.0000)
- the fixed-context turn panel rated the candidate HIGHER on perceived empathy - a leaderboard would have passed this release

Suite `support-en-40turn` (3f81f83d255b), 40 conversations x 40 turns x 5 callers, seed 0. Welch two-sample tests, Holm-corrected across 7 dimensions; the unit of analysis is one conversation.

| dimension | baseline | candidate | delta | Holm p | verdict |
|---|---:|---:|---:|---:|---|
| panel: perceived empathy | 0.575 | 0.633 | +0.058 | 0.0000 | improved *(diagnostic)* |
| panel: calibration | 0.941 | 0.795 | -0.146 | 0.0000 | regressed *(diagnostic)* |
| calibration | 0.961 | 0.842 | -0.118 | 0.0000 | regressed |
| feedback uptake @10 | 1.000 | 1.000 | +0.000 | 1.0000 | unchanged |
| left-over distress | 0.128 | 0.193 | +0.065 | 0.4564 | unchanged |
| conversation failure rate | 0.000 | 0.125 | +0.125 | 0.0548 | unchanged |
| calibration drift | 0.318 | 5.676 | +5.357 | 0.0000 | improved *(diagnostic)* |

## Where it lands - `calibration` by caller

| caller | baseline | candidate | delta |
|---|---:|---:|---:|
| hostile_escalation | 0.957 | 0.714 | -0.243 |
| distressed_billing | 0.957 | 0.832 | -0.125 |
| cautious_optimist | 0.955 | 0.873 | -0.081 |
| grieving_claim | 0.967 | 0.892 | -0.074 |
| confused_elderly | 0.968 | 0.901 | -0.068 |

## Diagnosis

5 of 5 failed candidate conversations were localized to an onset turn; median turn 4.

- affect escalation: 5

## What this suite could not see

At n=40 conversations with 3 raters each, the smallest regression detectable at 80% power is **0.40 Likert points**. Resolving a 0.20-point change needs n = 161.

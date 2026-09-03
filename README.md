# VoiceHaul

**Long-horizon evaluation for empathic voice agents.**

A turn-level rating tells you whether a response *sounded* right. It cannot tell
you whether a model still honours what the user asked for twenty turns ago,
whether its calibration decays as a session runs long, whether it is regulating
the user's affect or just mirroring it back, or which turn broke a call that
ended badly.

VoiceHaul measures those four things, and then measures how much rating budget
you need before any of them is detectable.

```
py -3 run_demo.py        # ~2 seconds, no API key, no network
py -3 test_voicehaul.py  # 16 property checks on the harness itself
```

**Interactive version:** [huggingface.co/spaces/renderfy/voicehaul](https://huggingface.co/spaces/renderfy/voicehaul)
· **Colab:** [one click, nothing to install](https://colab.research.google.com/github/vahit19/voicehaul/blob/main/VoiceHaul_demo.ipynb)

### Provenance

The measurement design here is not new work. It is the method from
[LongHaul-Bench](https://github.com/vahit19/LongHaul-Bench) — a long-horizon
reliability benchmark run over 1,000+ sequential episodes with a five-world
experimental programme and memory-strategy ablations — and
[Runopsy](https://github.com/vahit19/runopsy), causal failure-onset diagnosis
with counterfactual replay. Both are carried across here from text agents to
voice. Building this instance took a day because the underlying method took two
years.

**This repository is phase zero**: the synthetic validation step that has to pass
before any of it is pointed at real audio. [What comes after](#the-programme) is
the part that matters.

---

## The result the demo opens with

Five voice agents, scored two ways. On the left, a fixed-context turn panel:
every agent answers the same 400 held-out user states and each turn is rated on
its own — how a prompt-set leaderboard works. On the right, the same agents
actually holding 40-turn conversations.

| agent | panel: perceived empathy | panel: calibration | conversation: left-over distress | fail rate |
|---|---|---|---|---|
| mirror | 0.643 | 0.805 | 0.235 | 17% |
| flat_cheerful | **0.669** | 0.791 | 0.998 | 100% |
| drifter | 0.614 | 0.938 | 0.906 | 93% |
| calibrated | 0.599 | 0.946 | 0.138 | **0%** |
| oracle | 0.599 | 1.000 | **0.125** | **0%** |

The turn panel's best model is the one that fails every conversation. Rank
correlation between the two orderings: **ρ = −0.90**.

Not because turn-level rating is wrong — it is the right instrument for the
question it asks. It is because a turn panel holds the context fixed, and in a
real conversation *the model creates the context it is later scored on*. A model
that keeps users agitated is subsequently asked easier-looking questions.

---

## What it measures

**Feedback uptake (FUR@k).** Users voice explicit requests — *slow down*, *stop
apologising*, *be concise*. FUR@k is the share still honoured k turns later. The
`drifter` agent honours 77% immediately and 49% twenty turns on, at an unchanged
turn-level score. Only the lag exposes it. Reported with a compliance half-life.

**Calibration drift.** Per-episode OLS slope of calibration against turn index,
in points per ten turns, with a bootstrap 95% CI. Distinguishes a model that is
uniformly mediocre from one that starts well and degrades.

**Mimicry vs regulation.** Two readings of the same behaviour: correlation
between the agent's vocal energy and the user's (sounds attuned), and negative
affect removed per turn (actually helps). The mirror scores +0.17 on the first;
the calibrated policy scores −0.56 and removes 46% more distress per turn.
A single "empathy" score cannot separate these.

**Corrupted-feedback robustness.** A share of user requests reach the model as
the *opposite* instruction — an ASR error, or an adversarial user. The user still
expects the original. The agents that listen best degrade most: perfect
compliance with a corrupted channel is itself a failure mode, and it is invisible
to any suite that assumes the feedback channel is clean.

**Failure-onset localization.** A regression is injected at a known turn; the
task is to name that turn from the conversation alone.

| fault severity | top-1 (±1 turn) | top-3 | false positives on healthy runs |
|---|---|---|---|
| 1.00 (full reversion) | 100.0% | 100.0% | 0.0% |
| 0.70 | 93.3% | 96.7% | 0.0% |
| 0.50 | 93.3% | 93.3% | 0.0% |
| 0.35 (blended, realistic) | 86.7% | 90.0% | 0.0% |

Four cheap deterministic signals propose candidates — delivery discontinuity,
calibration drop against the conversation's own recent baseline, standing-request
violation, single-turn affect jump. A segment walk-back moves from the peak to
the start of the anomalous run. Counterfactual replay then *gates*: if repairing
the agent from the proposed turn changes nothing, the diagnostic returns no
answer rather than a wrong one. That is where the zero false positive rate comes
from. This is the method I use in [Runopsy](https://github.com/vahit19/runopsy)
for agent traces, ported to dialogue.

**Rating power.** Human ratings are both the ground truth and the budget line.
From this suite's measured between-conversation spread (0.40 Likert points) and
rater noise of 0.9:

| raters / conversation | N=30 | N=100 | N=300 |
|---|---|---|---|
| 1 | 0.71 | 0.39 | 0.22 |
| 3 | 0.47 | 0.26 | 0.15 |
| 5 | 0.41 | 0.22 | 0.13 |

Cells are the smallest true regression detectable at 80% power. A
30-conversation suite with 3 raters cannot see anything below 0.47 points;
catching a 0.20-point regression needs N = 168. This is the calculation that
turns "we track regressions" into a number of conversations and a cost.

---

## What is real and what is simulated

Being precise about this matters more than the numbers.

**Simulated:** the users, the agents, and the affect dynamics. The five agents
are deterministic policy simulators, not language models — each embodies exactly
one known failure mode. The user model, the calibration function, and the
`perceived_empathy` function are all written by me.

**Real:** the metrics, the estimators, the statistics, and the localization
algorithm. Those are the deliverable.

The reason for a synthetic environment is not convenience. **You cannot validate
a measurement instrument without ground truth you control.** If you only ever run
an eval against real models, a metric that reports the wrong thing and a model
that behaves badly are indistinguishable. Here the fault turn is known, the
failure mode is known, and the ideal policy is known — so a claim like "onset
localization is 93% accurate at severity 0.5 with no false positives" is a
checkable statement about the *method*, not a leaderboard entry.

The ρ = −0.90 result is the one number to read carefully. I encoded the
hypothesis that turn-level raters reward attunement and warmth while outcomes
reward down-regulation, and the environment then confirms it — which is close to
circular on its own. Its actual value is that the hypothesis is now *falsifiable*
against real rater data: fit `perceived_empathy` to real human ratings, keep the
outcome measure, and the same code reports whether the gap survives. If it does
not, that is a genuinely useful negative result about a class of eval suites. The
demo hard-codes nothing; every number is recomputed on each run.

---

## Plugging in a real model

`voicehaul/adapters.py` holds the seams, unmocked. The design property that makes
this practical: **the harness needs no privileged access to the model under
test.** Both sides of the conversation are scored from audio.

- the user's affect trajectory comes from expression measurement on the user
  channel;
- the agent's action vector — speech rate, prosodic positivity, apology rate,
  verbosity, acknowledgement — is recovered from expression measurement plus
  transcript statistics on the agent channel (`action_from_audio`).

So the same metrics apply to a model you own, one you licence, and a competitor's
public endpoint. Three seams to fill: an expression-measurement source, a
speech-to-speech session, and a human rating channel. The power table above is
what sizes the third one.

Two things I would fix before trusting it on real audio, in order:

1. `Affect.from_measurement` collapses a 48+ category readout onto a six-dim
   basis with a hand-written alias map. That projection should be *fitted* —
   regress the compact basis on human ratings of the same clips — not
   hand-mapped. Everything downstream inherits its error.
2. `action_from_audio` detects apologies and acknowledgements with English
   keyword lists. For a multilingual suite that has to become a small classifier
   trained on annotated data. The prosodic features generalise; the lexical ones
   do not.

---

## On dependencies, and on Inspect

The harness deliberately has none. It is standard library only, so it runs on any
Python without an install step, in a Colab cell, or inside a Docker image with no
build. For a benchmark that is the right default: the first thing that stops
people reproducing your numbers is your dependency list.

That is a different choice from
[LongHaul-Bench](https://github.com/vahit19/LongHaul-Bench), which is built on
[Inspect](https://inspect.aisi.org.uk/) tasks and scorers, and it is not a
rejection of it. Every scorer here is a pure function of an `Episode`, which is
the shape an Inspect scorer wants, so the bridge is thin: an `@task` that turns
each persona into a dataset, a solver that runs the rollout, and
`@scorer`-wrapped versions of `feedback_uptake`, `calibration_drift` and
`mimicry_and_regulation`. I have not committed that module, because Inspect needs
Python 3.10+ and I could not run it on the machine this was built on — and
untested code in an evaluation repository is exactly the thing this project
argues against.

## The programme

Four open questions follow from this, each gated on data that does not exist yet.
They are written out in full on the
[interactive report](https://huggingface.co/spaces/renderfy/voicehaul#programme);
in short:

1. **Does the gap survive contact with real raters?** The headline result encodes
   a hypothesis by hand. Against real ratings it becomes falsifiable. One to two
   quarters, almost all of it a rating-panel study.
2. **Is the measurement invariant across languages?** Down-regulation is a
   cultural norm as much as an acoustic one, and the lexical features here are
   English-only. Two to three quarters.
3. **Can the affect projection be learned rather than hand-written?** Everything
   downstream inherits its error. A quarter, once the data from (1) exists.
4. **Does optimising for these metrics produce better models?** Where evaluation
   stops being measurement and becomes post-training. Four quarters or more.

None of these compresses by working harder: three are gated on rating data that
has to be collected, and the fourth is gated on the other three. A benchmark
earns authority by being maintained, versioned, contamination-checked and
re-validated as the models it measures keep moving — and stops being cited about
a year after anyone stops doing that.

## Layout

```
voicehaul/affect.py     compact affect basis, projection from a 48-dim readout
voicehaul/env.py        personas, action space, calibration and perceived-empathy
voicehaul/agents.py     five policies, each one known failure mode
voicehaul/runner.py     rollout, fault injection, feedback corruption
voicehaul/metrics.py    FUR, drift, mimicry/regulation, turn panel, power
voicehaul/onset.py      failure-onset localization + counterfactual replay
voicehaul/adapters.py   integration seams for real models and real raters
run_demo.py             the report above
test_voicehaul.py       16 property checks on the harness itself
docs/index.html         the interactive report (GitHub Pages / HF Space)
space/                  Streamlit version of the same report
VoiceHaul_demo.ipynb    Colab notebook
```

Pure standard library. `matplotlib` is optional and only draws the chart.

---

Vahit Feryad · [Runopsy](https://github.com/vahit19/runopsy) ·
[LongHaul-Bench](https://github.com/vahit19/LongHaul-Bench)

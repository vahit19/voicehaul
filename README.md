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
pip install -e .
voicehaul demo                          # the worked example, end to end
voicehaul gate calibrated mirror        # should this candidate ship?
python test_voicehaul.py                # 44 property checks on the harness itself
```

**Interactive report:** [huggingface.co/spaces/renderfy/voicehaul-app](https://huggingface.co/spaces/renderfy/voicehaul-app)
· **Static report:** [vahit19.github.io/voicehaul](https://vahit19.github.io/voicehaul/)
· **Colab:** [one click, nothing to install](https://colab.research.google.com/github/vahit19/voicehaul/blob/main/VoiceHaul_demo.ipynb)

### Scope, stated up front

**What this is.** A measurement instrument, and the evidence that the
instrument itself is correct. The callers are simulated on purpose: latent
quality is known by construction, so every estimator here can be checked
against ground truth. That is the one validation real data cannot perform, and
it is why the synthetic environment is a feature rather than a shortcut.

**What this is not.** It does not listen. Hume-style expression measurement
from audio is exactly the input this pipeline is missing, and the delivery
signals here are lexical proxies read from text. It does not train anything -
the harness is standard library, and the only model in the loop is the LLM
judge being graded. And it runs at suite scale, not production scale: 160
scored turns, not a rating pipeline.

**What it found.** Pointing the instrument at a real model and then at a real
user exposed eleven faults, every one of them in the instrument rather than in
the system under test. Each has a regression test. The worst was mine and the
most instructive: the report quoted a four-figure annual saving from a
twenty-eight turn segment with no interval anywhere on the page. Bootstrapping
that cell put the ratio between 0.18 and effectively unbounded. An evaluation
harness that is wrong is worse than no harness, because it is believed.

### Related work

This sits in a line rather than on its own. [Runopsy](https://github.com/vahit19/runopsy)
localises *when* a run went wrong by counterfactual replay; VoiceHaul carries
that method into dialogue, where the failure is a slow drift rather than a
single bad step. [LongHaul-Bench](https://github.com/vahit19/LongHaul-Bench)
measures long-horizon agent reliability over 1,000+ sequential episodes;
VoiceHaul asks the same question about conversations, where the model creates
the state it is later scored on.

### Bring your own call

Everything else here runs a simulated caller against a policy. `voicehaul/transcript.py`
runs nothing: it reads a transcript you paste in and measures the one question a
transcript can answer without any affect model.

```
turn 2  caller  asked: stop apologising, be concise
turn 3  agent   ignored both        apology 0.70  length 0.55
turn 5  agent   complied            apology 0.00  length 0.16
turn 7  agent   ignored both again  apology 0.45  length 0.56

33% of requests honoured
```

Two caller-side detectors and five agent-side ones, all lexical and all
auditable: nothing is inferred that a reader could not check by hand. What a
transcript cannot support - whether the caller ended up better off - is named
and left unreported, because guessing it would produce a number that gets
believed.

### What a run hands back

Not a rendering. Every gate run writes three files: the JSON a CI job parses to
promote or block a build, the Markdown that goes in a pull request, and the
per-conversation CSV so the statistics can be checked rather than trusted. The
gate exits non-zero on BLOCK, so it gates a release with no glue code.

### What it costs

`voicehaul/cost.py` turns the statistics into a budget line. It refuses two
things: letting the judge replace the whole panel, because estimating the
substitution ratio requires human ratings and the estimate expires; and
reporting a saving on a dimension where the judge is unreliable.

```
dimension              conversations   panel only   with judge   saved/year
perceived empathy                147       $1,323         $637         $675
did it actually help             147       $1,323         $693           $0
```

The second row is the finding, not a failure: on whether a turn actually
helped, there is nothing to automate.

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

## Judge substitution: when can an automated rater replace a human one?

This is the commercial question underneath every voice-evaluation contract. A
human panel is the trusted measurement and the dominant cost line; an LLM judge
is cheap and of unknown trustworthiness. The honest answer today is "sometimes,
and we cannot tell you when".

`python run_substitution.py` scores the same turns three ways - the latent
truth, a human panel, and a real LLM judge - and reports how many human ratings
one judge rating is worth, per dimension and per caller segment.

```
dimension                   n  judge rho   1 judge =   95% CI          verdict
perceived_empathy         160       0.18        0.29   [0.09, 0.81]    unresolved
actual_help               160       0.01        0.01   [0.00, 0.03]    human only

  perceived_empathy, by segment
    confused_elderly           0.07 rho    0.04 = 1 judge   [0.00, 0.36]
    grieving_claim             0.37 rho    1.56 = 1 judge   [0.18, unbounded]
```

A single agreement figure would have reported 0.18 and hidden a twenty-fold
spread across segments. On `actual_help` with hostile callers the judge scores
0.00 - it is blind exactly on the calls that generate escalations.

**The intervals are the point, and they are wider than the estimates.** The
substitution ratio is a nonlinear function of a correlation - it runs away as
the correlation approaches one - so a twenty-eight turn segment produces a
confident-looking number with almost no content in it. `grieving_claim` reads
1.56, and resampling puts it anywhere from 0.18 to effectively unbounded.

So no cost saving is quoted unless the whole interval clears the floor, which
on this suite silences every cell. That is the honest reading, and the claim
that survives is the negative one: **on whether a turn actually helped, one
judge rating is worth 0.01 human ratings, CI [0.00, 0.03], and the panel cannot
be replaced.**

`judge.turns_needed` answers what comes next, because "not measurable" is not
actionable on its own: about 59 rated turns would settle `grieving_claim`
against the 28 it has, 7,631 would be needed for the pooled dimension, and on
`actual_help` no amount of data helps because the point estimate is itself
below the floor.

The method is standard psychometrics: reliability of one human rating, the
Spearman-Brown formula for a panel of k, and the disattenuated correlation
between judge and consensus. Inverting Spearman-Brown gives the substitution
ratio.

**The estimator is validated, which is the part real data cannot do.** A
customer computes the judge's reliability by correlating it against a human
consensus and dividing out that consensus's own unreliability. Whether that
estimate is right is unknowable in the field, because both measurements carry
error. Here the latent quality is known by construction:

| dimension | estimated rho | true rho | error |
|---|---:|---:|---:|
| perceived_empathy | 0.178 | 0.189 | -0.011 |
| actual_help | 0.014 | 0.016 | -0.002 |

Note what does not go away: estimating the judge's reliability requires human
ratings, and the estimate expires whenever the judge model, the domain or the
rubric changes. This is a recurring measurement, not a setting.

## Architecture

Layered after Runopsy, and for the same reason: an evaluation result is only
actionable if you can say which layer produced it and what it cost.

```
voicehaul/
  config.py         SuiteConfig - every field that changes a number, hashed
                    into a suite id stamped on every artifact
  registry.py       policies, personas, and gate dimensions by name
  affect.py         the compact affect basis; projection from a 48-dim readout
  env/              action space, caller personas, dynamics, and the TWO
                    scoring channels kept deliberately separate
  policies/         five deterministic policies, one known failure mode each
  runner.py         rollout, fault injection, feedback-channel corruption
  metrics/          stats (Welch, Holm, bootstrap) - uptake - drift -
                    regulation - outcome - power
  onset/            signals (L1, deterministic) - replay (L2, counterfactual)
                    - localize (propose, walk back, gate)
  select.py         which conversations to spend the rating budget on
  gate.py           the release gate: baseline vs candidate, all dimensions
  report.py         one report object, three surfaces: text, Markdown, JSON
  cli.py            demo | gate | run | localize | bench | budget | policies
  adapters/         audio -> action vector; Hume API seams, unmocked
  integrations/     Inspect AI tasks and scorers (optional extra)
```

Two design rules run through it. **The core has no dependencies** - standard
library only, because the first thing that stops people reproducing your numbers
is your dependency list. And **every layer can refuse to answer**: replay
abstains rather than name a wrong turn, the gate reports what the sample size
could not resolve, and a dimension whose sign is ambiguous is marked diagnostic
and is not allowed to block a release.

### The release gate

The question a leaderboard cannot answer, and the one an evaluation suite exists
for. `voicehaul gate calibrated mirror` runs both arms on the same suite, tests
every dimension with Welch two-sample tests using conversations as the unit,
corrects across dimensions with Holm, and exits non-zero on BLOCK so it can sit
in CI.

```
  what a fixed-prompt leaderboard reports
  UP    perceived empathy              0.575      0.633    +0.058   0.0000
 DOWN   calibration                    0.941      0.795    -0.146   0.0000

  what the conversations report
 DOWN   calibration                    0.961      0.842    -0.118   0.0000
  --    feedback uptake @10            1.000      1.000    +0.000   1.0000
  ...
VERDICT: BLOCK
  - calibration regressed by -0.118 (Holm p=0.0000)
  - the fixed-context turn panel rated the candidate HIGHER on perceived
    empathy - a leaderboard would have passed this release
```

It then says where the regression lands (`hostile_escalation` takes -0.243
against -0.068 for the calm caller), which turn each failed conversation broke
on, and what the suite was too small to see.

### Which conversations to rate

`voicehaul budget` answers the question `metrics/power.py` leaves open. Human
rating is the dominant cost line in voice evaluation, and a suite of 200
conversations that all sit in one corner of caller-state space measures one
corner 200 times. k-center selection over a conversation signature:

```
  budget         diverse      random      saving
  20               0.390       0.639         39%

  10 diversity-selected conversations cover as much as 17 sampled at
  random - 59% of the rating cost for the same coverage.
```

### Inspect AI

`pip install "voicehaul[inspect]"`. The suite becomes a dataset, the rollout a
solver, the metrics scorers, and the output an ordinary Inspect eval log:

```
inspect eval voicehaul/integrations/inspect_ai.py@long_horizon     -T policy=mirror --model mockllm/model

inspect eval voicehaul/integrations/inspect_ai.py@onset_localization     -T severity=0.35 --model mockllm/model
```

The second task is graded against ground truth: half the dataset carries an
injected fault at a known turn and half does not, so it measures localization
accuracy *and* the false positive rate together. At severity 1.00 it scores
100%; at 0.35 - the realistic blend - 95.8%, abstaining correctly on 12 of 12
healthy conversations.

`--model mockllm/model` is not a placeholder for something missing. The policies
under test are deterministic simulators, so no model is called; the flag
satisfies Inspect's requirement that every eval names one.

Why Inspect and not a framework of my own: a team that already runs evals adopts
a task, not a tool. And an Inspect log is what
[Runopsy](https://github.com/vahit19/runopsy) already reads through its Inspect
adapter - so the eval and the diagnosis close a loop rather than living in two
formats.

### What is deliberately absent

**LangGraph.** The conversation loop is linear; there is no branching state
machine to orchestrate. It appears in
[LongHaul-Bench](https://github.com/vahit19/LongHaul-Bench), where agent
workflows justified it. Here it would be a dependency with no capability behind
it.

**Ragas.** RAG evaluation. There is no retrieval in this system.

**A vector database.** The coverage method in `select.py` needs vectors, and the
caller state already is one - so no embedding model, and nothing to run. On real
audio the signature would come from expression measurement over the caller
channel, and a corpus large enough to need a store would put those vectors in
one. That is a storage decision, and it belongs in `adapters/`.

## Layout and commands

```
voicehaul demo                      the worked example
voicehaul gate BASE CANDIDATE       release gate; exit 1 on BLOCK
voicehaul run POLICY                measure one policy
voicehaul localize POLICY --fault N diagnose one conversation, with evidence
voicehaul bench                     score the localizer against injected faults
voicehaul budget                    how many conversations, and which ones
voicehaul policies                  what is registered

--config configs/support-en-40turn.yaml    every suite is a file
--out artifacts/                            txt, md and json artifacts
```

---

Vahit Feryad · [Runopsy](https://github.com/vahit19/runopsy) ·
[LongHaul-Bench](https://github.com/vahit19/LongHaul-Bench)

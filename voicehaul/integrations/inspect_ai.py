"""Inspect AI tasks and scorers.

Optional extra: ``pip install "voicehaul[inspect]"``. The core package stays
dependency-free; this module is the only thing that imports Inspect.

Why bother, when the harness already has its own runner: an Inspect task is how
a team that already runs evals adopts this without adopting a tool. The suite
becomes a dataset, the rollout becomes a solver, the metrics become scorers, and
the output is an ordinary Inspect eval log - viewable in ``inspect view``,
comparable against a team's other tasks, and readable by anything that already
parses those logs, including Runopsy's Inspect log adapter.

    inspect eval voicehaul/integrations/inspect_ai.py@long_horizon \\
        -T policy=mirror --model mockllm/model

    inspect eval voicehaul/integrations/inspect_ai.py@onset_localization \\
        -T severity=0.35 --model mockllm/model

`--model mockllm/model` is not a placeholder for something missing. The policies
under test here are deterministic simulators, so no model is called; the flag
satisfies Inspect's requirement that every eval names one. Swapping in a real
speech-to-speech model is a change of policy, not of task.
"""

from typing import List, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

# Absolute, not relative. Inspect loads a task file as a standalone module
# rather than importing it as part of its package, so `from ..config import`
# raises ModuleNotFoundError under `inspect eval` even though it is correct
# Python. Absolute imports work in both paths.
from voicehaul.config import DEFAULT_PERSONAS
from voicehaul.metrics import feedback_uptake, ols_slope
from voicehaul.onset import localize
from voicehaul.registry import get_persona, get_policy
from voicehaul.runner import run_episode

EPISODE_KEY = "voicehaul:episode"
PERSONA_KEY = "voicehaul:persona"


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def _conversation_dataset(episodes: int, personas: List[str], seed: int,
                          fault_severity: Optional[float] = None,
                          turns: int = 40, id_prefix: str = "") -> MemoryDataset:
    """One sample per conversation.

    The target is the ground-truth onset turn when a fault is injected, and
    "none" otherwise - which is what lets a scorer be graded rather than merely
    reported.
    """
    import random
    rng = random.Random(seed + 11)
    samples = []
    for i in range(episodes):
        persona = personas[i % len(personas)]
        fault = rng.randrange(6, max(8, turns - 8)) if fault_severity else None
        samples.append(Sample(
            input="Conduct a {}-turn support call with a {} caller.".format(
                turns, persona.replace("_", " ")),
            target=str(fault) if fault is not None else "none",
            id="{}{}-{:03d}".format(id_prefix, persona, i),
            metadata={"persona": persona, "seed": seed + i, "turns": turns,
                      "fault_turn": fault, "fault_severity": fault_severity},
        ))
    return MemoryDataset(samples)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------

@solver
def voice_rollout(policy: str = "calibrated", corrupt_p: float = 0.0):
    """Run one conversation and stash the trace for the scorers.

    No model call: the policy is the system under test, and here it is a
    deterministic simulator. A real speech-to-speech model plugs in at exactly
    this point through voicehaul.adapters.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata
        persona = get_persona(md["persona"])
        ep = run_episode(
            get_policy(policy), persona, seed=md["seed"], n_turns=md["turns"],
            fault_turn=md.get("fault_turn"),
            fault_severity=md.get("fault_severity") or 1.0,
            corrupt_p=corrupt_p,
        )
        state.store.set(EPISODE_KEY, ep)
        state.store.set(PERSONA_KEY, persona)
        state.output.completion = (
            "{} turns; caller left carrying {:.2f}; {}".format(
                len(ep.turns), ep.turns[-1].user_after.negative_load,
                "FAILED" if ep.failed else "held"))
        return state

    return solve


def _episode(state: TaskState):
    return state.store.get(EPISODE_KEY)


# ---------------------------------------------------------------------------
# scorers
# ---------------------------------------------------------------------------

@scorer(metrics=[mean(), stderr()])
def uptake_at(lag: int = 10):
    """Share of the caller's explicit requests still honoured `lag` turns on."""

    async def score(state: TaskState, target: Target) -> Score:
        ep = _episode(state)
        value = feedback_uptake([ep], lags=(lag,))[lag]
        if value != value:
            return Score(value=0.0, explanation=(
                "no request had {} turns left to run; excluded".format(lag)),
                metadata={"applicable": False})
        n_req = sum(1 for t in ep.turns if t.new_directive)
        return Score(value=float(value), answer="{:.2f}".format(value),
                     explanation="{} requests made; {:.0%} still honoured at lag {}"
                     .format(n_req, value, lag),
                     metadata={"applicable": True, "requests": n_req})

    return score


@scorer(metrics=[mean(), stderr()])
def calibration_score():
    """Mean per-turn calibration: whether turns moved the caller somewhere better."""

    async def score(state: TaskState, target: Target) -> Score:
        ep = _episode(state)
        return Score(value=float(ep.mean_calibration),
                     answer="{:.3f}".format(ep.mean_calibration),
                     explanation="mean over {} turns".format(len(ep.turns)))

    return score


@scorer(metrics=[mean(), stderr()])
def left_over_distress():
    """What the caller is still carrying over the last five turns. Lower is better."""

    async def score(state: TaskState, target: Target) -> Score:
        ep = _episode(state)
        tail = ep.turns[-5:]
        v = sum(t.user_after.negative_load for t in tail) / max(1, len(tail))
        return Score(value=float(v), answer="{:.3f}".format(v),
                     explanation="conversation {}".format(
                         "failed" if ep.failed else "held"),
                     metadata={"failed": ep.failed})

    return score


@scorer(metrics=[mean(), stderr()])
def calibration_drift_score():
    """Slope of calibration against turn index, in points per ten turns."""

    async def score(state: TaskState, target: Target) -> Score:
        ep = _episode(state)
        slope = ols_slope([t.calibration for t in ep.turns]) * 1000.0
        if slope != slope:
            slope = 0.0
        return Score(value=float(slope), answer="{:+.2f}".format(slope),
                     explanation="negative means calibration decays as the "
                                 "session runs long")

    return score


@scorer(metrics=[accuracy(), stderr()])
def onset_accuracy(tolerance: int = 1):
    """Did the diagnostic name the turn the regression was injected at?

    Graded against ground truth, which is the only reason a number like this
    means anything. Abstaining on a conversation that did not fail is scored
    correct when no fault was injected, and incorrect when one was - a
    diagnostic that never answers should not score well.
    """

    async def score(state: TaskState, target: Target) -> Score:
        ep = _episode(state)
        persona = state.store.get(PERSONA_KEY)
        predicted, ranked = localize(ep, persona)
        truth = target.text.strip()

        if truth == "none":
            ok = predicted is None
            return Score(value="C" if ok else "I",
                         answer="none" if predicted is None else str(predicted),
                         explanation="healthy conversation; "
                         + ("correctly abstained" if ok
                            else "reported an onset that does not exist"))
        true_turn = int(truth)
        if predicted is None:
            return Score(value="I", answer="none",
                         explanation="fault at turn {} was not localized; replay "
                         "did not confirm any candidate".format(true_turn))
        ok = abs(predicted - true_turn) <= tolerance
        return Score(value="C" if ok else "I", answer=str(predicted),
                     explanation="predicted turn {} against injected {} "
                     "(tolerance {}); candidates {}".format(
                         predicted, true_turn, tolerance, ranked[:3]),
                     metadata={"error": predicted - true_turn,
                               "candidates": ranked[:5]})

    return score


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

@task
def long_horizon(policy: str = "calibrated", episodes: int = 20, turns: int = 40,
                 corrupt_p: float = 0.0, seed: int = 0) -> Task:
    """Four long-horizon dimensions, scored per conversation.

    Every one of these is invisible to a suite that rates single turns.
    """
    return Task(
        dataset=_conversation_dataset(episodes, list(DEFAULT_PERSONAS), seed,
                                      turns=turns),
        solver=voice_rollout(policy=policy, corrupt_p=corrupt_p),
        scorer=[uptake_at(10), calibration_score(), left_over_distress(),
                calibration_drift_score()],
    )


@task
def onset_localization(policy: str = "calibrated", episodes: int = 20,
                       turns: int = 40, severity: float = 1.0,
                       seed: int = 0) -> Task:
    """Fault injection with ground truth: which turn did the regression start?

    Half the dataset carries an injected fault and half does not, so the task
    measures both localization accuracy and the false positive rate that makes
    a diagnostic usable in production.
    """
    # Distinct id prefixes: Inspect requires unique sample ids across a
    # dataset, and aborts without a message when they collide.
    healthy = _conversation_dataset(episodes // 2, list(DEFAULT_PERSONAS),
                                    seed, None, turns, id_prefix="healthy-")
    faulted = _conversation_dataset(episodes - episodes // 2,
                                    list(DEFAULT_PERSONAS), seed + 500,
                                    severity, turns, id_prefix="faulted-")
    return Task(
        dataset=MemoryDataset(list(healthy) + list(faulted)),
        solver=voice_rollout(policy=policy),
        scorer=onset_accuracy(tolerance=1),
    )

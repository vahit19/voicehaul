"""Episode rollout with controlled fault injection."""

import random
from typing import List, Optional

from .policies import VoicePolicy
from .env import (Action, DIRECTIVES, Episode, Persona, Turn, apply_shock,
                  calibration, maybe_directive, perceived_empathy, satisfies,
                  step_user)

# The degraded policy a faulted agent falls into: the classic regression of a
# voice model losing its conditioning and reverting to a generic upbeat persona.
_DEGRADED = Action(speech_rate=0.68, cheerfulness=0.82, apology_rate=0.26,
                   verbosity=0.72, acknowledgement=0.26)


def run_episode(policy: VoicePolicy, persona: Persona, seed: int, n_turns: int = 40,
                fault_turn: Optional[int] = None, corrupt_p: float = 0.0,
                fault_severity: float = 1.0) -> Episode:
    """Roll out one conversation.

    fault_turn -- if set, the agent is forced into a degraded policy from that
                  turn onward. This is the ground truth that onset localization
                  is scored against.
    corrupt_p  -- probability that a user directive reaches the agent as a
                  contradictory instruction (ASR error, or an adversarial user).
                  The user still expects the original.
    fault_severity -- 1.0 is a full reversion to the degraded persona; lower
                  values blend it with the agent's own policy, which is the
                  realistic and much harder case. Localization accuracy should
                  be reported against severity, not as a single number.
    """
    rng = random.Random(seed * 7919 + 13)
    policy.reset(seed)
    ep = Episode(persona=persona.name, agent=policy.name, seed=seed,
                 true_fault_turn=fault_turn)
    user = persona.start
    standing: List[str] = []
    streak = 0

    for t in range(n_turns):
        user, shock = apply_shock(user, persona, rng)

        action = policy.act(user, t)
        faulted = fault_turn is not None and t >= fault_turn
        if faulted:
            w = max(0.0, min(1.0, fault_severity))
            action = Action(**{k: (1 - w) * getattr(action, k) + w * getattr(_DEGRADED, k)
                               for k in action.as_dict()})

        cal = calibration(user, action, standing)
        perc = perceived_empathy(user, action)
        violated = any(not satisfies(action, d) for d in standing)
        streak = streak + 1 if cal < 0.58 else 0
        nxt = step_user(user, action, persona, cal, streak, standing, rng)

        new_d = maybe_directive(user, action, persona, standing, rng)
        if new_d is not None:
            standing.append(new_d)
            delivered = new_d
            if rng.random() < corrupt_p:
                delivered = rng.choice([d for d in DIRECTIVES if d != new_d])
                ep.corrupted_turns.append(t)
            policy.observe_directive(delivered)

        ep.turns.append(Turn(index=t, user_before=user, action=action,
                             user_after=nxt, calibration=cal,
                             standing_directives=list(standing),
                             new_directive=new_d, directive_violated=violated,
                             perceived=perc, shock=shock, faulted=faulted))
        user = nxt

    return ep


def run_suite(policy: VoicePolicy, personas: List[Persona], n_episodes: int,
              n_turns: int = 40, corrupt_p: float = 0.0,
              seed0: int = 0) -> List[Episode]:
    out = []
    for i in range(n_episodes):
        persona = personas[i % len(personas)]
        out.append(run_episode(agent, persona, seed0 + i, n_turns,
                               corrupt_p=corrupt_p))
    return out

"""Plans and replanning controllers.

A *plan* is the full remaining sequence of UAV sorties (per round: UAV -> action list),
obtained by rolling a route policy out in the world the planner BELIEVES in (events seen
so far, no future events). It is executed in the TRUE world, where later events can
delay the UGV. Three controllers decide when to plan again:

  open       plan once at t = 0 on the static map, never replan (no event is ever observed)
  periodic   replan at every recharge-round boundary (as in the base paper's case study)
  event      replan only if an observed event makes the remaining plan infeasible:
             re-time the plan on the updated roads; replan when a rendezvous became
             unreachable, the predicted energy exceeds the budget, or a UAV would wait
             more than `tau_wait` minutes longer than planned ("not reachable on time").

Replanning happens at round boundaries and keeps the completed route (the rollout starts
from the true current state, only the remaining tasks are re-planned).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .env import MissionEnv, MissionResult

Policy = Callable[[MissionEnv, "object"], int]


@dataclass
class Plan:
    rounds: list                       # list of [(uav, [actions]), ...] per round
    planned_wait: dict                 # (round index in plan, uav) -> planned wait (min)
    pred_energy: float
    pred_makespan: float
    pred_failed: bool
    cursor: int = 0                    # next round to execute

    @property
    def remaining(self) -> int:
        return len(self.rounds) - self.cursor


@dataclass
class ControllerInfo:
    replans: int = 0
    forced_replans: int = 0
    belief_changes: int = 0
    triggers: list = field(default_factory=list)   # (time, reason)


def _group_rounds(actions, r0):
    rounds: dict[int, dict[int, list]] = {}
    for r, u, a in actions:
        rounds.setdefault(r, {}).setdefault(u, []).append(a)
    return [list(rounds[r].items()) for r in sorted(rounds)]


def rollout_plan(env: MissionEnv, policy, horizon: int | None = None) -> Plan:
    """Roll `policy` out from the boundary state of `env` in the believed (static) world."""
    t_check = env.round_start_time()
    world = env.with_timeline(env.timeline.observed(t_check))
    world.belief = env.belief
    world._refresh_ground_info()
    n0, r0 = len(world.actions), world.rounds
    while True:
        u = world.active_uav()
        if u is None or (horizon is not None and world.rounds >= r0 + horizon):
            break
        mask = world.mask()
        if world.done:
            break
        world.step(int(policy(world, mask)))
    res = world.result()
    rounds = _group_rounds(world.actions[n0:], r0)
    waits = {}
    for s in world.services:
        if s["round"] >= r0:
            waits[(s["round"] - r0, s["uav"])] = s["wait"]
    return Plan(rounds, waits, res.energy, res.makespan, res.failed)


def expected_positions(round_actions, n_tasks: int) -> dict:
    """UAV -> task index where its sortie in this round is planned to end (ground task)."""
    return {u: acts[-1] - n_tasks for u, acts in round_actions if acts and acts[-1] >= n_tasks}


def execute_round(env: MissionEnv, round_actions) -> None:
    """Fly one planned round (all sorties + the UGV phase) in `env`."""
    env.set_round_order([u for u, _ in round_actions])
    for u, acts in round_actions:
        if env.active_uav() != u:
            raise RuntimeError("plan execution lost sync with the environment")
        for a in acts:
            if env.done:
                return
            env.step(int(a))
    env.settle()                           # runs the UGV service phase of this round


def retime(env: MissionEnv, plan: Plan) -> tuple[MissionResult, dict]:
    """Predict the remaining plan under the *believed* roads (static from now on)."""
    t_check = env.round_start_time()
    world = env.with_timeline(env.timeline.observed(t_check))
    r0 = world.rounds
    for k in range(plan.cursor, len(plan.rounds)):
        world.settle()
        if world.done:
            break
        execute_round(world, plan.rounds[k])
    world.settle()
    waits = {}
    for s in world.services:
        if s["round"] >= r0:
            waits[(s["round"] - r0 + plan.cursor, s["uav"])] = s["wait"]
    return world.result(), waits


def plan_is_valid(env: MissionEnv, plan: Plan, tau_wait: float) -> tuple[bool, str]:
    try:
        res, waits = retime(env, plan)
    except RuntimeError:                     # the remaining plan cannot even be flown any more
        return False, "remaining plan infeasible (battery)"
    if res.failed:
        return False, f"predicted failure: {res.reason}"
    if env.budget is not None and res.energy > env.budget + 1e-6:
        return False, "predicted energy exceeds budget"
    worst = max((waits[k] - plan.planned_wait.get(k, 0.0) for k in waits), default=0.0)
    if worst > tau_wait:
        return False, f"a UAV would wait {worst:.1f} min longer than planned"
    return True, "ok"


def run_controller(
    env: MissionEnv,
    planner: Callable[[MissionEnv, int | None], Plan],
    mode: str,
    tau_wait: float = 5.0,
) -> tuple[MissionResult, ControllerInfo]:
    """Run a whole mission in `env` (true events inside) under a replanning mode."""
    if mode not in {"open", "periodic", "event"}:
        raise ValueError("mode must be open, periodic or event")
    info = ControllerInfo()
    env.divert = mode != "open"        # replanning controllers can re-target a stranded UAV mid-round
    horizon = 1 if mode == "periodic" else None
    plan = planner(env, horizon)
    diverted = False
    while True:
        env.settle()
        if env.done:
            break
        # round boundary: queue refreshed, nothing pending, no sortie started
        if mode != "open" and env.rounds > 0:
            t = env.round_start_time()
            changed = env.observe(t)
            info.belief_changes += int(changed)
            if diverted:                                   # a UAV was re-targeted: plan no longer matches reality
                plan = planner(env, horizon)
                info.replans += 1
                info.triggers.append((t, "UAV diverted after a road event"))
            elif mode == "periodic":
                plan = planner(env, horizon)
                info.replans += 1
            elif plan.remaining == 0:
                plan = planner(env, horizon)
                info.replans += 1
                info.forced_replans += 1
            elif changed:
                ok, why = plan_is_valid(env, plan, tau_wait)
                if not ok:
                    plan = planner(env, horizon)
                    info.replans += 1
                    info.triggers.append((t, why))
        if plan.remaining == 0:
            env._fail("plan exhausted with tasks left" if mode == "open" else "planner returned an empty plan")
            break
        expected = expected_positions(plan.rounds[plan.cursor], env.K)
        execute_round(env, plan.rounds[plan.cursor])
        plan.cursor += 1
        diverted = any(env.cur[u] != g for u, g in expected.items())
    return env.result(), info

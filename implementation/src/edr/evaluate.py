"""Paired evaluation: identical test instances, budgets and road events for every policy."""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import stats

from .env import MissionEnv
from .events import RoadEvent, make_event
from .heuristics import nearest_feasible
from .model import RouteModel
from .params import Params
from .plan import Plan, _group_rounds, rollout_plan, run_controller
from .rl import NeuralPolicy, Setting, new_env, reference_runs, rollout, stack_feats
from .scenario import Scenario, generate_scenario

TEST_SEED0 = 2_000_000_000


# ------------------------------------------------------------------ models
def save_model(model: RouteModel, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": model.state_dict(), "use_budget": model.use_budget, "use_pace": model.use_pace}, path)


def load_model(path: str) -> RouteModel:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    m = RouteModel(use_budget=blob["use_budget"], use_pace=blob.get("use_pace", False))
    m.load_state_dict(blob["state"])
    return m.eval()


# ------------------------------------------------------------------ test cases
@dataclass
class Case:
    seed: int
    scn: Scenario
    t0: float            # static makespan of the base policy P0
    b0: float            # static energy of P0  (B0)
    drives: list         # (time, edge) of P0's UGV drives (where events are placed)

    def event(self, kind: str, level: str, timing: str, disconnect: bool = False) -> RoadEvent:
        return make_event(self.scn.roads, self.drives, self.t0, kind, level, timing,
                          disconnect=disconnect, rng=np.random.default_rng(self.seed))


def build_cases(s: Setting, base: RouteModel, n: int, seed0: int = TEST_SEED0) -> list[Case]:
    """n test instances. Instances on which the base policy itself fails on static roads have
    no meaningful B0 and are skipped (the next seed takes their place)."""
    cases: list[Case] = []
    k = 0
    while len(cases) < n:
        m = n - len(cases)
        scns = [generate_scenario(seed0 + k + j, s.n_aerial, s.n_ground, s.roads, s.params) for j in range(m)]
        envs = [new_env(s, scn) for scn in scns]
        rollout(base.eval(), envs, "greedy", feats=stack_feats(scns))
        for j, (scn, e) in enumerate(zip(scns, envs)):
            r = e.result()
            if not r.failed:
                cases.append(Case(seed0 + k + j, scn, r.makespan, r.energy, [(d[1], d[5]) for d in r.ugv_drives]))
        k += m
    return cases


# ------------------------------------------------------------------ planners
def greedy_planner(model: RouteModel):
    policy = NeuralPolicy(model, greedy=True)
    return lambda env, horizon: rollout_plan(env, policy, horizon)


def heuristic_planner():
    return lambda env, horizon: rollout_plan(env, nearest_feasible, horizon)


def route_filter_planner(model: RouteModel, n_samples: int = 64, seed: int = 0):
    """P1: sample routes from the base policy and keep the fastest one with E <= B
    (if none fits the budget, the lowest-energy route)."""
    torch.manual_seed(seed)
    model = model.eval()

    def plan(env: MissionEnv, horizon=None) -> Plan:
        t_check = env.round_start_time()
        base_world = env.with_timeline(env.timeline.observed(t_check))
        base_world.belief = env.belief
        base_world._refresh_ground_info()
        n0, r0 = len(base_world.actions), base_world.rounds
        worlds = [base_world.clone() for _ in range(n_samples)]
        rollout(model, worlds[1:], "sample")
        rollout(model, worlds[:1], "greedy")
        res = [w.result() for w in worlds]
        budget = env.budget
        ok = [i for i, r in enumerate(res) if not r.failed and (budget is None or r.energy <= budget + 1e-6)]
        if ok:
            best = min(ok, key=lambda i: res[i].makespan)
        else:
            good = [i for i, r in enumerate(res) if not r.failed] or list(range(len(res)))
            best = min(good, key=lambda i: res[i].energy)
        w, r = worlds[best], res[best]
        waits = {(sv["round"] - r0, sv["uav"]): sv["wait"] for sv in w.services if sv["round"] >= r0}
        return Plan(_group_rounds(w.actions[n0:], r0), waits, r.energy, r.makespan, r.failed)

    return plan


# ------------------------------------------------------------------ one evaluation
@dataclass
class Spec:
    name: str
    kind: str = "neural"                 # neural | heuristic | route_filter
    model_path: str | None = None
    mode: str = "open"                   # open | periodic | event
    road_aware: bool = False
    assign_metric: str = "euclid"
    use_budget: bool = False             # set env.budget (B = beta*B0) for this policy
    n_samples: int = 64
    tau_wait: float = 5.0
    jit: bool = False                    # just-in-time departure (see env.jit)


_MODELS: dict[str, RouteModel] = {}


def _model(path: str) -> RouteModel:
    if path not in _MODELS:
        _MODELS[path] = load_model(path)
    return _MODELS[path]


def run_one(s: Setting, case: Case, spec: Spec, beta: float | None, event: RoadEvent | None) -> dict:
    env = new_env(s, case.scn, events=(event,) if event else (), road_aware=spec.road_aware,
                  assign_metric=spec.assign_metric)
    budget = beta * case.b0 if beta is not None else None
    # Only budget-aware policies see B (decoder input b_t and the replanning trigger); the
    # budget-blind ones are merely scored against it.
    env.jit = spec.jit
    env.budget = budget if (spec.use_budget or spec.kind == "route_filter") else None
    if spec.kind == "heuristic":
        planner = heuristic_planner()
    elif spec.kind == "route_filter":
        planner = route_filter_planner(_model(spec.model_path), spec.n_samples, seed=case.seed)
    else:
        planner = greedy_planner(_model(spec.model_path))
    res, info = run_controller(env, planner, spec.mode, tau_wait=spec.tau_wait)
    return dict(
        seed=case.seed, T=res.makespan, E=res.energy, E_fly=res.energy_fly, E_wait=res.energy_wait,
        wait=res.wait_time, failed=int(res.failed), reason=res.reason or "", cost=res.cost,
        rounds=res.rounds, sorties=res.sorties, replans=info.replans, forced=info.forced_replans,
        B=budget if budget is not None else float("nan"), B0=case.b0, T0=case.t0,
        over=int(budget is not None and res.energy > budget + 1e-6),
        overshoot=max(0.0, res.energy - budget) / case.b0 if budget is not None else 0.0,
        event_t=event.t if event else float("nan"),
    )


def _worker(args):
    torch.set_num_threads(1)
    s, case, spec, beta, event = args
    return run_one(s, case, spec, beta, event)


def evaluate(s: Setting, cases: list[Case], spec: Spec, beta: float | None = None,
             events: list | None = None, workers: int = 1) -> list[dict]:
    events = events if events is not None else [None] * len(cases)
    jobs = [(s, c, spec, beta, e) for c, e in zip(cases, events)]
    if workers <= 1:
        torch.set_num_threads(1)
        return [run_one(*j) for j in jobs]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_worker, jobs, chunksize=max(1, len(jobs) // (workers * 4)))


# ------------------------------------------------------------------ statistics
def ci95(x) -> float:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summarize(rows: list[dict]) -> dict:
    g = lambda k: np.array([r[k] for r in rows], float)
    out = {"n": len(rows)}
    for k in ("T", "E", "E_fly", "E_wait", "wait", "replans", "sorties", "overshoot"):
        out[k] = float(g(k).mean()); out[k + "_sd"] = float(g(k).std(ddof=1)) if len(rows) > 1 else 0.0
        out[k + "_ci"] = ci95(g(k))
    out["fail_rate"] = float(g("failed").mean())
    out["over_rate"] = float(g("over").mean())
    return out


def paired(rows_a: list[dict], rows_b: list[dict], key: str) -> dict:
    """Mean paired difference a-b with 95% CI over matched instances."""
    d = np.array([a[key] - b[key] for a, b in zip(rows_a, rows_b)], float)
    return {"mean": float(d.mean()), "ci": ci95(d), "wins_a": float((d < 0).mean()), "n": len(d)}

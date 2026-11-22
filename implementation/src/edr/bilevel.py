"""Bilevel 'UGV first, UAV second' baseline (base paper, Sec. III), re-implemented with OR-Tools.

Outer level  : minimum set cover (CP-SAT) picks the refuel stops among the ground points so that every aerial task is
               within half a battery range of a stop; the stops are ordered along the UGV route (open TSP on road distance).
Inner level  : one sub-problem per stop: every UAV flies a sortie from where it is, visits the tasks allocated to that
               stop (nearest stop) and lands at the stop. Solved as a capacity-constrained routing problem (OR-Tools) with
               the objective of minimum total UAV flight distance and a metaheuristic: GLS, Tabu Search or Simulated
               Annealing. Tasks that do not fit in one sortie per UAV are re-solved as an extra round at the same stop.
The resulting sorties are executed in the SAME simulator as the learned policies (greedy UGV assignment, waiting,
energy), so times and energies are directly comparable.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

from .env import MissionEnv, MissionResult
from .plan import execute_round

_LS = routing_enums_pb2.LocalSearchMetaheuristic
META = {"GLS": _LS.GUIDED_LOCAL_SEARCH, "TS": _LS.TABU_SEARCH, "SA": _LS.SIMULATED_ANNEALING}


def min_set_cover(env: MissionEnv) -> list[int]:
    """Ground tasks to use as refuel stops (always includes the depot)."""
    G, K = env.G, env.K
    radius = 0.5 * env.p.max_range_km
    d = env.task_dist[G:, :G]                              # aerial x ground
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{j}") for j in range(G)]
    m.Add(x[env.scn.depot_task] == 1)
    for i in range(K - G):
        cover = [x[j] for j in range(G) if d[i, j] <= radius + 1e-9]
        if not cover:
            raise ValueError("an aerial task is out of reach of every ground point")
        m.Add(sum(cover) >= 1)
    m.Minimize(sum(x))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.Solve(m)
    return [j for j in range(G) if solver.Value(x[j])]


def order_stops(env: MissionEnv, stops: list[int]) -> list[int]:
    """Open path from the depot through every stop, minimum road distance (exact for <= 8 stops)."""
    depot = env.scn.depot_task
    others = [g for g in stops if g != depot]
    D = env.roads.dist_matrix()
    node = lambda g: int(env.scn.node[g])
    path_len = lambda order: sum(D[node(a), node(b)] for a, b in zip(order[:-1], order[1:]))
    if len(others) <= 8:
        best = min(itertools.permutations(others), key=lambda p: path_len([depot, *p]))
        return [depot, *best]
    order = [depot]                                        # nearest neighbour, then 2-opt
    rest = set(others)
    while rest:
        nxt = min(rest, key=lambda g: D[node(order[-1]), node(g)])
        order.append(nxt); rest.remove(nxt)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if path_len(cand) < path_len(order) - 1e-9:
                    order, improved = cand, True
    return order


def solve_sorties(env: MissionEnv, starts: list[int], dest: int, tasks: list[int], meta: str, ms: int):
    """One round of sorties: UAV v starts at task-point starts[v], visits some of `tasks`, ends at ground task `dest`.
    Returns (routes, dropped): routes[v] = ordered list of task indices (possibly empty)."""
    M = len(starts)
    xy = env.scn.xy
    pts = [xy[s] for s in starts] + [xy[dest]] + [xy[t] for t in tasks]      # nodes: starts, destination, tasks
    n = len(pts)
    dist = (np.linalg.norm(np.array(pts)[:, None] - np.array(pts)[None], axis=-1) * 1000).astype(int)
    mgr = pywrapcp.RoutingIndexManager(n, M, list(range(M)), [M] * M)
    rt = pywrapcp.RoutingModel(mgr)
    cb = rt.RegisterTransitCallback(lambda i, j: int(dist[mgr.IndexToNode(i), mgr.IndexToNode(j)]))
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    rt.AddDimension(cb, 0, int(1000 * (env.p.max_range_km - 0.02)), True, "dist")
    # (vehicles are 'not used when empty' by default: an idle UAV stays where it is at no cost)
    for k in range(len(tasks)):
        rt.AddDisjunction([mgr.NodeToIndex(M + 1 + k)], 10**7)              # may be dropped, at a huge penalty
    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    prm.local_search_metaheuristic = META[meta]
    prm.time_limit.FromMilliseconds(ms)
    sol = rt.SolveWithParameters(prm)
    if sol is None:
        return [[] for _ in range(M)], list(tasks)
    routes, served = [], set()
    for v in range(M):
        idx, r = rt.Start(v), []
        while not rt.IsEnd(idx):
            nd = mgr.IndexToNode(idx)
            if nd > M:
                r.append(tasks[nd - M - 1]); served.add(tasks[nd - M - 1])
            idx = sol.Value(rt.NextVar(idx))
        routes.append(r)
    return routes, [t for t in tasks if t not in served]


def run_bilevel(env: MissionEnv, meta: str = "GLS", ms: int = 800, max_repeat: int = 8) -> MissionResult:
    """Plan with the bilevel heuristic and execute the sorties in `env` (fresh environment)."""
    G, K = env.G, env.K
    stops = min_set_cover(env)
    order = order_stops(env, stops)
    depot = env.scn.depot_task
    stop_set = set(stops)
    owner = {}                                              # task -> nearest selected stop
    for t in range(K):
        if t == depot:
            continue
        owner[t] = min(stops, key=lambda g: env.task_dist[t, g])
    cur = [depot] * env.M
    rounds = []
    for si, g in enumerate(order):
        tasks = [t for t, o in owner.items() if o == g and t not in stop_set]
        need_visit_stop = g != depot
        repeat = 0
        while tasks or need_visit_stop:
            if not tasks:                                   # stop with no allocated task: just land there once
                u = min(range(env.M), key=lambda v: env.task_dist[cur[v], g])
                rounds.append([(u, [K + g])]); cur[u] = g; need_visit_stop = False
                break
            routes, dropped = solve_sorties(env, cur, g, tasks, meta, ms)
            rnd = []
            for v, r in enumerate(routes):
                if r:
                    rnd.append((v, [*r, K + g])); cur[v] = g
            if not rnd:
                break                                       # nothing could be served: leave to the failure check
            rounds.append(rnd)
            need_visit_stop = False
            tasks = dropped
            repeat += 1
            if repeat > max_repeat:
                break
    for rnd in rounds:
        env.settle()
        if env.done:
            break
        try:
            execute_round(env, rnd)
        except RuntimeError:
            env._fail("bilevel plan infeasible")
            break
    env.settle()
    if not env.done and not env.visited.all():
        env._fail("bilevel plan left tasks unvisited")
    return env.result()

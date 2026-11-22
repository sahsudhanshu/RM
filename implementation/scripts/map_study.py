#!/usr/bin/env python
"""Which road maps make the problem informative? Rule-based policy (no training), 2U-1G by default.

For each map: static mission time/energy/waiting; how much the 18 road events hurt an open-loop
mission; how much road layer + event-triggered replanning recovers; replans used. A map is
'informative' when events matter AND replanning has something to fix (detours exist)."""
import argparse, csv, itertools, multiprocessing as mp, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from edr.evaluate import Case, heuristic_planner
from edr.events import LEVELS, TIMINGS
from edr.heuristics import nearest_feasible, run_policy
from edr.plan import run_controller
from edr.rl import Setting, new_env
from edr.roads import ROAD_BUILDERS
from edr.scenario import generate_scenario

KINDS = ("block", "slow")


def one(args):
    roads, seed, uav, ugv = args
    s = Setting(15, 5, uav, ugv, roads)
    scn = generate_scenario(seed, 15, 5, roads)
    ref = run_policy(new_env(s, scn), nearest_feasible)
    if ref.failed:
        return None
    case = Case(seed, scn, ref.makespan, ref.energy, [(d[1], d[5]) for d in ref.ugv_drives])
    rows = []
    for kind, level, timing in itertools.product(KINDS, LEVELS, TIMINGS):
        ev = case.event(kind, level, timing)
        out = {}
        for mode, aware in (("open", False), ("event", True)):
            res, info = run_controller(new_env(s, scn, events=(ev,), road_aware=aware), heuristic_planner(), mode)
            out[mode] = (res, info)
        rows.append(dict(roads=roads, seed=seed, kind=kind, level=level, timing=timing, T0=ref.makespan, E0=ref.energy,
                         wait0=ref.wait_time, To=out["open"][0].makespan, Eo=out["open"][0].energy, Te=out["event"][0].makespan,
                         Ee=out["event"][0].energy, replans=out["event"][1].replans,
                         failed_o=int(out["open"][0].failed), failed_e=int(out["event"][0].failed)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uav", type=int, default=2); ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--maps", nargs="*", default=list(ROAD_BUILDERS))
    a = ap.parse_args()
    allrows = []
    print(f"{'map':8s} {'bridges':>8s} {'T0':>6s} {'E0':>6s} {'wait/E0':>8s} | {'open dT%':>9s} {'open dE%':>9s} | {'repl dT':>8s} {'repl dE%':>9s} {'replans':>7s} | {'fail open':>9s} {'fail repl':>9s}")
    for m in a.maps:
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(one, [(m, 2_000_000_000 + k, a.uav, a.ugv) for k in range(a.n)])
        rows = [r for x in res if x for r in x]
        allrows += rows
        g = lambda k: np.array([r[k] for r in rows], float)
        rm = ROAD_BUILDERS[m]()
        br = np.mean([rm.is_bridge(i) for i in range(len(rm.edges))])
        print(f"{m:8s} {100*br:7.0f}% {g('T0').mean():6.1f} {g('E0').mean():6.0f} {100*(g('wait0')*13.776/g('E0')).mean():7.0f}% | "
              f"{100*(g('To')/g('T0')-1).mean():+9.1f} {100*(g('Eo')/g('E0')-1).mean():+9.1f} | "
              f"{(g('Te')-g('To')).mean():+8.1f} {100*(g('Ee')/g('Eo')-1).mean():+9.1f} {g('replans').mean():7.2f} | "
              f"{100*g('failed_o').mean():8.1f}% {100*g('failed_e').mean():8.1f}%", flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "map_study.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    print("wrote", out)

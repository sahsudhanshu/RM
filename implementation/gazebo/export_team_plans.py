#!/usr/bin/env python
"""Export multi-UAV / multi-UGV demo plans (e.g. 4 UAVs + 2 UGVs) for the Gazebo demo: run the trained team policy on the ring-road map with
a blocked-road event (open-loop vs event-triggered replanning), pick the instance where replanning helps most and both UGVs and all UAVs are
busy, and write demo/plans/team-<team>-{event,open}.json (metres / seconds, same schema as export_plans.py).

  .venv/bin/python gazebo/export_team_plans.py            # 4U-2G and 2U-2G
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, torch
from edr.evaluate import build_cases, greedy_planner, load_model
from edr.plan import run_controller
from edr.rl import Setting, new_env

torch.set_num_threads(4)
HOVER, FUEL = 13.776, 287.7          # kJ per minute of hovering, battery kJ
TEAMS = [("4u2g", 4, 2, "results_kaggle/loop-4u2g/models/U15G5_4U2G/p0.pt"), ("2u2g", 2, 2, "results_kaggle/loop-2u2g/models/U15G5_2U2G/p0.pt")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def pack(res, info, xy):
    legs = [[u, t0, t1, p0[0], p0[1], p1[0], p1[1], 1 if kind == "land" else 0, e] for (u, t0, t1, p0, p1, kind, e, rnd) in res.legs]
    return dict(T=res.makespan, E=res.energy, wait=res.wait_time, replans=info.replans, failed=bool(res.failed), legs=legs,
                services=[[sv["uav"], sv["ugv"], sv["touchdown"], sv["start"], sv["end"], sv["node"]] for sv in res.services],
                drives=[[d[0], d[1], d[2], d[3], d[4], d[5], d[6]] for d in res.ugv_drives])


def convert(case_id, title, mode, run, scn, ev, n_uav, n_ugv):
    xy = [[float(x) * 1000, float(y) * 1000] for x, y in scn.roads.xy]
    depot = scn.xy[scn.depot_task]; depot = [float(depot[0]) * 1000, float(depot[1]) * 1000]
    ugvs = []
    for g in range(n_ugv):
        dr = [d for d in run["drives"] if d[0] == g]
        ugvs.append(dict(id=g, start=xy[dr[0][3]] if dr else depot, drives=[dict(t0=d[1] * 60, t1=d[2] * 60, a=int(d[3]), b=int(d[4]), kappa=d[6]) for d in dr]))
    uavs = []
    for u in range(n_uav):
        lg = [l for l in run["legs"] if l[0] == u]
        if not lg: uavs.append(dict(id=u, start=depot, legs=[])); continue
        uavs.append(dict(id=u, start=[lg[0][3] * 1000, lg[0][4] * 1000], legs=[dict(t0=l[1] * 60, t1=l[2] * 60, p0=[l[3] * 1000, l[4] * 1000], p1=[l[5] * 1000, l[6] * 1000], land=l[7], E=l[8]) for l in lg]))
    services = [dict(uav=s[0], ugv=s[1], touchdown=s[2] * 60, start=s[3] * 60, end=s[4] * 60, node=int(s[5])) for s in run["services"]]
    return dict(id=f"{case_id}__{mode}", case=case_id, mode=mode, title=title, map="loop", T=run["T"] * 60, E=run["E"], wait=run["wait"] * 60, failed=run["failed"], replans=run["replans"],
                E_legs=sum(l[8] for l in run["legs"]), hover_kj_per_s=HOVER / 60, fuel=FUEL, roads=dict(xy=xy, edges=[list(map(int, e)) for e in scn.roads.edges]),
                tasks=dict(xy=[[float(x) * 1000, float(y) * 1000] for x, y in scn.xy], n_ground=int(scn.n_ground), depot=depot), ugvs=ugvs, uavs=[u for u in uavs if u["legs"]] or uavs,
                services=services, event=dict(t=ev.t * 60, kind=ev.kind, edges=list(ev.edges), kappa=ev.kappa))


for name, m, n, rel in TEAMS:
    s = Setting(15, 5, m, n, "loop"); base = load_model(os.path.join(ROOT, rel)); cases = build_cases(s, base, N); cand = []
    for c in cases:
        ev = c.event("block", "medium", "middle")
        if not ev.edges or len(ev.edges) > 2: continue
        runs = []
        for aware in (False, True):
            env = new_env(s, c.scn, events=(ev,), road_aware=aware); runs.append(run_controller(env, greedy_planner(base), "event" if aware else "open"))
        (ro, io), (re, ie) = runs
        if ro.failed or re.failed: continue
        busy = len({d[0] for d in re.ugv_drives}) == n and len({l[0] for l in re.legs}) == m
        cand.append((ro.makespan - re.makespan, c, ev, runs, busy))
    cand.sort(key=lambda x: (-int(x[4]), -x[0]))
    print(f"{name}: {len(cand)} candidates; top: " + ", ".join(f"seed {x[1].seed} gain {x[0]:.0f} min busy={x[4]}" for x in cand[:4]), flush=True)
    gain, c, ev, runs, busy = cand[0]; (ro, io), (re, ie) = runs
    cid = f"team-{name}"; title = f"Ring-road map · {m} UAVs + {n} UGVs · tree blocks the road · replanning saves {gain:.0f} min"
    for mode, (res, info) in (("open", (ro, io)), ("event", (re, ie))):
        p = convert(cid, title, mode, pack(res, info, None), c.scn, ev, m, n)
        json.dump(p, open(os.path.join(HERE, "demo", "plans", f"{cid}-{mode}.json"), "w"))
    print(f"  wrote team-{name}: open T={ro.makespan:.0f} min E={ro.energy:.0f} | event T={re.makespan:.0f} min E={re.energy:.0f} replans={ie.replans} | event at {ev.t:.0f} min on edges {list(ev.edges)}", flush=True)

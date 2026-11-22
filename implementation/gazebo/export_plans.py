#!/usr/bin/env python
"""Convert the planned missions in viewer/missions.json to metres/seconds JSON plans for the Gazebo replay."""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "..", "viewer", "missions.json")))
HOVER = data["hover_kj_per_min"]
ids = sys.argv[1:]  # case ids to export (default: all)


def convert(case, mode):
    run = case["open" if mode == "open" else "event_run"]
    xy = [[x * 1000, y * 1000] for x, y in case["roads"]["xy"]]
    legs, drives = run["legs"], run["drives"]
    uavs = sorted({l[0] for l in legs}); ugvs = sorted({d[0] for d in drives} | {s[1] for s in run["services"]})
    depot = case["tasks"]["xy"][case["tasks"]["depot"]]; depot = [depot[0] * 1000, depot[1] * 1000]
    plan_ugvs = []
    for g in ugvs:
        dr = [d for d in drives if d[0] == g]
        start = xy[dr[0][3]] if dr else depot
        plan_ugvs.append(dict(id=g, start=start, drives=[dict(t0=d[1] * 60, t1=d[2] * 60, a=d[3], b=d[4], kappa=d[6]) for d in dr]))
    plan_uavs = []
    for u in uavs:
        lg = [l for l in legs if l[0] == u]
        plan_uavs.append(dict(id=u, start=[lg[0][3] * 1000, lg[0][4] * 1000], legs=[dict(t0=l[1] * 60, t1=l[2] * 60, p0=[l[3] * 1000, l[4] * 1000], p1=[l[5] * 1000, l[6] * 1000], land=l[7], E=l[8]) for l in lg]))
    services = [dict(uav=s[0], ugv=s[1], touchdown=s[2] * 60, start=s[3] * 60, end=s[4] * 60, node=s[5]) for s in run["services"]]
    ev = case["event"]
    e_legs = sum(l[8] for l in legs)
    return dict(id=f"{case['id']}__{mode}", case=case["id"], mode=mode, title=case["title"], map=case["map"], T=run["T"] * 60, E=run["E"], wait=run["wait"] * 60,
                failed=run["failed"], replans=run["replans"], E_legs=e_legs, hover_kj_per_s=HOVER / 60, fuel=data["fuel"],
                roads=dict(xy=xy, edges=case["roads"]["edges"]), tasks=dict(xy=[[x * 1000, y * 1000] for x, y in case["tasks"]["xy"]], n_ground=case["tasks"]["n_ground"], depot=depot),
                ugvs=plan_ugvs, uavs=plan_uavs, services=services, event=dict(t=ev["t"] * 60, kind=ev["kind"], edges=ev["edges"], kappa=ev["kappa"]))


for c in data["cases"]:
    if ids and c["id"] not in ids: continue
    for mode in ("open", "event"):
        p = convert(c, mode)
        json.dump(p, open(os.path.join(HERE, "plans", p["id"] + ".json"), "w"))
        print(f"{p['id']:60s} T={p['T']/60:7.1f} min  E={p['E']:7.1f} kJ  sum legs={p['E_legs']:7.1f}  wait={p['wait']/60:6.1f} min  failed={p['failed']}")

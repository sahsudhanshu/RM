#!/usr/bin/env python
"""Export demo missions (open-loop vs event-triggered, same instance + road event) as JSON for the 3D viewer."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from edr.evaluate import build_cases, greedy_planner, load_model
from edr.plan import run_controller
from edr.rl import Setting, new_env

ROOT = os.path.join(os.path.dirname(__file__), "..")
torch.set_num_threads(2)
r3 = lambda x: round(float(x), 3)


def pack(res, info, scn):
    xy = scn.xy
    legs = []
    for (u, t0, t1, p0, p1, kind, e, rnd) in res.legs:
        task = int(np.argmin(np.linalg.norm(xy - np.array(p1), axis=1)))
        legs.append([u, r3(t0), r3(t1), r3(p0[0]), r3(p0[1]), r3(p1[0]), r3(p1[1]), 1 if kind == "land" else 0, r3(e), rnd, task])
    return dict(
        T=r3(res.makespan), E=r3(res.energy), wait=r3(res.wait_time), replans=info.replans, failed=bool(res.failed),
        reason=res.reason or "",
        legs=legs,
        services=[[sv["uav"], sv["ugv"], r3(sv["touchdown"]), r3(sv["start"]), r3(sv["end"]), sv["node"]] for sv in res.services],
        drives=[[d[0], r3(d[1]), r3(d[2]), d[3], d[4], d[5], d[6]] for d in res.ugv_drives],
    )


MAPS = [  # (road map, model dir, label, event settings)
    ("loop", "U15G5_2U1G", "Ring-road map", [("slow", "high", "early"), ("block", "medium", "middle")]),
    ("border", "U15G5_2U1G", "Border surveillance", [("block", "medium", "middle"), ("slow", "high", "early")]),
    ("paper", "U15G5_2U1G_paper", "Base paper's map", [("slow", "high", "early")]),
]
N = 40
out = []


def add_case(roads, label, tag, kind, level, timing, c, ev, runs, rank_note=""):
    rm = c.scn.roads
    extra = {}
    if rm.band is not None:
        (x0, y0), (x1, y1), half = rm.band
        extra = dict(strip=[[r3(x0), r3(y0)], [r3(x1), r3(y1)], half], border=[[r3(x0), r3(y0 + 1.8)], [r3(x1), r3(y1 + 1.8)]])
    out.append(dict(
        id=f"{roads}-{kind}-{level}-{timing}-{tag}", tag=tag, note=rank_note,
        title=f"{label} · {kind} edges, {level}, {timing} · {tag}", seed=int(c.seed), map=roads,
        roads=dict(xy=[[r3(a_), r3(b_)] for a_, b_ in rm.xy], edges=[list(map(int, e)) for e in rm.edges]),
        tasks=dict(xy=[[r3(a_), r3(b_)] for a_, b_ in c.scn.xy], n_ground=c.scn.n_ground, node=[int(n) for n in c.scn.node], depot=int(c.scn.depot_task)),
        event=dict(t=r3(ev.t), kind=ev.kind, edges=list(ev.edges), kappa=ev.kappa),
        budget=r3(0.9 * c.b0), static=dict(T=r3(c.t0), E=r3(c.b0)),
        open=pack(runs[0][0], runs[0][1], c.scn), event_run=pack(runs[1][0], runs[1][1], c.scn), **extra,
    ))
    o, e = out[-1]["open"], out[-1]["event_run"]
    print(f"{out[-1]['id']:52s} seed {c.seed} open T={o['T']:.0f} E={o['E']:.0f} fail={o['failed']} | event T={e['T']:.0f} E={e['E']:.0f} fail={e['failed']} replans={e['replans']}")


def run_pair(s, c, base, ev):
    runs = []
    for mode, aware in (("open", False), ("event", True)):
        env = new_env(s, c.scn, events=(ev,), road_aware=aware)
        runs.append(run_controller(env, greedy_planner(base), mode))
    return runs


for roads, mdir, label, combos in MAPS:
    s = Setting(15, 5, 2, 1, roads)
    base = load_model(os.path.join(ROOT, "results", "models", mdir, "p0.pt"))
    cases = build_cases(s, base, N)
    for kind, level, timing in combos:
        pool = []
        for c in cases:
            ev = c.event(kind, level, timing)
            if not ev.edges:
                continue
            runs = run_pair(s, c, base, ev)
            pool.append((runs[0][0].cost - runs[1][0].cost, c, ev, runs))
        pool.sort(key=lambda x: x[0])
        gains = [g for g, *_ in pool]
        picks = [("worst case", pool[0]), ("typical case", pool[len(pool) // 2]), ("best case", pool[-1])]
        note = f"of {len(pool)} test instances: replanning changes mission cost by {{g:+.0f}} min here; median {np.median(gains):+.0f}, range {gains[0]:+.0f} to {gains[-1]:+.0f}"
        for tag, (g, c, ev, runs) in picks:
            add_case(roads, label, tag, kind, level, timing, c, ev, runs, note.format(g=g))
    # stress test: a bridge is cut (the map is disconnected) -> missions can fail
    stress = []
    for c in cases:
        ev = c.event("block", "low", "early", disconnect=True)
        if not c.scn.roads.is_bridge(ev.edges[0]):
            continue
        runs = run_pair(s, c, base, ev)
        stress.append((runs[0][0].failed, runs[1][0].failed, c, ev, runs))
    stress.sort(key=lambda x: (-int(x[0]), int(x[1])))       # a case where open-loop fails, preferably replanning survives
    if stress:
        f_open, f_evt, c, ev, runs = stress[0]
        add_case(roads, label, "cut road (stress test)", "block", "bridge", "early", c, ev, runs,
                 f"a bridge is cut so the map is disconnected; across {len(stress)} such cases open-loop fails in {sum(x[0] for x in stress)} and replanning in {sum(x[1] for x in stress)}")
meta = dict(fuel=287.7, recharge=15.0, hover_kj_per_min=13.776, cases=out)
path = os.path.join(ROOT, "viewer", "missions.json")
json.dump(meta, open(path, "w"), separators=(",", ":"))
print("wrote", path, os.path.getsize(path) // 1024, "KB")

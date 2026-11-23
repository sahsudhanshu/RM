#!/usr/bin/env python
"""Add multi-vehicle cases (4 UAVs + 2 UGVs, 2 UAVs + 2 UGVs on the ring road) to viewer/missions.json, then rebuild the viewer.
Idempotent: cases that already carry a `team` field are replaced. Same instance and road event under open-loop vs event-triggered replanning.

  .venv/bin/python scripts/export_viewer_multi.py && .venv/bin/python scripts/build_viewer.py
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from edr.evaluate import build_cases, greedy_planner, load_model
from edr.plan import run_controller
from edr.rl import Setting, new_env

ROOT = os.path.join(os.path.dirname(__file__), ".."); torch.set_num_threads(4)
path = os.path.join(ROOT, "viewer", "missions.json"); data = json.load(open(path))
data["cases"] = [c for c in data["cases"] if not c.get("team")]
r3 = lambda x: round(float(x), 3)
TEAMS = [("4 UAVs + 2 UGVs", 4, 2, "results_kaggle/loop-4u2g/models/U15G5_4U2G/p0.pt", "4U2G"), ("2 UAVs + 2 UGVs", 2, 2, "results_kaggle/loop-2u2g/models/U15G5_2U2G/p0.pt", "2U2G")]
EVENTS = [("slow", "high", "early"), ("block", "medium", "middle")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def pack(res, info, scn):
    legs = []
    for (u, t0, t1, p0, p1, kind, e, rnd) in res.legs:
        task = int(np.argmin(np.linalg.norm(scn.xy - np.array(p1), axis=1)))
        legs.append([u, r3(t0), r3(t1), r3(p0[0]), r3(p0[1]), r3(p1[0]), r3(p1[1]), 1 if kind == "land" else 0, r3(e), rnd, task])
    return dict(T=r3(res.makespan), E=r3(res.energy), wait=r3(res.wait_time), replans=info.replans, failed=bool(res.failed), reason=res.reason or "", legs=legs,
                services=[[sv["uav"], sv["ugv"], r3(sv["touchdown"]), r3(sv["start"]), r3(sv["end"]), sv["node"]] for sv in res.services],
                drives=[[d[0], r3(d[1]), r3(d[2]), d[3], d[4], d[5], d[6]] for d in res.ugv_drives])


def run_pair(s, c, base, ev):
    out = []
    for mode, aware in (("open", False), ("event", True)):
        env = new_env(s, c.scn, events=(ev,), road_aware=aware); out.append(run_controller(env, greedy_planner(base), mode))
    return out


def add_case(team, tag, kind, level, timing, c, ev, runs, note, nu, ng, short):
    rm = c.scn.roads
    data["cases"].append(dict(
        id=f"loop-{short}-{kind}-{level}-{timing}-{tag}", tag=tag, note=note, team=team, n_uav=nu, n_ugv=ng,
        title=f"Ring-road map · {kind} edges, {level}, {timing} · {tag}", seed=int(c.seed), map="loop",
        roads=dict(xy=[[r3(a), r3(b)] for a, b in rm.xy], edges=[list(map(int, e)) for e in rm.edges]),
        tasks=dict(xy=[[r3(a), r3(b)] for a, b in c.scn.xy], n_ground=c.scn.n_ground, node=[int(n) for n in c.scn.node], depot=int(c.scn.depot_task)),
        event=dict(t=r3(ev.t), kind=ev.kind, edges=list(ev.edges), kappa=ev.kappa), budget=r3(0.9 * c.b0), static=dict(T=r3(c.t0), E=r3(c.b0)),
        open=pack(runs[0][0], runs[0][1], c.scn), event_run=pack(runs[1][0], runs[1][1], c.scn)))
    o, e = data["cases"][-1]["open"], data["cases"][-1]["event_run"]
    print(f"{team:16s} {kind}-{level}-{timing} {tag:24s} open T={o['T']:.0f} fail={o['failed']} | event T={e['T']:.0f} fail={e['failed']} replans={e['replans']}", flush=True)


for team, nu, ng, mp, short in TEAMS:
    s = Setting(15, 5, nu, ng, "loop"); base = load_model(os.path.join(ROOT, mp)); cases = build_cases(s, base, N)
    for kind, level, timing in EVENTS:
        pool = []
        for c in cases:
            ev = c.event(kind, level, timing)
            if not ev.edges: continue
            runs = run_pair(s, c, base, ev)
            if runs[0][0].failed or runs[1][0].failed: continue
            if len({d[0] for d in runs[1][0].ugv_drives}) < ng: continue                 # every truck is used
            pool.append((runs[0][0].cost - runs[1][0].cost, c, ev, runs))
        pool.sort(key=lambda x: x[0]); gains = [g for g, *_ in pool]
        note = f"of {len(pool)} test instances ({team}): replanning changes mission cost by {{g:+.0f}} min here; median {np.median(gains):+.0f}, range {gains[0]:+.0f} to {gains[-1]:+.0f}"
        for tag, (g, c, ev, runs) in (("worst case", pool[0]), ("typical case", pool[len(pool) // 2]), ("best case", pool[-1])):
            add_case(team, tag, kind, level, timing, c, ev, runs, note.format(g=g), nu, ng, short)
    stress = []
    for c in cases:
        ev = c.event("block", "low", "early", disconnect=True)
        if not c.scn.roads.is_bridge(ev.edges[0]): continue
        runs = run_pair(s, c, base, ev); stress.append((runs[0][0].failed, runs[1][0].failed, c, ev, runs))
    stress.sort(key=lambda x: (-int(x[0]), int(x[1])))
    if stress:
        fo, fe, c, ev, runs = stress[0]
        add_case(team, "cut road (stress test)", "block", "bridge", "early", c, ev, runs, f"a bridge is cut so the map is disconnected ({team}); across {len(stress)} such cases open-loop fails in {sum(x[0] for x in stress)} and replanning in {sum(x[1] for x in stress)}", nu, ng, short)
json.dump(data, open(path, "w"), separators=(",", ":")); print("wrote", path, os.path.getsize(path) // 1024, "KB;", len(data["cases"]), "cases")

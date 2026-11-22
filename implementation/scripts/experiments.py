#!/usr/bin/env python
"""Run the experiment stages and write raw per-instance rows to results/<fleet>/<stage>.csv.

  python scripts/experiments.py A  --uav 2 --ugv 1 --n 100     # static roads, budgets (RQ1, RQ2)
  python scripts/experiments.py B  --uav 2 --ugv 1 --n 100     # blocked / slow roads (RQ3, RQ4)
  python scripts/experiments.py S  --uav 2 --ugv 1 --n 100     # stress test: disconnecting block
  python scripts/experiments.py V  ...                         # M1 validation vs base paper trends

Every policy sees the same test instances, budgets (B = beta * B0) and road events.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from edr.events import LEVELS, TIMINGS
from edr.evaluate import Spec, build_cases, evaluate, load_model
from edr.heuristics import nearest_feasible
from edr.params import Params
from edr.rl import Setting

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
BETAS = (0.7, 0.8, 0.9, 1.0)
KINDS = ("block", "slow")


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def tag(rows, **kw):
    for r in rows:
        r.update(kw)
    return rows


def mp(models, name):
    return os.path.join(models, f"{name}.pt")


def have(models, name):
    return os.path.exists(mp(models, name))


def stage_a(s, cases, models, workers):
    rows = []
    jobs = [("P0", Spec("P0", model_path=mp(models, "p0")), BETAS),
            ("P0+ (control)", Spec("P0+", model_path=mp(models, "p0c")), BETAS) if have(models, "p0c") else None,
            ("P1 route filter", Spec("P1", kind="route_filter", model_path=mp(models, "p0"), n_samples=64), BETAS),
            ("P2 fixed penalty", Spec("P2", model_path=mp(models, "p2")), BETAS) if have(models, "p2") else None,
            ("P4 budget-conditioned", Spec("P4", model_path=mp(models, "p4"), use_budget=True), BETAS) if have(models, "p4") else None,
            ("P4 no budget input", Spec("P4nb", model_path=mp(models, "p4_nb")), BETAS) if have(models, "p4_nb") else None,
            ("P4x budget + pace", Spec("P4x", model_path=mp(models, "p4x"), use_budget=True), BETAS) if have(models, "p4x") else None,
            ("Distilled search, time only", Spec("Bt", model_path=mp(models, "bon_t")), BETAS) if have(models, "bon_t") else None,
            ("Distilled search, budget-blind", Spec("Bnb", model_path=mp(models, "bon_nb")), BETAS) if have(models, "bon_nb") else None,
            ("Distilled search, budget-aware", Spec("Bp4x", model_path=mp(models, "bon_p4x"), use_budget=True), BETAS) if have(models, "bon_p4x") else None]
    for v, nm in (("p3h_b90", "P3 high-cap lambda (0.9)"), ("p3h_b100", "P3 high-cap lambda (1.0)")):
        if have(models, v):
            jobs.append((nm, Spec("P3h", model_path=mp(models, v)), (float(v[-2:]) / 100 if v.endswith("90") else 1.0,)))
    for b in (70, 80, 90):
        if have(models, f"p3_b{b}"):
            jobs.append((f"P3 Lagrangian", Spec("P3", model_path=mp(models, f"p3_b{b}")), (b / 100,)))
    for item in filter(None, jobs):
        name, spec, betas = item
        for beta in betas:
            t = time.time()
            r = evaluate(s, cases, spec, beta=beta, workers=workers)
            rows += tag(r, policy=name, beta=beta)
            print(f"A  {name:24s} beta={beta:.1f}  T {np.mean([x['T'] for x in r]):6.1f}  E {np.mean([x['E'] for x in r]):6.0f}  "
                  f"over {np.mean([x['over'] for x in r]):.2f}  fail {np.mean([x['failed'] for x in r]):.2f}  [{time.time()-t:.0f}s]", flush=True)
    return rows


def stage_b_specs(models):
    P0, P4 = mp(models, "p0"), mp(models, "p4")
    S = [("P0 open-loop (base)", Spec("P0-open", model_path=P0, mode="open")),
         ("P0 periodic + road layer", Spec("P0-per", model_path=P0, mode="periodic", road_aware=True)),
         ("P0 event + road layer", Spec("P0-evt", model_path=P0, mode="event", road_aware=True))]
    if have(models, "p0c"):
        C = mp(models, "p0c")
        S += [("P0+ (control) open-loop", Spec("P0c-open", model_path=C, mode="open")),
              ("P0+ (control) event + road layer", Spec("P0c-evt", model_path=C, mode="event", road_aware=True)),
              ("P0+ event, road-distance assignment", Spec("P0c-evt-road", model_path=C, mode="event", road_aware=True,
                                                          assign_metric="road"))]
    if have(models, "p4"):
        S += [("P4 open-loop", Spec("P4-open", model_path=P4, mode="open", use_budget=True)),
              ("P4 periodic + road layer", Spec("P4-per", model_path=P4, mode="periodic", road_aware=True, use_budget=True)),
              ("P4 event + road layer", Spec("P4-evt", model_path=P4, mode="event", road_aware=True, use_budget=True))]
    if have(models, "p4_ev"):
        S += [("P4 (road-event trained) event", Spec("P4ev-evt", model_path=mp(models, "p4_ev"), mode="event", road_aware=True, use_budget=True))]
    if have(models, "p0_ev"):
        S += [("P0 (road-event trained) event", Spec("P0ev-evt", model_path=mp(models, "p0_ev"), mode="event", road_aware=True))]
    for v, nm in (("bon_t", "Distilled search, time only"), ("bon_nb", "Distilled search, budget-blind"), ("bon_p4x", "Distilled search, budget-aware")):
        if have(models, v):
            S += [(f"{nm}, event + road layer", Spec(v, model_path=mp(models, v), mode="event", road_aware=True, use_budget=(v == "bon_p4x")))]
    if have(models, "p3_b90"):
        S += [("P3 (beta=0.9) event + road layer", Spec("P3-evt", model_path=mp(models, "p3_b90"), mode="event", road_aware=True))]
    S += [("P4 event, road-distance assignment", Spec("P4-evt-road", model_path=P4, mode="event", road_aware=True,
                                                     use_budget=True, assign_metric="road"))] if have(models, "p4") else []
    S += [("P0 event, road-distance assignment", Spec("P0-evt-road", model_path=P0, mode="event", road_aware=True,
                                                     assign_metric="road"))]
    S += [("P1 route filter, event", Spec("P1-evt", kind="route_filter", model_path=P0, mode="event", road_aware=True, n_samples=32))]
    return S


def stage_b(s, cases, models, workers, beta=0.9, small=False):
    rows = []
    rm = cases[0].scn.roads
    # On a tree (the base paper's map) every edge is a bridge: a "block" would disconnect the map, so only
    # slow events are meaningful there (disconnections are the stress test, stage S).
    kinds = KINDS if any(not rm.is_bridge(i) for i in range(len(rm.edges))) else ("slow",)
    grid = [(None, None, None)] + list(itertools.product(kinds, LEVELS, TIMINGS))
    if small:
        grid = [(None, None, None), ("slow", "high", "early"), ("block", "medium", "middle"), ("slow", "medium", "late")]
    for name, spec in stage_b_specs(models):
        t = time.time()
        for (kind, level, timing) in grid:
            events = [None if kind is None else c.event(kind, level, timing) for c in cases]
            r = evaluate(s, cases, spec, beta=beta, events=events, workers=workers)
            rows += tag(r, policy=name, beta=beta, kind=kind or "none", level=level or "none", timing=timing or "none")
        sub = [x for x in rows if x["policy"] == name and x["kind"] != "none"]
        print(f"B  {name:38s} T {np.mean([x['T'] for x in sub]):6.1f}  E {np.mean([x['E'] for x in sub]):6.0f}  "
              f"over {np.mean([x['over'] for x in sub]):.2f}  fail {np.mean([x['failed'] for x in sub]):.3f}  "
              f"replans {np.mean([x['replans'] for x in sub]):.2f}  [{time.time()-t:.0f}s]", flush=True)
    return rows


def stage_s(s, cases, models, workers, beta=0.9):
    """Stress test: a single bridge is blocked, stranding part of the map."""
    rows = []
    for name, spec in stage_b_specs(models)[:3] + [x for x in stage_b_specs(models) if x[0] in ("P4 open-loop", "P4 event + road layer")]:
        for timing in TIMINGS:
            events = [c.event("block", "low", timing, disconnect=True) for c in cases]
            keep = [i for i, (c, e) in enumerate(zip(cases, events)) if c.scn.roads.is_bridge(e.edges[0])]
            r = evaluate(s, [cases[i] for i in keep], spec, beta=beta, events=[events[i] for i in keep], workers=workers)
            rows += tag(r, policy=name, beta=beta, kind="block-disconnect", level="bridge", timing=timing)
        sub = [x for x in rows if x["policy"] == name]
        print(f"S  {name:38s} fail {np.mean([x['failed'] for x in sub]):.3f}  n={len(sub)}", flush=True)
    return rows


def stage_v(s, cases, models, workers):
    """M1: base-paper trend check (static roads, base policy)."""
    rows = []
    for name, spec in [("P0 greedy", Spec("P0", model_path=mp(models, "p0"))),
                       ("nearest-feasible rule", Spec("rule", kind="heuristic"))]:
        rows += tag(evaluate(s, cases, spec, workers=workers), policy=name)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["A", "B", "S", "V"])
    ap.add_argument("--uav", type=int, default=2)
    ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--aerial", type=int, default=15)
    ap.add_argument("--ground", type=int, default=5)
    ap.add_argument("--roads", default="loop")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0, help="training seed of the models to evaluate")
    ap.add_argument("--wait-mode", default="hover", choices=["hover", "landed"])
    ap.add_argument("--models-root", default="results", help="folder that contains models/<tag> (relative to implementation/)")
    ap.add_argument("--out-root", default="results", help="where the stage CSV goes (relative to implementation/)")
    ap.add_argument("--small", action="store_true", help="Stage B on 3 representative events only (large sizes)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    s = Setting(a.aerial, a.ground, a.uav, a.ugv, a.roads, Params(wait_mode=a.wait_mode))
    sfx = f"_seed{a.seed}" if a.seed else ""
    # models are always the ones trained on U15G5 (zero-shot generalisation to larger sizes, as in the base paper)
    train_tag = Setting(15, 5, a.uav, a.ugv, a.roads, Params(wait_mode=a.wait_mode)).tag
    models = os.path.join(ROOT, "..", a.models_root, "models", train_tag + sfx)
    base = load_model(os.path.join(models, "p0.pt"))
    cases = build_cases(s, base, a.n)
    print(f"{len(cases)} cases; B0 = {np.mean([c.b0 for c in cases]):.0f} kJ, T0 = {np.mean([c.t0 for c in cases]):.1f} min", flush=True)
    out = os.path.join(ROOT, "..", a.out_root, s.tag, f"stage{a.stage}{sfx}.csv")
    extra = ([a.beta] if a.stage in "BS" else []) + ([a.small] if a.stage == "B" else [])
    rows = {"A": stage_a, "B": stage_b, "S": stage_s, "V": stage_v}[a.stage](s, cases, models, a.workers, *extra)
    write(out, rows)

#!/usr/bin/env python
"""Summarise the Gazebo replays: planned (our simulator) vs realised (Gazebo physics) mission time, UAV energy and UGV timing."""
import csv, glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
files = [(f, "flat") for f in sorted(glob.glob(os.path.join(HERE, "results", "*.json")))] + [(f, "terrain") for f in sorted(glob.glob(os.path.join(HERE, "results_terrain", "*.json")))]
for f, ground in files:
    r = json.load(open(f))
    if r["unfinished"]: print("UNFINISHED", r["id"]); continue
    rows.append(dict(ground=ground, case=r["case"], mode=r["mode"], T_plan_min=r["T_plan_s"] / 60, T_gz_min=r["T_gz_s"] / 60, dT_min=(r["T_gz_s"] - r["T_plan_s"]) / 60,
                     E_plan_kJ=r["E_plan"], E_gz_kJ=r["E_gz"], dE_pct=100 * (r["E_gz"] / r["E_plan"] - 1), ugv_drives=r["ugv_drives"],
                     ugv_err_mean_s=r["ugv_arrival_err_mean_s"], ugv_err_max_s=r["ugv_arrival_err_max_s"], uav_snaps=r["uav_snaps"], wall_min=r["wall_s"] / 60))
with open(os.path.join(HERE, "gazebo_vs_plan.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{'ground':7s}{'case':44s} {'mode':6s} {'T plan':>7s} {'T gz':>7s} {'dT':>6s} {'E plan':>7s} {'E gz':>7s} {'dE%':>6s} {'UGV err mean/max s':>19s}")
for r in rows:
    print(f"{r['ground']:7s}{r['case']:44s} {r['mode']:6s} {r['T_plan_min']:7.1f} {r['T_gz_min']:7.1f} {r['dT_min']:6.1f} {r['E_plan_kJ']:7.0f} {r['E_gz_kJ']:7.0f} {r['dE_pct']:6.1f} {r['ugv_err_mean_s']:9.1f}/{r['ugv_err_max_s']:.0f}")
import numpy as np
for g in ("flat", "terrain"):
    rr = [r for r in rows if r["ground"] == g]
    if rr: print(f"{g}: n={len(rr)}  mean |dT| = {np.mean([abs(r['dT_min']) for r in rr]):.2f} min ({100*np.mean([abs(r['dT_min'])/r['T_plan_min'] for r in rr]):.2f} %), max |dT| = {max(abs(r['dT_min']) for r in rr):.2f} min, mean dE = {np.mean([r['dE_pct'] for r in rr]):.2f} %, mean UGV arrival error = {np.mean([r['ugv_err_mean_s'] for r in rr]):.1f} s")
print(f"\nmean |dT| = {np.mean([abs(r['dT_min']) for r in rows]):.2f} min ({100*np.mean([abs(r['dT_min'])/r['T_plan_min'] for r in rows]):.2f} %), mean dE = {np.mean([r['dE_pct'] for r in rows]):.2f} %, n = {len(rows)}")

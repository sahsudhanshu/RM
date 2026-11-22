#!/usr/bin/env python
"""Figures and the numeric report from the raw experiment CSVs (results/<fleet>/stage*.csv)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edr.evaluate import ci95
from edr.viz import AQUA, BLUE, GRID, INK, MUTED, ORANGE, RED, style_axes

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
VIOLET, GRAY = "#6250d6", "#8a8984"
POLICY_STYLE = {            # colour (fixed per entity) + marker (secondary encoding)
    "P0": (GRAY, "o"), "P0+ (control)": ("#b9b8b0", "o"), "P1 route filter": (VIOLET, "D"),
    "P2 fixed penalty": (ORANGE, "s"), "P3 Lagrangian": (AQUA, "^"),
    "P4 budget-conditioned": (BLUE, "o"), "P4 no budget input": ("#8fb6ea", "v"),
}


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def mean(rows, k):
    return float(np.mean([r[k] for r in rows])) if rows else float("nan")


def fig_budget(rows, out, tag):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    pols = list(dict.fromkeys(r["policy"] for r in rows))
    for p in pols:
        col, mk = POLICY_STYLE.get(p, (MUTED, "o"))
        pts = []
        for b in sorted({r["beta"] for r in rows if r["policy"] == p}):
            rr = sel(rows, policy=p, beta=b)
            pts.append((b, np.mean([r["E"] / r["B0"] for r in rr]), np.mean([r["T"] / r["T0"] for r in rr]),
                        np.mean([r["over"] for r in rr])))
        pts = np.array(pts)
        axs[0].plot(pts[:, 1], pts[:, 2], color=col, marker=mk, ms=7, lw=1.4, mec="white", label=p)
        axs[1].plot(pts[:, 0], pts[:, 3] * 100, color=col, marker=mk, ms=7, lw=1.4, mec="white", label=p)
    for ax in axs:
        style_axes(ax)
    axs[0].axhline(1, color=GRID, lw=1); axs[0].axvline(1, color=GRID, lw=1)
    axs[0].set_xlabel("mean energy  E / B0", color=MUTED); axs[0].set_ylabel("mean mission time  T / T0", color=MUTED)
    axs[0].set_title("Time-energy trade-off (static roads)", loc="left", fontsize=11, color=INK)
    axs[1].set_xlabel("budget level  beta  (B = beta * B0)", color=MUTED); axs[1].set_ylabel("instances over budget (%)", color=MUTED)
    axs[1].set_title("Budget violations", loc="left", fontsize=11, color=INK)
    axs[1].set_xticks([0.7, 0.8, 0.9, 1.0])
    axs[1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"Stage A - {tag}", x=.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_events(rows, out, tag, policy="P0 open-loop (base)"):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white", sharey=False)
    static = {r["seed"]: r for r in sel(rows, policy=policy, kind="none")}
    for ax, key, lab in zip(axs, ("T", "E"), ("mission time increase (%)", "UAV energy increase (%)")):
        style_axes(ax)
        xs, w = np.arange(3), 0.36
        for j, (kind, col) in enumerate((("block", RED), ("slow", "#eda100"))):
            ms, cs = [], []
            for lv in ("low", "medium", "high"):
                rr = sel(rows, policy=policy, kind=kind, level=lv)
                d = [100 * (r[key] / static[r["seed"]][key] - 1) for r in rr]
                ms.append(np.mean(d)); cs.append(ci95(d))
            ax.bar(xs + (j - .5) * w, ms, w * .92, yerr=cs, color=col, ecolor=INK, capsize=3, label=f"{kind} edges")
        ax.set_xticks(xs, ["low", "medium", "high"]); ax.set_xlabel("event level", color=MUTED); ax.set_ylabel(lab, color=MUTED)
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Stage B - effect of road events on the base policy, open loop ({tag})", x=.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_replan(rows, out, tag, pairs):
    """pairs: [(label, [policy names for open, periodic, event])]"""
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.8), facecolor="white")
    modes = [("open-loop", GRAY, "o"), ("periodic", ORANGE, "s"), ("event-triggered", BLUE, "D")]
    keys = [("T", "mission time (min)"), ("E", "UAV energy (kJ)"), ("over", "over budget (%)"), ("replans", "replans per mission")]
    for ax, (k, lab) in zip(axs, keys):
        style_axes(ax)
        for gi, (glabel, names) in enumerate(pairs):
            for mi, (name, (mlab, col, mk)) in enumerate(zip(names, modes)):
                rr = [r for r in sel(rows, policy=name) if r["kind"] != "none"]
                if not rr:
                    continue
                v = np.array([r[k] for r in rr]) * (100 if k == "over" else 1)
                ax.errorbar(gi + (mi - 1) * .22, v.mean(), yerr=ci95(v), color=col, marker=mk, ms=7, capsize=3, mec="white", ls="")
        ax.set_xticks(range(len(pairs)), [p[0] for p in pairs]); ax.set_title(lab, loc="left", fontsize=10, color=INK)
        ax.set_xlim(-.6, len(pairs) - .4)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", mec="white", ms=7, label=l) for l, c, m in modes]
    axs[0].legend(handles=handles, frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Stage B - open-loop vs periodic vs event-triggered replanning, all 18 events ({tag}); bars = 95% CI",
                 x=.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_train(models_dir, out, tag):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), facecolor="white")
    p0 = os.path.join(models_dir, "train_p0.json")
    if os.path.exists(p0):
        rows = json.load(open(p0))
        axs[0].plot([r["T"] for r in rows], color=GRAY, lw=1, alpha=.5)
        k = 10
        t = np.array([r["T"] for r in rows]); sm = np.convolve(t, np.ones(k) / k, mode="valid")
        axs[0].plot(np.arange(len(sm)) + k - 1, sm, color=BLUE, lw=2)
        axs[0].set_title("P0: sampled mission time (min)", loc="left", fontsize=10); axs[0].set_xlabel("REINFORCE step", color=MUTED)
    for name, col in (("p3_b70", AQUA), ("p3_b80", BLUE), ("p3_b90", ORANGE)):
        f = os.path.join(models_dir, f"train_{name}.json")
        if os.path.exists(f):
            rows = json.load(open(f))
            axs[1].plot([r["lam"] for r in rows], color=col, lw=2, label=f"beta = 0.{name[-2:]}")
            axs[2].plot([r["viol"] for r in rows], color=col, lw=1.4, alpha=.9)
    axs[1].set_title("P3: Lagrange multiplier lambda", loc="left", fontsize=10); axs[1].set_xlabel("step", color=MUTED)
    axs[1].legend(frameon=False, fontsize=8)
    axs[2].axhline(0, color=INK, lw=.8); axs[2].set_title("P3: batch mean (E - B)/B0", loc="left", fontsize=10); axs[2].set_xlabel("step", color=MUTED)
    for ax in axs:
        style_axes(ax)
    fig.suptitle(f"Training curves ({tag})", x=.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def table(rows, group, keys=("T", "E", "over", "failed", "replans"), fmt=None):
    out = ["| " + " | ".join(["policy"] + list(keys) + ["n"]) + " |", "|" + "---|" * (len(keys) + 2)]
    for g in dict.fromkeys(r[group] for r in rows):
        rr = [r for r in rows if r[group] == g]
        cells = []
        for k in keys:
            v = np.array([r[k] for r in rr], float)
            if k in ("over", "failed"):
                cells.append(f"{100*v.mean():.1f}%")
            elif k == "replans":
                cells.append(f"{v.mean():.2f}")
            else:
                cells.append(f"{v.mean():.1f} ± {ci95(v):.1f}")
        out.append("| " + " | ".join([str(g)] + cells + [str(len(rr))]) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="U15G5_2U1G")
    a = ap.parse_args()
    d = os.path.join(ROOT, a.tag)
    figs = os.path.join(d, "figures"); os.makedirs(figs, exist_ok=True)
    fig_train(os.path.join(ROOT, "models", a.tag), os.path.join(figs, "training.png"), a.tag)
    if os.path.exists(os.path.join(d, "stageA.csv")):
        fig_budget(read(os.path.join(d, "stageA.csv")), os.path.join(figs, "stageA_budget.png"), a.tag)
    if os.path.exists(os.path.join(d, "stageB.csv")):
        rows = read(os.path.join(d, "stageB.csv"))
        fig_events(rows, os.path.join(figs, "stageB_event_effect.png"), a.tag)
        fig_replan(rows, os.path.join(figs, "stageB_replanning.png"), a.tag, [
            ("P0 (base policy)", ["P0 open-loop (base)", "P0 periodic + road layer", "P0 event + road layer"]),
            ("P0+ (control)", ["P0+ (control) open-loop", "-", "P0+ (control) event + road layer"]),
            ("P4 (budget-conditioned)", ["P4 open-loop", "P4 periodic + road layer", "P4 event + road layer"])])
    print("figures written to", figs)

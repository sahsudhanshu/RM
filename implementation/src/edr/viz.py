"""Scenario plots and the animated mission simulation (open-loop vs replanning side by side)."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

# validated categorical slots 1-3 (blue, orange, aqua) + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d2", "#fcfcfb"
RED = "#e34948"
SLOW = "#eda100"      # slowed road (distinct from the UAV colours)
UAV_COL = [BLUE, ORANGE, AQUA, "#6250d6"]


def style_axes(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, alpha=0.5, lw=0.5)


def draw_roads(ax, scn, events=(), t=None, lw=2.2):
    rm = scn.roads
    for i, (u, v) in enumerate(rm.edges):
        ax.plot(*zip(rm.xy[u], rm.xy[v]), color="#b9b8b0", lw=lw, zorder=1, solid_capstyle="round")
    for ev in events:
        if t is not None and t < ev.t:
            continue
        for e in ev.edges:
            u, v = rm.edges[e]
            col = RED if ev.kind == "block" else SLOW
            ax.plot(*zip(rm.xy[u], rm.xy[v]), color=col, lw=lw + 4.0, zorder=2, solid_capstyle="round", alpha=.9)


def plot_mission(ax, scn, res, events=(), title="", show_legs=True):
    style_axes(ax)
    draw_roads(ax, scn, events)
    G = scn.n_ground
    drives = np.array([(d[3], d[4]) for d in res.ugv_drives], int) if res.ugv_drives else np.zeros((0, 2), int)
    for a, b in drives:
        ax.plot(*zip(scn.roads.xy[a], scn.roads.xy[b]), color=AQUA, lw=2.6, alpha=.9, zorder=3, solid_capstyle="round")
    if show_legs:
        for (u, t0, t1, p0, p1, kind, e, rnd) in res.legs:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=UAV_COL[u % 4], lw=1.1, ls="--" if kind == "visit" else "-",
                    alpha=.8, zorder=4)
    ax.scatter(scn.xy[G:, 0], scn.xy[G:, 1], facecolors="none", edgecolors=RED, s=34, lw=1.2, zorder=5)
    ax.scatter(scn.xy[:G, 0], scn.xy[:G, 1], c=BLUE, marker="s", s=44, zorder=6, edgecolors="white", lw=.8)
    d = scn.xy[scn.depot_task]
    ax.scatter([d[0]], [d[1]], marker="*", s=190, c=INK, zorder=7, edgecolors="white", lw=.8)
    ax.set_aspect("equal"); ax.set_xlim(0, 20); ax.set_ylim(0, 20)
    ax.set_xlabel("x (km)", fontsize=8, color=MUTED); ax.set_ylabel("y (km)", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


# ----------------------------------------------------------------------- animation
class Timeline:
    """Sample positions / battery / cumulative energy of one finished mission at any time t."""

    def __init__(self, scn, res, params):
        self.scn, self.res, self.p = scn, res, params
        self.legs = sorted(res.legs, key=lambda l: (l[0], l[1]))
        self.services = res.services
        self.drives = sorted(res.ugv_drives, key=lambda d: (d[0], d[1]))
        self.T = res.makespan
        depot = scn.xy[scn.depot_task]
        self.start_xy = depot
        self.n_uav = max(l[0] for l in res.legs) + 1 if res.legs else 1
        self.n_ugv = max([d[0] for d in res.ugv_drives] + [s["ugv"] for s in res.services] + [0]) + 1

    def uav(self, u, t):
        """-> (x, y, battery_fraction, state)"""
        F = self.p.fuel_kj
        pos, fuel, state = self.start_xy, F, "idle"
        for (uu, t0, t1, p0, p1, kind, e, rnd) in self.legs:
            if uu != u:
                continue
            if t < t0:
                break
            if t < t1:
                f = (t - t0) / max(t1 - t0, 1e-9)
                return (p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]), (fuel - f * e) / F, "flying")
            pos, fuel = np.array(p1), fuel - e
            for sv in self.services:
                if sv["uav"] == u and abs(sv["touchdown"] - t1) < 1e-6 and kind == "land":
                    if t < sv["start"]:
                        return pos[0], pos[1], fuel / F, "waiting"
                    if t < sv["end"]:
                        f = (t - sv["start"]) / max(sv["end"] - sv["start"], 1e-9)
                        return pos[0], pos[1], (fuel + f * (F - fuel)) / F, "charging"
                    fuel = F
        return pos[0], pos[1], fuel / F, "idle"

    def ugv(self, n, t):
        rm = self.scn.roads
        node_xy = rm.xy[self.scn.depot_node]
        for (nn, ta, tb, a, b, eid, kap) in self.drives:
            if nn != n:
                continue
            if t < ta:
                break
            if t < tb:
                f = (t - ta) / max(tb - ta, 1e-9)
                return rm.xy[a] + f * (rm.xy[b] - rm.xy[a])
            node_xy = rm.xy[b]
        return node_xy

    def energy(self, t):
        """Cumulative mission energy (kJ): flight + hover waiting up to time t."""
        e = 0.0
        for (u, t0, t1, p0, p1, kind, en, rnd) in self.legs:
            e += en * np.clip((t - t0) / max(t1 - t0, 1e-9), 0, 1)
        for sv in self.services:
            e += self.p.hover_kj_per_min * (np.clip(t, sv["touchdown"], sv["start"]) - sv["touchdown"])
        return e

    def visited_at(self, t):
        v = {self.scn.depot_task}
        for (u, t0, t1, p0, p1, kind, e, rnd) in self.legs:
            if t1 <= t:
                d = np.linalg.norm(self.scn.xy - np.array(p1), axis=1)
                v.add(int(np.argmin(d)))
        return v


def animate_pair(scn, left, right, params, events, out_path, labels=("Open-loop", "Event-triggered replanning"),
                 budget=None, fps=20, frames=240, dpi=110, gif=False):
    """Side-by-side mission animation of the SAME instance and road event under two controllers."""
    tls = [Timeline(scn, r, params) for r in (left, right)]
    T = max(t.T for t in tls)
    ts = np.linspace(0, T, frames)
    fig = plt.figure(figsize=(12, 7.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.25], hspace=.22, wspace=.12, left=.05, right=.98, top=.93, bottom=.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    axe = fig.add_subplot(gs[1, :])
    ev_t = events[0].t if events else None
    ugv_art, uav_art, task_art, txt_art, bat_art, road_art = [], [], [], [], [], []
    for ax, res, tl, lab in zip(axes, (left, right), tls, labels):
        plot_mission(ax, scn, res, (), title=lab, show_legs=False)
        road_art.append([])
        ax.scatter(scn.xy[scn.n_ground:, 0], scn.xy[scn.n_ground:, 1], facecolors="none", edgecolors=RED, s=34, lw=1.2, zorder=5)
        sc_vis = ax.scatter([], [], c=RED, s=34, zorder=6)
        task_art.append(sc_vis)
        ugv_art.append(ax.scatter([], [], marker="s", s=150, c=AQUA, edgecolors=INK, lw=1.2, zorder=8))
        uav_art.append([ax.scatter([], [], marker="^", s=110, c=UAV_COL[u % 4], edgecolors=INK, lw=1, zorder=9)
                        for u in range(tl.n_uav)])
        txt_art.append(ax.text(.02, .02, "", transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom",
                               bbox=dict(boxstyle="round,pad=.3", fc="white", ec=GRID, alpha=.9)))
        bat_art.append(ax.text(.98, .02, "", transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom", ha="right"))
    style_axes(axe)
    for res, lab, col, tl in zip((left, right), labels, (MUTED, BLUE), tls):
        curve = [tl.energy(t) for t in np.linspace(0, tl.T, 200)]
        axe.plot(np.linspace(0, tl.T, 200), curve, color=col, lw=1.2, alpha=.35)
    cursor = [axe.plot([], [], color=c, lw=2.2, label=l)[0] for c, l in zip((MUTED, BLUE), labels)]
    dots = [axe.plot([], [], "o", color=c, ms=6, mec="white")[0] for c in (MUTED, BLUE)]
    if budget:
        axe.axhline(budget, color=RED, lw=1.2, ls=(0, (4, 3)))
        axe.text(T * 0.995, budget, "energy budget B = 0.9 B0 ", color=RED, fontsize=8, va="bottom", ha="right")
    if ev_t is not None:
        axe.axvline(ev_t, color=SLOW, lw=1.5)
        axe.text(ev_t, axe.get_ylim()[1] * .02, " road event", color=MUTED, fontsize=8, va="bottom")
    axe.set_xlim(0, T); axe.set_ylim(0, max(max(tl.energy(tl.T) for tl in tls), budget or 0) * 1.08)
    axe.set_xlabel("mission time (min)", fontsize=9, color=MUTED); axe.set_ylabel("cumulative UAV energy (kJ)", fontsize=9, color=MUTED)
    axe.legend(loc="lower right", fontsize=8, frameon=False)
    clock = fig.suptitle("", fontsize=12, color=INK, x=.05, ha="left")
    for ax in axes:
        for l in ax.lines:
            pass
    legend = [Line2D([], [], marker="^", ls="", color=UAV_COL[0], mec=INK, label="UAV"),
              Line2D([], [], marker="s", ls="", color=AQUA, mec=INK, label="UGV"),
              Line2D([], [], color=RED, lw=4, label="blocked road"),
              Line2D([], [], color=SLOW, lw=4, label="slowed road")]
    axes[0].legend(handles=legend, loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    shown = {}

    def update(k):
        t = ts[k]
        clock.set_text(f"t = {t:6.1f} min")
        for i, (ax, tl, res) in enumerate(zip(axes, tls, (left, right))):
            if events and t >= events[0].t and not shown.get(i):
                for ev in events:
                    for e in ev.edges:
                        u, v = scn.roads.edges[e]
                        col = RED if ev.kind == "block" else SLOW
                        ax.plot(*zip(scn.roads.xy[u], scn.roads.xy[v]), color=col, lw=6.2, zorder=2, alpha=.9,
                                solid_capstyle="round")
                shown[i] = True
            vis = tl.visited_at(t)
            idx = [j for j in vis if j >= scn.n_ground]
            task_art[i].set_offsets(scn.xy[idx] if idx else np.empty((0, 2)))
            ugv_art[i].set_offsets([tl.ugv(0, t)])
            states = []
            for u in range(tl.n_uav):
                x, y, b, st = tl.uav(u, t)
                uav_art[i][u].set_offsets([[x, y]])
                states.append(f"UAV{u+1} {int(round(b*100)):3d}% {st}")
            txt_art[i].set_text(f"E = {tl.energy(min(t, tl.T)):6.0f} kJ")
            bat_art[i].set_text("\n".join(states))
            tt = np.linspace(0, min(t, tl.T), 80)
            cursor[i].set_data(tt, [tl.energy(x) for x in tt])
            dots[i].set_data([min(t, tl.T)], [tl.energy(min(t, tl.T))])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    if gif:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=dpi // 2)
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
    plt.close(fig)
    return out_path

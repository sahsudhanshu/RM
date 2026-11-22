"""Road events: edges become blocked or slow at a given mission time.

A policy learns about an event only when it occurs (`EventTimeline.observed`), whereas
the UGV physically experiences the true road state (`EventTimeline.state_at`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roads import NORMAL, RoadMap, RoadState

LEVELS = ("low", "medium", "high")
TIMINGS = {"early": 0.15, "middle": 0.5, "late": 0.8}   # fraction of the static mission time
BLOCK_EDGES = {"low": 1, "medium": 2, "high": 3}
SLOW_EDGES = {"low": 2, "medium": 4, "high": 6}
SLOW_KAPPA = {"low": 2.0, "medium": 3.0, "high": 4.0}


@dataclass(frozen=True)
class RoadEvent:
    t: float                    # minutes since mission start
    kind: str                   # "block" | "slow"
    edges: tuple                # affected edge ids
    kappa: float = 1.0          # travel-time multiplier for "slow"

    def __post_init__(self) -> None:
        if self.kind not in {"block", "slow"}:
            raise ValueError("kind must be 'block' or 'slow'")
        if self.kind == "slow" and self.kappa <= 1.0:
            raise ValueError("a slow event needs kappa > 1")


def state_after(events: tuple) -> RoadState:
    blocked: set[int] = set()
    kappa: dict[int, float] = {}
    for ev in events:
        if ev.kind == "block":
            blocked.update(ev.edges)
        else:
            for e in ev.edges:
                kappa[e] = max(kappa.get(e, 1.0), ev.kappa)
    return RoadState(frozenset(blocked), tuple(sorted(kappa.items())))


class EventTimeline:
    def __init__(self, events=()) -> None:
        self.events = tuple(sorted(events, key=lambda e: e.t))

    def observed(self, t: float) -> tuple:
        return tuple(e for e in self.events if e.t <= t + 1e-9)

    def state_at(self, t: float) -> RoadState:
        return state_after(self.observed(t))

    def __bool__(self) -> bool:
        return bool(self.events)


def make_event(
    roads: RoadMap,
    ugv_edge_visits: list,
    t_static: float,
    kind: str,
    level: str,
    timing: str,
    disconnect: bool = False,
    rng: np.random.Generator | None = None,
) -> RoadEvent:
    """Build the event for one test case.

    The edges are the ones the UGV is about to drive on in the *static-road* run of the
    base policy after the trigger time, so the event actually bites. `ugv_edge_visits`
    is a list of (time, edge_id) in drive order. Unless `disconnect=True`, a blocked edge
    must keep the map connected (the deck's risk mitigation: first test blockages that
    keep an alternative path).
    """
    t_e = TIMINGS[timing] * t_static
    n_edges = (BLOCK_EDGES if kind == "block" else SLOW_EDGES)[level]
    upcoming = []
    for t, e in ugv_edge_visits:
        if t >= t_e and e not in upcoming:
            upcoming.append(e)
    if not upcoming:                       # trigger after the UGV's last drive: take the tail
        upcoming = list(dict.fromkeys(e for _, e in ugv_edge_visits))[::-1]
    chosen: list[int] = []
    pool = list(upcoming)
    if kind == "block" and not disconnect:
        # prefer edges whose joint removal leaves the map connected
        for e in pool:
            trial = RoadState(frozenset(chosen + [e]))
            if roads.is_connected(trial):
                chosen.append(e)
            if len(chosen) == n_edges:
                break
    elif kind == "block" and disconnect:
        bridges = [e for e in pool if roads.is_bridge(e)]
        chosen = bridges[:1]
    else:
        chosen = pool[:n_edges]
    if not chosen:
        if rng is None:
            rng = np.random.default_rng(0)
        cand = [e for e in range(len(roads.edges)) if not roads.is_bridge(e)]
        chosen = list(rng.choice(cand, size=min(n_edges, len(cand)), replace=False))
    return RoadEvent(
        t=float(t_e), kind=kind, edges=tuple(int(e) for e in chosen),
        kappa=SLOW_KAPPA[level] if kind == "slow" else 1.0,
    )

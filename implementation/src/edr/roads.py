"""Road network with dynamic edge states (blocked / slowed) and shortest-path queries.

A `RoadMap` is immutable geometry. A `RoadState` says which edges are blocked and which
are slowed (travel-cost multiplier kappa > 1). Shortest paths are cached per state.
All path costs are in "effective km" (length * kappa); divide by the UGV speed for minutes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

INF = float("inf")


@dataclass(frozen=True)
class RoadState:
    blocked: frozenset = frozenset()
    kappa: tuple = ()  # sorted tuple of (edge_id, multiplier)

    def kappa_of(self, edge_id: int) -> float:
        for e, k in self.kappa:
            if e == edge_id:
                return k
        return 1.0


NORMAL = RoadState()


class RoadMap:
    def __init__(self, xy: np.ndarray, edges: list[tuple[int, int]], name: str = "roads") -> None:
        self.name = name
        self.xy = np.asarray(xy, dtype=float)
        self.n = len(self.xy)
        self.edges = [tuple(sorted(e)) for e in edges]
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("duplicate road edges")
        self.length = np.array(
            [math.dist(self.xy[u], self.xy[v]) for u, v in self.edges], dtype=float
        )
        self.edge_id = {e: i for i, e in enumerate(self.edges)}
        self.adj: list[list[tuple[int, int]]] = [[] for _ in range(self.n)]
        for i, (u, v) in enumerate(self.edges):
            self.adj[u].append((v, i))
            self.adj[v].append((u, i))
        self._cache: dict[RoadState, tuple[np.ndarray, np.ndarray]] = {}
        self.mandatory: tuple = ()      # road nodes that are always ground task points (map-specific)
        self.band = None                # ((x0,y0),(x1,y1),half_width): sample aerial tasks in this strip
        if not self.is_connected(NORMAL):
            raise ValueError("road map must be connected in its normal state")

    def eid(self, a: int, b: int) -> int:
        return self.edge_id[(a, b) if a < b else (b, a)]

    def edge_cost(self, state: RoadState) -> np.ndarray:
        cost = self.length.copy()
        for e, k in state.kappa:
            cost[e] *= k
        for e in state.blocked:
            cost[e] = INF
        return cost

    def _solve(self, state: RoadState) -> tuple[np.ndarray, np.ndarray]:
        if state in self._cache:
            return self._cache[state]
        n = self.n
        cost = self.edge_cost(state)
        dist = np.full((n, n), INF)
        nxt = np.full((n, n), -1, dtype=int)
        np.fill_diagonal(dist, 0.0)
        for i in range(n):
            nxt[i, i] = i
        for e, (u, v) in enumerate(self.edges):
            if cost[e] < dist[u, v]:
                dist[u, v] = dist[v, u] = cost[e]
                nxt[u, v] = v
                nxt[v, u] = u
        for k in range(n):
            via = dist[:, k : k + 1] + dist[k : k + 1, :]
            better = via < dist - 1e-12
            dist = np.where(better, via, dist)
            nxt = np.where(better, nxt[:, k : k + 1], nxt)
        self._cache[state] = (dist, nxt)
        return dist, nxt

    def dist_matrix(self, state: RoadState = NORMAL) -> np.ndarray:
        """All-pairs effective distance (km); inf where unreachable."""
        return self._solve(state)[0]

    def next_hop(self, src: int, dst: int, state: RoadState) -> int:
        """First node on a shortest src->dst path (-1 if unreachable)."""
        return int(self._solve(state)[1][src, dst])

    def path(self, src: int, dst: int, state: RoadState = NORMAL) -> list[int]:
        if not np.isfinite(self.dist_matrix(state)[src, dst]):
            return []
        out = [src]
        while out[-1] != dst:
            out.append(self.next_hop(out[-1], dst, state))
        return out

    def is_connected(self, state: RoadState) -> bool:
        cost = self.edge_cost(state)
        seen, stack = {0}, [0]
        while stack:
            a = stack.pop()
            for b, e in self.adj[a]:
                if np.isfinite(cost[e]) and b not in seen:
                    seen.add(b)
                    stack.append(b)
        return len(seen) == self.n

    def is_bridge(self, edge_id: int, state: RoadState = NORMAL) -> bool:
        blocked = RoadState(state.blocked | {edge_id}, state.kappa)
        return not self.is_connected(blocked)

    def travel(self, src: int, dst: int, t0: float, speed_kmmin: float, state_at):
        """Time-dependent UGV drive. The driver re-routes at every node using the road
        state at that moment (so a blockage met en route is detoured around).

        `state_at(t) -> RoadState`. Returns (arrival_time, trace) with trace = [(node, time)];
        if `dst` becomes unreachable on the way, arrival_time is None and the last trace entry
        is where the UGV got stuck.
        """
        node, t = src, t0
        trace = [(node, t)]
        for _ in range(4 * self.n):
            if node == dst:
                return t, trace
            state = state_at(t)
            nxt = self.next_hop(node, dst, state)
            if nxt < 0:
                return None, trace
            e = self.eid(node, nxt)
            t += self.length[e] * state.kappa_of(e) / speed_kmmin
            node = nxt
            trace.append((node, t))
        return None, trace


def tree_roads() -> RoadMap:
    """Branching road tree (no cycles): a base-paper-like layout used only for the
    M1 reproduction check. A blocked edge here disconnects the map."""
    pts: list[tuple[float, float]] = [(6.0, 8.0)]
    edges: list[tuple[int, int]] = []

    def branch(points):
        prev = 0
        for p in points:
            pts.append(p)
            edges.append((prev, len(pts) - 1))
            prev = len(pts) - 1

    branch([(6.0 - 0.75 * i, 8.0) for i in range(1, 5)])
    branch([(6.0 + 0.55 * i, 8.0 + 0.55 * i) for i in range(1, 8)])
    branch([(6.0 + 0.6 * i, 8.0 - 0.45 * i) for i in range(1, 7)])
    return RoadMap(np.array(pts), edges, name="tree")


def loop_roads() -> RoadMap:
    """Road map WITH loops so that a blocked edge normally has a detour.

    A 24-node ring (R = 2.6 km around (8, 8)), a north-south chord with 5 interior
    nodes (theta shape), an east spur (5), a west spur (4) and a north-east spur (3).
    41 nodes; ring + chord edges are non-bridges, spur edges are bridges.
    """
    cx, cy, radius = 8.0, 8.0, 2.6
    pts: list[tuple[float, float]] = []
    edges: list[tuple[int, int]] = []
    ring = 24
    for k in range(ring):
        ang = 2 * math.pi * k / ring
        pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
    for k in range(ring):
        edges.append((k, (k + 1) % ring))
    # chord from the top ring node (k=6, angle 90deg) to the bottom (k=18)
    prev = 6
    for i in range(1, 6):
        pts.append((cx, cy + radius - i * (2 * radius) / 6))
        edges.append((prev, len(pts) - 1))
        prev = len(pts) - 1
    edges.append((prev, 18))

    def spur(start, points):
        prev = start
        for p in points:
            pts.append(p)
            edges.append((prev, len(pts) - 1))
            prev = len(pts) - 1

    spur(0, [(cx + radius + 0.6 * i, cy + 0.1 * i) for i in range(1, 6)])        # east
    spur(12, [(cx - radius - 0.6 * i, cy - 0.08 * i) for i in range(1, 5)])        # west
    spur(4, [(cx + 1.3 + 0.5 * i, cy + radius + 0.4 * i) for i in range(1, 4)])   # north-east
    return RoadMap(np.array(pts), edges, name="loop")


def grid_roads() -> RoadMap:
    """City-block grid: 6x6 junctions, 1.8 km spacing, ~15% of the links removed (fixed seed) while
    staying connected. Many alternative routes, so a blocked edge normally costs a short detour."""
    n, sp, ox, oy = 6, 1.8, 4.5, 4.5
    xy = [(ox + i * sp, oy + j * sp) for j in range(n) for i in range(n)]
    idx = lambda i, j: j * n + i
    edges = [(idx(i, j), idx(i + 1, j)) for j in range(n) for i in range(n - 1)] + \
            [(idx(i, j), idx(i, j + 1)) for j in range(n - 1) for i in range(n)]
    rng = np.random.default_rng(7)
    for k in rng.permutation(len(edges)):
        if len(edges) <= int(0.85 * 60):
            break
        trial = [e for e in edges if e != edges[k]]
        try:
            RoadMap(np.array(xy), trial)
        except ValueError:
            continue
        edges = trial
    return RoadMap(np.array(xy), edges, name="grid")


def radial_roads() -> RoadMap:
    """Ring-radial town: centre + 3 rings (R = 1.6, 3.2, 4.8 km, 12 nodes each) + 12 spokes."""
    cx, cy, rings, per = 9.0, 9.0, (1.6, 3.2, 4.8), 12
    xy = [(cx, cy)]
    edges = []
    for r_i, rad in enumerate(rings):
        for k in range(per):
            ang = 2 * math.pi * k / per
            xy.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    node = lambda r_i, k: 1 + r_i * per + (k % per)
    for r_i in range(len(rings)):
        for k in range(per):
            edges.append((node(r_i, k), node(r_i, k + 1)))          # ring
            if r_i + 1 < len(rings):
                edges.append((node(r_i, k), node(r_i + 1, k)))      # spoke segment
    for k in range(0, per, 2):
        edges.append((0, node(0, k)))                                # centre to every other inner node
    return RoadMap(np.array(xy), edges, name="radial")


def geo_roads() -> RoadMap:
    """Irregular planar road graph (like a real district): 44 random junctions in a 11x11 km patch
    (min spacing 1.1 km, fixed seed), Delaunay triangulation, long and redundant links pruned
    to mean degree ~2.9 while connected."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(11)
    pts: list[np.ndarray] = []
    while len(pts) < 44:
        p = rng.uniform(0, 11, 2)
        if all(np.linalg.norm(p - q) >= 1.1 for q in pts):
            pts.append(p)
    xy = np.array(pts) + np.array([3.5, 3.5])
    tri = Delaunay(xy)
    es = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            u, v = sorted((int(simplex[a]), int(simplex[b])))
            es.add((u, v))
    edges = sorted(es, key=lambda e: -math.dist(xy[e[0]], xy[e[1]]))
    target = int(2.9 * len(xy) / 2)
    keep = list(edges)
    for e in edges:
        if len(keep) <= target:
            break
        trial = [x for x in keep if x != e]
        try:
            RoadMap(xy, trial)
        except ValueError:
            continue
        keep = trial
    return RoadMap(xy, keep, name="geo")


# The 28 ground-point candidates of the base paper's scenario generator (authors' public repo,
# scenario_generator_v2.py: all_ugv_points; units read as km). Three chains meet at (6, 8).
PAPER_POINTS = [(6, 8), (5.5, 8), (5, 8), (4.5, 8), (4, 8), (3.5, 8), (3, 8),
                (6.36, 8.36), (6.73, 8.73), (7.09, 9.09), (7.45, 9.45), (7.82, 9.82), (8.18, 10.18), (8.55, 10.55),
                (8.91, 10.91), (9.27, 11.27), (9.64, 11.64), (10, 12),
                (6.4, 7.7), (6.8, 7.4), (7.2, 7.1), (7.6, 6.8), (8, 6.5), (8.4, 6.2), (8.8, 5.9), (9.2, 5.6), (9.6, 5.3), (10, 5)]


def paper_roads() -> RoadMap:
    """The base paper's map: west chain (idx 0-6), north-east chain (7-17), south-east chain (18-27),
    joined at the junction (6, 8) = idx 0. The authors' `ugv_road.csv` (edge list) is not public, so the
    connections are reconstructed from the chain order; the always-present mandatory points are the
    three chain ends (3, 8), (10, 12), (10, 5) as in the authors' generator."""
    xy = np.array(PAPER_POINTS, float)
    edges = [(i, i + 1) for i in range(0, 6)]                       # west chain
    edges += [(0, 7)] + [(i, i + 1) for i in range(7, 17)]          # north-east chain
    edges += [(0, 18)] + [(i, i + 1) for i in range(18, 27)]        # south-east chain
    rm = RoadMap(xy, edges, name="paper")
    rm.mandatory = (6, 17, 27)
    return rm


def border_roads() -> RoadMap:
    """Border-surveillance region (army patrol): a slanted border (y = 16.5 - 0.25 x), a patrol road 1.6 km
    south of it, a parallel inner supply road 5 km south of the border, four connector roads between them
    (so there are loops and detours) and three forward-outpost spurs toward the border (mandatory points).
    Aerial surveillance points are sampled in a strip along the border instead of around roads."""
    yb = lambda x: 16.5 - 0.25 * x
    pts: list[tuple[float, float]] = []
    edges: list[tuple[int, int]] = []

    def add(p):
        pts.append(p)
        return len(pts) - 1

    border = [add((x, yb(x) - 1.6 + 0.25 * math.sin(1.3 * x))) for x in np.arange(2.0, 18.01, 1.0)]
    inner = [add((x, yb(x) - 5.0 + 0.3 * math.sin(0.9 * x + 1.0))) for x in np.arange(3.0, 16.3, 1.2)]
    edges += list(zip(border[:-1], border[1:])) + list(zip(inner[:-1], inner[1:]))
    for xb in (4.0, 8.0, 12.0, 16.0):                                 # connector roads
        b = border[int(round(xb - 2.0))]
        j = min(range(len(inner)), key=lambda k: abs(pts[inner[k]][0] - xb))
        top, bot = np.array(pts[b]), np.array(pts[inner[j]])
        prev = b
        for f in (0.25, 0.5, 0.75):
            q = top + f * (bot - top) + np.array([0.15 * math.sin(6 * f + xb), 0.0])
            cur = add((float(q[0]), float(q[1])))
            edges.append((prev, cur)); prev = cur
        edges.append((prev, inner[j]))
    outposts = []
    for xo in (6.0, 11.0, 15.0):                                      # forward outposts toward the border
        b = border[int(round(xo - 2.0))]
        x0, y0 = pts[b]
        a1 = add((x0 + 0.15, y0 + 0.6)); a2 = add((x0 + 0.3, y0 + 1.2))
        edges += [(b, a1), (a1, a2)]
        outposts.append(a2)
    rm = RoadMap(np.array(pts), edges, name="border")
    rm.mandatory = tuple(outposts)
    rm.band = ((1.5, yb(1.5) - 1.8), (18.5, yb(18.5) - 1.8), 2.2)
    return rm


ROAD_BUILDERS = {"tree": tree_roads, "paper": paper_roads, "loop": loop_roads, "grid": grid_roads,
                 "radial": radial_roads, "geo": geo_roads, "border": border_roads}

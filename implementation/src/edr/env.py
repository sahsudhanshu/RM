"""Sortie-wise UAV-UGV mission simulator with mission-energy accounting and road events.

Protocol (base paper, Sec. IV-A "agent selection strategy"):
  round = every UAV flies one sortie (smallest time dimension first), ending at a
  rendezvous on a ground point; then the UGVs serve the landed UAVs: greedy UAV->UGV
  assignment by proximity, one UAV at a time per UGV; then the next round starts.

Action space (size K + G): a < K  -> visit task a; a >= K -> recharge at ground task a-K.

Energy: E = sum over UAVs of  P_fly * t_fly + P_wait * t_wait  (kJ). The per-sortie battery
constraint F_a only covers flight energy (as in the base paper); hover waiting is charged
to the mission energy budget (see context/context.md, assumption 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import EventTimeline
from .params import Params
from .roads import NORMAL, RoadState
from .scenario import Scenario


@dataclass
class Request:
    uav: int
    task: int           # ground task index where the UAV landed
    node: int           # road node of that task
    touchdown: float


@dataclass
class MissionResult:
    makespan: float
    energy: float
    energy_fly: float
    energy_wait: float
    wait_time: float
    failed: bool
    reason: str | None
    cost: float                 # makespan + penalty * failed
    rounds: int
    sorties: int
    visited: int
    n_tasks: int
    legs: list = field(default_factory=list)       # UAV flight legs
    services: list = field(default_factory=list)   # recharge services
    ugv_drives: list = field(default_factory=list) # (ugv, t0, t1, node_from, node_to, edge, kappa)
    actions: list = field(default_factory=list)    # (round, uav, action) in execution order


class MissionEnv:
    def __init__(
        self,
        scenario: Scenario,
        n_uav: int,
        n_ugv: int,
        params: Params = Params(),
        events=(),
        road_aware: bool = False,
        assign_metric: str = "euclid",
    ) -> None:
        if assign_metric not in {"euclid", "road"}:
            raise ValueError("assign_metric must be 'euclid' or 'road'")
        self.scn = scenario
        self.p = params
        self.M, self.N = n_uav, n_ugv
        self.K, self.G = scenario.n_tasks, scenario.n_ground
        self.timeline = EventTimeline(events)        # what physically happens
        self.belief: RoadState = NORMAL              # what the planner knows
        self.road_aware = road_aware
        self.divert = False                          # replanning controllers may re-target a stranded UAV
        self.jit = False                             # just-in-time departure: a UAV that expects to wait waits on the ground (free) instead of hovering
        self.assign_metric = assign_metric
        self.roads = scenario.roads

        xy = scenario.xy
        self.task_dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)   # (K, K)
        self._refresh_ground_info()

        depot = scenario.depot_task
        self.visited = np.zeros(self.K, dtype=bool)
        self.visited[depot] = True
        self.cur = np.full(self.M, depot, dtype=int)           # task index of each UAV
        self.t_uav = np.zeros(self.M)
        self.fuel = np.full(self.M, params.fuel_kj)
        self.e_fly = np.zeros(self.M)
        self.e_wait = np.zeros(self.M)
        self.wait_t = np.zeros(self.M)
        self.ugv_node = np.full(self.N, scenario.depot_node, dtype=int)
        self.ugv_time = np.zeros(self.N)
        self.queue: list[int] = list(range(self.M))   # all UAVs start at t=0 at the depot
        self._cur_uav: int | None = None
        self.sortie_has_task = False
        self.pending: list[Request] = []
        self.rounds = 0
        self.sorties = 0
        self.done = False
        self.failed = False
        self.reason: str | None = None
        self.legs: list = []
        self.services: list = []
        self.ugv_drives: list = []
        self.actions: list = []
        self.budget: float | None = None            # mission energy budget B (kJ), if any
        self._best_visited, self._last_progress = int(self.visited.sum()), 0   # loop detection (see stalled_rounds)

    # ------------------------------------------------------------------ state helpers
    def clone(self) -> "MissionEnv":
        new = object.__new__(MissionEnv)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(new, k, v.copy())
            elif isinstance(v, list):
                setattr(new, k, list(v))
            else:
                setattr(new, k, v)
        new.pending = [Request(r.uav, r.task, r.node, r.touchdown) for r in self.pending]
        return new

    def with_timeline(self, events=()) -> "MissionEnv":
        """Clone with a different *true* event timeline (used for belief-world planning)."""
        new = self.clone()
        new.timeline = EventTimeline(events)
        return new

    @property
    def energy_used(self) -> float:
        return float(self.e_fly.sum() + self.e_wait.sum())

    @property
    def budget_left_frac(self) -> float:
        if self.budget is None:
            return 1.0
        return (self.budget - self.energy_used) / self.budget

    def observe(self, t: float) -> bool:
        """Update the planner's road belief with every event that has occurred by time t.
        Returns True if the belief changed."""
        new = self.timeline.state_at(t)
        if new != self.belief:
            self.belief = new
            self._refresh_ground_info()
            return True
        return False

    def _refresh_ground_info(self) -> None:
        """Ground tasks the UGVs can reach under the believed road state."""
        d = self.roads.dist_matrix(self.belief)
        nodes = self.scn.node[: self.G]
        depot = self.scn.depot_node
        # a ground task is usable if it is road-connected to the depot component
        self.ground_ok = np.isfinite(d[depot, nodes])
        g_xy = self.scn.xy[: self.G]
        dist_tg = np.linalg.norm(self.scn.xy[:, None, :] - g_xy[None, :, :], axis=-1)  # (K, G)
        masked = np.where(self.ground_ok[None, :], dist_tg, np.inf) if self.road_aware else dist_tg
        self.near_ground = masked.min(axis=1)       # distance to nearest (usable) ground task

    # ------------------------------------------------------------------ decision loop
    def settle(self) -> None:
        """Bring the env to a decision point without starting a sortie: once no UAV is left
        to fly in the current round, serve the landed UAVs (UGV phase) and open the next
        round, or end the mission."""
        if self.done or self._cur_uav is not None:
            return
        if self.visited.all():
            self.queue.clear()                       # remaining UAVs have nothing to fly
        if self.queue:
            return
        if self.pending:
            self._service_round()                    # sets self.queue for the next round
            if self.failed:
                return
            if self.visited.all():
                self.done = True
            elif self.rounds >= self.p.max_rounds:
                self._fail("max rounds exceeded")
            return
        self.done = True
        if not self.visited.all():
            self._fail("no UAV could continue")

    def active_uav(self) -> int | None:
        """The UAV that must act now (advances rounds / UGV service as needed).
        Returns None when the mission is over (check `env.failed`)."""
        self.settle()
        if self.done:
            return None
        if self._cur_uav is None:
            self._cur_uav = self.queue.pop(0)
            self.sortie_has_task = False
            self.sorties += 1
        return self._cur_uav

    def set_round_order(self, order) -> None:
        """Override the decision order for the coming round (used when executing a plan)."""
        if self._cur_uav is not None or self.pending:
            raise RuntimeError("the round order can only be set at a round boundary")
        self.queue = list(order)

    def round_start_time(self) -> float:
        return float(self.t_uav.min())

    def _fail(self, reason: str) -> None:
        self.failed = True
        self.reason = reason
        self.done = True
        self._cur_uav = None

    # ------------------------------------------------------------------ action mask
    def _flight_kj(self, dist_km):
        return dist_km * self.p.fly_kj_per_km

    def mask(self) -> np.ndarray:
        """Feasible actions for the active UAV (size K+G). If nothing is feasible the
        mission is marked failed ('UAV stuck') and `env.done` becomes True."""
        u = self._cur_uav
        if u is None:
            raise RuntimeError("no active UAV")
        K, G, eps = self.K, self.G, 1e-6
        c, fuel = self.cur[u], self.fuel[u]
        d = self.task_dist[c]
        m = np.zeros(K + G, dtype=bool)
        visit = (~self.visited) & (self._flight_kj(d + self.near_ground) <= fuel + eps)
        recharge = self._flight_kj(d[:G]) <= fuel + eps
        if self.road_aware:
            recharge &= self.ground_ok
        if not self.sortie_has_task:
            if visit.any():
                recharge[:] = False            # a sortie must make progress before landing
            elif c < G and recharge.sum() > 1:
                recharge[c] = False            # nothing to visit: reposition, not recharge in place
        m[:K] = visit
        m[K:] = recharge
        if self.road_aware and recharge.any():
            # Road layer: keep rendezvous that are reachable ON TIME (projected wait within
            # tolerance), but never at the price of progress: prefer rendezvous from which an
            # unvisited task can still be flown (otherwise a late-everywhere situation lets
            # the UAVs shuttle between two landing sites forever).
            wait = np.array([self.projected_wait(u, g) if recharge[g] else np.inf for g in range(G)])
            finite = np.isfinite(wait)
            on_time = finite & (wait <= self.p.wait_tol_min)
            useful = self._useful_rendezvous()
            m[K:] = False
            if (on_time & useful).any():
                m[K:] = on_time & useful
            elif (finite & useful).any():                  # nothing on time: least-late useful
                m[K + int(np.argmin(np.where(finite & useful, wait, np.inf)))] = True
            elif on_time.any():
                m[K:] = on_time
            elif finite.any():
                m[K + int(np.argmin(wait))] = True
        if not m.any():
            self._fail("UAV stuck: no feasible action")
        return m

    def _useful_rendezvous(self) -> np.ndarray:
        """(G,) True where, after a full recharge at that ground task, some unvisited task
        could be visited and the UAV could still reach a usable ground point afterwards."""
        if self.visited.all():
            return np.ones(self.G, dtype=bool)
        reach = self._flight_kj(self.task_dist[: self.G] + self.near_ground[None, :]) <= self.p.fuel_kj + 1e-6
        return (reach & ~self.visited[None, :]).any(axis=1)

    def dyn_features(self) -> np.ndarray:
        """(K+G, 2) float32 decoder features for the active UAV (km / 10)."""
        c = self.cur[self._cur_uav]
        out = np.zeros((self.K + self.G, 2), dtype=np.float32)
        out[: self.K, 0] = self.task_dist[c] / 10.0
        out[self.K :, 0] = self.task_dist[c, : self.G] / 10.0
        unv = ~self.visited
        if unv.any():
            out[self.K :, 1] = self.task_dist[: self.G][:, unv].min(axis=1) / 10.0
        return out

    # ------------------------------------------------------------------ UGV planning view
    def _assign(self, requests, ugv_node, state: RoadState):
        """Greedy proximity assignment (frozen base rule). Returns {ugv: [Request]}."""
        groups: dict[int, list[Request]] = {n: [] for n in range(self.N)}
        d_road = self.roads.dist_matrix(state) if self.assign_metric == "road" else None
        for r in requests:
            if self.assign_metric == "euclid":
                xy_r = self.roads.xy[r.node]
                score = [float(np.linalg.norm(self.roads.xy[ugv_node[n]] - xy_r)) for n in range(self.N)]
            else:
                score = [float(d_road[ugv_node[n], r.node]) for n in range(self.N)]
            n_best = int(np.argmin(score))          # ties -> lowest id
            groups[n_best].append(r)
        for n in groups:
            groups[n].sort(key=lambda r: (r.touchdown, r.uav))
        return groups

    def projected_wait(self, u: int, g: int) -> float:
        """Believed wait (min) if UAV u landed at ground task g now, after the pending
        landings of this round are served (static belief roads)."""
        t_land = self.t_uav[u] + self.task_dist[self.cur[u], g] / self.p.uav_kmmin
        cand = Request(u, g, int(self.scn.node[g]), float(t_land))
        reqs = self.pending + [cand]
        groups = self._assign(reqs, self.ugv_node, self.belief)
        d = self.roads.dist_matrix(self.belief)
        for n, queue in groups.items():
            node, ready = self.ugv_node[n], self.ugv_time[n]
            for r in queue:
                if not np.isfinite(d[node, r.node]):
                    if r is cand:
                        return np.inf
                    continue
                arr = ready + d[node, r.node] / self.p.ugv_kmmin
                start = max(arr, r.touchdown)
                if r is cand:
                    return float(start - r.touchdown)
                node, ready = r.node, start + self.p.recharge_min
        return np.inf

    # ------------------------------------------------------------------ stepping
    def step(self, action: int) -> None:
        u = self._cur_uav
        if u is None or self.done:
            raise RuntimeError("no active UAV")
        K, G = self.K, self.G
        if not (0 <= action < K + G):
            raise ValueError("action out of range")
        land = action >= K
        target = action - K if land else action
        c = self.cur[u]
        dist = float(self.task_dist[c, target])
        e = self._flight_kj(dist)
        if e > self.fuel[u] + 1e-6:
            raise RuntimeError(f"UAV {u} would run out of battery in flight")
        t0 = self.t_uav[u]
        self.t_uav[u] += dist / self.p.uav_kmmin
        self.fuel[u] -= e
        self.e_fly[u] += e
        self.cur[u] = target
        self.visited[target] = True
        self.legs.append((u, t0, float(self.t_uav[u]), tuple(self.scn.xy[c]), tuple(self.scn.xy[target]),
                          "land" if land else "visit", float(e), self.rounds))
        self.actions.append((self.rounds, u, int(action)))
        if land:
            self.pending.append(Request(u, target, int(self.scn.node[target]), float(self.t_uav[u])))
            self._cur_uav = None
        else:
            self.sortie_has_task = True

    # ------------------------------------------------------------------ UGV phase (true world)
    def _service_round(self) -> None:
        groups = self._assign(self.pending, self.ugv_node, self.belief)
        for n, queue in groups.items():
            for r in queue:
                if not self._serve(n, r):
                    self.pending = []
                    return
        self.pending = []
        self.rounds += 1
        if int(self.visited.sum()) > self._best_visited:
            self._best_visited, self._last_progress = int(self.visited.sum()), self.rounds
        self.queue = sorted(range(self.M), key=lambda i: (self.t_uav[i], i))

    @property
    def stalled_rounds(self) -> int:
        """Rounds since a new task was last visited (a learned policy that loops keeps this growing)."""
        return self.rounds - self._last_progress

    def _log_drive(self, n, trace) -> None:
        for (a, ta), (b, tb) in zip(trace[:-1], trace[1:]):
            eid = self.roads.eid(a, b)
            self.ugv_drives.append((n, ta, tb, a, b, eid, self.timeline.state_at(ta).kappa_of(eid)))

    def _serve(self, n: int, r: Request) -> bool:
        """UGV n drives to the landed UAV and recharges it. Returns False on mission failure."""
        predicted = None
        if self.jit:                                     # wait the planner expects, from its BELIEVED roads
            d_b = self.roads.dist_matrix(self.belief)[int(self.ugv_node[n]), r.node]
            if np.isfinite(d_b):
                predicted = max(0.0, self.ugv_time[n] + d_b / self.p.ugv_kmmin - r.touchdown)
        while True:
            arrival, trace = self.roads.travel(int(self.ugv_node[n]), r.node, float(self.ugv_time[n]),
                                               self.p.ugv_kmmin, self.timeline.state_at)
            self._log_drive(n, trace)
            if arrival is not None:
                break
            node_x, t_x = trace[-1]                      # the UGV is stuck here: road cut ahead
            if not self.divert:
                self._fail(f"UGV {n} cannot reach rendezvous node {r.node}")
                return False
            r2 = self._divert(n, r, node_x, t_x)
            if r2 is None:
                self._fail(f"UGV {n} cannot reach rendezvous node {r.node} and no UAV diversion is possible")
                return False
            r = r2
        start = max(arrival, r.touchdown)
        wait = start - r.touchdown
        end = start + self.p.recharge_min
        delay = min(wait, predicted) if (self.jit and predicted is not None) else 0.0   # spent on the ground, not hovering
        self.e_wait[r.uav] += (wait - delay) * self.p.hover_kj_per_min
        self.wait_t[r.uav] += wait
        self.t_uav[r.uav] = end
        self.fuel[r.uav] = self.p.fuel_kj
        self.ugv_node[n], self.ugv_time[n] = r.node, end
        self.services.append(dict(
            uav=r.uav, ugv=n, node=r.node, task=r.task, touchdown=r.touchdown,
            ugv_arrival=float(arrival), start=float(start), end=float(end),
            wait=float(wait), round=self.rounds, delay=float(delay),
        ))
        return True

    def _divert(self, n: int, r: Request, node_x: int, t_x: float) -> Request | None:
        """The planner learns the rendezvous is stranded: the hovering UAV flies on its remaining
        battery to the ground point where it can meet the UGV soonest. None if impossible."""
        u, eps = r.uav, 1e-6
        t_learn = max(t_x, r.touchdown)
        d = self.roads.dist_matrix(self.timeline.state_at(t_learn))
        best = None
        for g in range(self.G):
            node_g = int(self.scn.node[g])
            if g == r.task or not np.isfinite(d[node_x, node_g]):
                continue
            e = self._flight_kj(self.task_dist[r.task, g])
            if e > self.fuel[u] + eps:
                continue
            ft = self.task_dist[r.task, g] / self.p.uav_kmmin
            meet = max(t_learn + ft, t_learn + d[node_x, node_g] / self.p.ugv_kmmin)
            if best is None or meet < best[0]:
                best = (meet, g, ft, e, node_g)
        if best is None:
            return None
        _, g, ft, e, node_g = best
        pre = t_learn - r.touchdown                       # hovered at the stranded rendezvous
        self.e_wait[u] += pre * self.p.hover_kj_per_min
        self.wait_t[u] += pre
        self.legs.append((u, t_learn, t_learn + ft, tuple(self.scn.xy[r.task]), tuple(self.scn.xy[g]),
                          "land", float(e), self.rounds))
        self.fuel[u] -= e
        self.e_fly[u] += e
        self.cur[u] = g
        self.visited[g] = True
        self.ugv_node[n], self.ugv_time[n] = node_x, t_learn
        return Request(u, g, node_g, float(t_learn + ft))

    # ------------------------------------------------------------------ result
    def result(self) -> MissionResult:
        t_end = max([s["end"] for s in self.services], default=float(self.t_uav.max()))
        if self.failed:
            t_end = max(t_end, float(self.t_uav.max()), float(self.ugv_time.max()))
        e_fly, e_wait = float(self.e_fly.sum()), float(self.e_wait.sum())
        return MissionResult(
            makespan=float(t_end), energy=e_fly + e_wait, energy_fly=e_fly, energy_wait=e_wait,
            wait_time=float(self.wait_t.sum()), failed=self.failed, reason=self.reason,
            cost=float(t_end) + (self.p.penalty if self.failed else 0.0),
            rounds=self.rounds, sorties=self.sorties, visited=int(self.visited.sum()),
            n_tasks=self.K, legs=list(self.legs), services=list(self.services),
            ugv_drives=list(self.ugv_drives), actions=list(self.actions),
        )

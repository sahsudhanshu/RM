"""Mission instances: road map + ground/aerial task points + depot."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .params import Params
from .roads import ROAD_BUILDERS, RoadMap

_ROADS: dict[str, RoadMap] = {}


def get_roads(name: str) -> RoadMap:
    if name not in _ROADS:
        _ROADS[name] = ROAD_BUILDERS[name]()
    return _ROADS[name]


@dataclass(frozen=True)
class Scenario:
    """Tasks are indexed 0..G-1 ground (on road nodes), G..K-1 aerial."""

    roads: RoadMap
    xy: np.ndarray          # (K, 2) km
    n_ground: int
    node: np.ndarray        # (K,) road node of ground tasks, -1 for aerial
    depot_task: int         # index of the ground task where everyone starts
    seed: int
    width_km: float = 20.0
    height_km: float = 20.0

    @property
    def n_tasks(self) -> int:
        return len(self.xy)

    @property
    def n_aerial(self) -> int:
        return self.n_tasks - self.n_ground

    @property
    def depot_node(self) -> int:
        return int(self.node[self.depot_task])

    @property
    def name(self) -> str:
        return f"U{self.n_aerial}G{self.n_ground}"

    def features(self) -> np.ndarray:
        """Encoder input (K, 3): normalised x, y and the ground flag b_i."""
        f = np.zeros((self.n_tasks, 3), dtype=np.float32)
        f[:, 0] = self.xy[:, 0] / self.width_km
        f[:, 1] = self.xy[:, 1] / self.height_km
        f[: self.n_ground, 2] = 1.0
        return f


def generate_scenario(
    seed: int,
    n_aerial: int = 15,
    n_ground: int = 5,
    roads: str = "loop",
    params: Params = Params(),
) -> Scenario:
    rm = get_roads(roads)
    if n_ground > rm.n:
        raise ValueError(f"{roads} map has only {rm.n} road nodes")
    rng = np.random.default_rng(seed)
    mand = list(rm.mandatory)
    if not mand:                                             # unchanged sampling: keeps earlier results reproducible
        gnodes = rng.choice(rm.n, size=n_ground, replace=False)
    else:
        if n_ground < len(mand):
            raise ValueError(f"{roads} map needs at least {len(mand)} ground points (its mandatory outposts)")
        rest = [n for n in range(rm.n) if n not in mand]
        gnodes = np.array(mand + [int(n) for n in rng.choice(rest, size=n_ground - len(mand), replace=False)], dtype=int)
        rng.shuffle(gnodes)
    ground_xy = rm.xy[gnodes]
    aerial = np.zeros((n_aerial, 2))
    for i in range(n_aerial):
        for _ in range(10_000):
            if rm.band is not None:                          # surveillance strip along a border
                (ax0, ay0), (ax1, ay1), half = rm.band
                f = rng.uniform()
                dx, dy = ax1 - ax0, ay1 - ay0
                nrm = math.hypot(dx, dy)
                off = rng.uniform(-half, half)
                x = ax0 + f * dx - off * dy / nrm
                y = ay0 + f * dy + off * dx / nrm
            else:                                            # base paper: disc around a ground point
                ax, ay = ground_xy[rng.integers(n_ground)]
                ang = rng.uniform(0.0, 2.0 * math.pi)
                rad = params.aerial_radius_km * math.sqrt(rng.random())  # uniform over the disc
                x, y = ax + rad * math.cos(ang), ay + rad * math.sin(ang)
            if 0.0 <= x <= params.width_km and 0.0 <= y <= params.height_km:
                aerial[i] = (x, y)
                break
        else:  # pragma: no cover
            raise RuntimeError("could not place an aerial task inside the map")
    xy = np.vstack([ground_xy, aerial])
    node = np.full(len(xy), -1, dtype=int)
    node[:n_ground] = gnodes
    return Scenario(
        roads=rm, xy=xy, n_ground=n_ground, node=node,
        depot_task=int(rng.integers(n_ground)), seed=seed,
        width_km=params.width_km, height_km=params.height_km,
    )

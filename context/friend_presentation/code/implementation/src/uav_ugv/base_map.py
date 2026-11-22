"""Fixed 20 km x 20 km road map and task generator used by the base setup."""

from __future__ import annotations

import math
import random

from .mission import MissionScenario, TaskKind, TaskPoint
from .road_network import RoadNetwork


MAP_WIDTH_KM = 20.0
MAP_HEIGHT_KM = 20.0
AERIAL_RADIUS_KM = 7.0

# The public base-paper generator defines three road branches around (6, 8).
# Node zero is their common junction; the three mandatory points are endpoints.
_WEST = [(6.0, 8.0), (5.5, 8.0), (5.0, 8.0), (4.5, 8.0), (4.0, 8.0), (3.5, 8.0), (3.0, 8.0)]
_NORTH_EAST = [
    (6.36, 8.36),
    (6.73, 8.73),
    (7.09, 9.09),
    (7.45, 9.45),
    (7.82, 9.82),
    (8.18, 10.18),
    (8.55, 10.55),
    (8.91, 10.91),
    (9.27, 11.27),
    (9.64, 11.64),
    (10.0, 12.0),
]
_SOUTH_EAST = [
    (6.4, 7.7),
    (6.8, 7.4),
    (7.2, 7.1),
    (7.6, 6.8),
    (8.0, 6.5),
    (8.4, 6.2),
    (8.8, 5.9),
    (9.2, 5.6),
    (9.6, 5.3),
    (10.0, 5.0),
]


def base_paper_road_network() -> RoadNetwork:
    """Return the connected, three-branch road network in kilometre units."""
    coordinates: dict[int, tuple[float, float]] = {}
    edges: list[tuple[int, int]] = []
    next_node = 0

    coordinates[next_node] = _WEST[0]
    junction = next_node
    next_node += 1
    previous = junction
    for coordinate in _WEST[1:]:
        coordinates[next_node] = coordinate
        edges.append((previous, next_node))
        previous = next_node
        next_node += 1
    for branch in (_NORTH_EAST, _SOUTH_EAST):
        previous = junction
        for coordinate in branch:
            coordinates[next_node] = coordinate
            edges.append((previous, next_node))
            previous = next_node
            next_node += 1
    return RoadNetwork.from_edges(coordinates, edges)


def mandatory_ground_nodes(road_network: RoadNetwork) -> tuple[int, int, int]:
    required_coordinates = {(3.0, 8.0), (10.0, 12.0), (10.0, 5.0)}
    lookup = {
        road_network.coordinates(node): node for node in road_network.graph.nodes
    }
    return tuple(lookup[coordinate] for coordinate in sorted(required_coordinates))


class BasePaperScenarioGenerator:
    """Generate U15G5-style scenarios on the published fixed road layout."""

    def __init__(
        self,
        width_km: float = MAP_WIDTH_KM,
        height_km: float = MAP_HEIGHT_KM,
        aerial_radius_km: float = AERIAL_RADIUS_KM,
        enforce_map_bounds: bool = True,
    ) -> None:
        if min(width_km, height_km, aerial_radius_km) <= 0:
            raise ValueError("map dimensions and radius must be positive")
        self.width_km = width_km
        self.height_km = height_km
        self.aerial_radius_km = aerial_radius_km
        self.enforce_map_bounds = enforce_map_bounds
        self.road_network = base_paper_road_network()

    def generate(
        self,
        aerial_task_count: int = 15,
        ground_task_count: int = 5,
        seed: int = 1234,
    ) -> MissionScenario:
        if aerial_task_count < 1:
            raise ValueError("aerial_task_count must be positive")
        required = mandatory_ground_nodes(self.road_network)
        if ground_task_count < len(required):
            raise ValueError(
                f"ground_task_count must be at least {len(required)} to include "
                "the base map's mandatory endpoints"
            )
        if ground_task_count > self.road_network.graph.number_of_nodes():
            raise ValueError("too many ground tasks for the fixed road map")

        rng = random.Random(seed)
        remaining = [
            node for node in self.road_network.graph.nodes if node not in required
        ]
        selected_ground = list(required) + rng.sample(
            remaining, ground_task_count - len(required)
        )
        rng.shuffle(selected_ground)
        ground_tasks = tuple(
            TaskPoint(
                task_id=f"g{index}",
                x=self.road_network.coordinates(node)[0],
                y=self.road_network.coordinates(node)[1],
                kind=TaskKind.GROUND,
                road_node=node,
            )
            for index, node in enumerate(selected_ground)
        )
        aerial_tasks = tuple(
            self._sample_aerial_task(index, ground_tasks, rng)
            for index in range(aerial_task_count)
        )
        depot_node = rng.choice(selected_ground)
        return MissionScenario(
            width_km=self.width_km,
            height_km=self.height_km,
            road_network=self.road_network,
            depot_node=depot_node,
            aerial_tasks=aerial_tasks,
            ground_tasks=ground_tasks,
            seed=seed,
        )

    def _sample_aerial_task(
        self,
        index: int,
        ground_tasks: tuple[TaskPoint, ...],
        rng: random.Random,
    ) -> TaskPoint:
        for _ in range(10_000):
            anchor = rng.choice(ground_tasks)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            # sqrt(U) produces a uniform spatial density over the disk.
            radius = self.aerial_radius_km * math.sqrt(rng.random())
            x = anchor.x + radius * math.cos(angle)
            y = anchor.y + radius * math.sin(angle)
            if self.enforce_map_bounds and not (
                0.0 <= x <= self.width_km and 0.0 <= y <= self.height_km
            ):
                continue
            return TaskPoint(
                task_id=f"a{index}",
                x=round(x, 4),
                y=round(y, 4),
                kind=TaskKind.AERIAL,
            )
        raise RuntimeError("could not sample an aerial point inside the map")

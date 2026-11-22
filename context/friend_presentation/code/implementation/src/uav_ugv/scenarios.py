"""Small deterministic and random 2D scenario generators."""

from __future__ import annotations

import random

from .domain import RechargeRequest, UGVState
from .road_network import RoadNetwork


def grid_road_network(rows: int, columns: int, spacing: float = 1.0) -> RoadNetwork:
    if rows < 1 or columns < 1 or spacing <= 0:
        raise ValueError("rows, columns, and spacing must be positive")
    coordinates: dict[int, tuple[float, float]] = {}
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns):
            node = row * columns + column
            coordinates[node] = (column * spacing, row * spacing)
            if row:
                edges.append((node - columns, node))
            if column:
                edges.append((node - 1, node))
    return RoadNetwork.from_edges(coordinates, edges)


def random_assignment_round(
    road_network: RoadNetwork,
    number_of_uavs: int,
    number_of_ugvs: int,
    seed: int,
    touchdown_window: tuple[float, float] = (0.0, 5.0),
) -> tuple[list[RechargeRequest], list[UGVState]]:
    if number_of_uavs < 1 or number_of_ugvs < 1:
        raise ValueError("at least one UAV and UGV are required")
    rng = random.Random(seed)
    nodes = list(road_network.graph.nodes)
    requests = [
        RechargeRequest(
            uav_id=index,
            landing_node=rng.choice(nodes),
            touchdown_time=rng.uniform(*touchdown_window),
            remaining_energy_fraction=rng.random() * 0.25,
        )
        for index in range(number_of_uavs)
    ]
    ugvs = [
        UGVState(
            ugv_id=index,
            current_node=rng.choice(nodes),
            ready_time=0.0,
        )
        for index in range(number_of_ugvs)
    ]
    return requests, ugvs

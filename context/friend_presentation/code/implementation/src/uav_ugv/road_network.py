"""Road graph utilities for road-constrained UGV movement."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import networkx as nx


class RoadNetwork:
    """A weighted undirected graph with Cartesian coordinates on every node."""

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph.copy()
        if not self.graph.nodes:
            raise ValueError("road network cannot be empty")
        for node, attributes in self.graph.nodes(data=True):
            if "x" not in attributes or "y" not in attributes:
                raise ValueError(f"road node {node!r} is missing x/y coordinates")
        if not nx.is_connected(self.graph):
            raise ValueError("road network must be connected")
        for left, right, attributes in self.graph.edges(data=True):
            if "distance" not in attributes:
                attributes["distance"] = self.euclidean_distance(left, right)
            if attributes["distance"] < 0:
                raise ValueError("road-edge distances must be non-negative")

    @classmethod
    def from_edges(
        cls,
        coordinates: Mapping[int, tuple[float, float]],
        edges: Iterable[tuple[int, int] | tuple[int, int, float]],
    ) -> "RoadNetwork":
        graph = nx.Graph()
        for node, (x, y) in coordinates.items():
            graph.add_node(node, x=float(x), y=float(y))
        for edge in edges:
            if len(edge) == 2:
                left, right = edge
                x1, y1 = coordinates[left]
                x2, y2 = coordinates[right]
                distance = math.hypot(x2 - x1, y2 - y1)
            else:
                left, right, distance = edge
            graph.add_edge(left, right, distance=float(distance))
        return cls(graph)

    def coordinates(self, node: int) -> tuple[float, float]:
        attributes = self.graph.nodes[node]
        return float(attributes["x"]), float(attributes["y"])

    def euclidean_distance(self, start: int, end: int) -> float:
        x1, y1 = self.coordinates(start)
        x2, y2 = self.coordinates(end)
        return math.hypot(x2 - x1, y2 - y1)

    def road_distance(self, start: int, end: int) -> float:
        return float(
            nx.shortest_path_length(
                self.graph,
                source=start,
                target=end,
                weight="distance",
            )
        )

    def travel_time(self, start: int, end: int, speed: float) -> float:
        if speed <= 0:
            raise ValueError("speed must be positive")
        return self.road_distance(start, end) / speed

    def shortest_path(self, start: int, end: int) -> list[int]:
        return list(
            nx.shortest_path(
                self.graph,
                source=start,
                target=end,
                weight="distance",
            )
        )

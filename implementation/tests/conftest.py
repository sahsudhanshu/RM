import numpy as np
import pytest

from edr.roads import RoadMap
from edr.scenario import Scenario


@pytest.fixture
def square_case():
    """Hand-checkable instance (3 tasks): road nodes 0:(0,0) 1:(2.7,0) 2:(0,2.7) 3:(2.7,2.7).
    Direct road 0-1 is 2.7 km; the detour 0-2-3-1 is 8.1 km. Ground tasks 0 (depot, node 0)
    and 1 (node 1); aerial task 2 at (1.35, 3.0)."""
    roads = RoadMap(
        np.array([[0, 0], [2.7, 0], [0, 2.7], [2.7, 2.7]], float),
        [(0, 1), (0, 2), (2, 3), (1, 3)], name="square",
    )
    xy = np.array([[0, 0], [2.7, 0], [1.35, 3.0]], float)
    return Scenario(roads=roads, xy=xy, n_ground=2, node=np.array([0, 1, -1]),
                    depot_task=0, seed=0)

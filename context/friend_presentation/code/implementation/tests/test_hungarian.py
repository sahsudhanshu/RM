import itertools

import numpy as np

from uav_ugv.sinkhorn import rectangular_hungarian


def test_rectangular_hungarian_matches_brute_force_optimum() -> None:
    cost = np.array(
        [
            [4.0, 1.0, 3.0, 7.0],
            [2.0, 0.0, 5.0, 8.0],
            [3.0, 2.0, 2.0, 4.0],
        ]
    )
    rows, columns = rectangular_hungarian(cost)
    obtained = cost[rows, columns].sum()
    expected = min(
        sum(cost[row, column] for row, column in enumerate(candidate))
        for candidate in itertools.permutations(range(cost.shape[1]), cost.shape[0])
    )
    assert obtained == expected
    assert len(set(columns.tolist())) == cost.shape[0]

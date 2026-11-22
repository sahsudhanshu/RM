import numpy as np

from uav_ugv.domain import RechargeRequest, UGVState
from uav_ugv.road_network import RoadNetwork
from uav_ugv.virtual_slots import build_slot_matrices, build_virtual_slots


def line_network() -> RoadNetwork:
    return RoadNetwork.from_edges(
        coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
        edges=[(0, 1), (1, 2)],
    )


def test_each_ugv_receives_enough_virtual_slots() -> None:
    ugvs = [UGVState(10, 0), UGVState(20, 2)]
    slots = build_virtual_slots(ugvs, request_count=3)
    assert len(slots) == 6
    assert [(slot.ugv_id, slot.position) for slot in slots] == [
        (10, 1),
        (10, 2),
        (10, 3),
        (20, 1),
        (20, 2),
        (20, 3),
    ]


def test_queue_delay_increases_by_one_recharge_duration_per_slot() -> None:
    request = RechargeRequest(0, landing_node=1, touchdown_time=0.0)
    ugv = UGVState(0, current_node=0, ready_time=0.0)
    matrices = build_slot_matrices(
        [request],
        [ugv],
        line_network(),
        ugv_speed=1.0,
        recharge_time=5.0,
        slots_per_ugv=3,
    )
    np.testing.assert_allclose(matrices.predicted_wait[0], [1.0, 6.0, 11.0])

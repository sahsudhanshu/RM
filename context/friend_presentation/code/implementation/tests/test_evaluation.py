from uav_ugv.evaluation import run_paired_benchmark
from uav_ugv.policies import NearestEuclideanPolicy, QueueAwareHungarianPolicy
from uav_ugv.scenarios import grid_road_network


def test_paired_benchmark_aggregates_each_policy() -> None:
    summary = run_paired_benchmark(
        road_network=grid_road_network(3, 3),
        policies=[NearestEuclideanPolicy(), QueueAwareHungarianPolicy()],
        number_of_instances=3,
        number_of_uavs=2,
        number_of_ugvs=2,
        ugv_speed=1.0,
        recharge_time=5.0,
    )
    assert set(summary) == {"nearest_euclidean", "queue_aware_hungarian"}
    assert summary["nearest_euclidean"]["makespan"]["mean"] > 0

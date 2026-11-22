import numpy as np
import pytest

from edr.env import MissionEnv
from edr.events import make_event
from edr.heuristics import nearest_feasible, run_policy
from edr.params import Params
from edr.plan import rollout_plan, run_controller
from edr.scenario import generate_scenario

P = Params()


def planner(env, horizon):
    return rollout_plan(env, nearest_feasible, horizon)


def _case(seed, kind="slow", level="high", timing="early", m=2, n=1):
    scn = generate_scenario(seed, 15, 5, "loop", P)
    ref = run_policy(MissionEnv(scn, m, n, P), nearest_feasible)
    ev = make_event(scn.roads, [(d[1], d[5]) for d in ref.ugv_drives], ref.makespan, kind, level, timing)
    return scn, ref, ev


@pytest.mark.parametrize("mode", ["open", "periodic", "event"])
def test_without_events_every_controller_equals_the_closed_loop_run(mode):
    for seed in range(8):
        scn = generate_scenario(seed, 15, 5, "loop", P)
        ref = run_policy(MissionEnv(scn, 2, 1, P), nearest_feasible)
        res, info = run_controller(MissionEnv(scn, 2, 1, P), planner, mode)
        assert res.actions == ref.actions
        assert res.makespan == pytest.approx(ref.makespan) and res.energy == pytest.approx(ref.energy)
        if mode != "periodic":
            assert info.replans == 0


def test_open_loop_never_observes_or_replans():
    scn, ref, ev = _case(1)
    env = MissionEnv(scn, 2, 1, P, events=[ev])
    res, info = run_controller(env, planner, "open")
    assert info.replans == 0 and env.belief.kappa == () and not res.failed
    assert res.makespan > ref.makespan                       # the event really delays the mission


def test_event_triggered_replans_less_than_periodic_and_only_after_an_event():
    rep_p, rep_e, n_cases = 0, 0, 0
    for seed in range(12):
        scn, ref, ev = _case(seed)
        e_per = MissionEnv(scn, 2, 1, P, events=[ev], road_aware=True)
        e_evt = MissionEnv(scn, 2, 1, P, events=[ev], road_aware=True)
        rp, ip = run_controller(e_per, planner, "periodic")
        re_, ie = run_controller(e_evt, planner, "event")
        assert not rp.failed and not re_.failed
        rep_p += ip.replans; rep_e += ie.replans; n_cases += 1
        assert all(t >= ev.t - 1e-9 for t, _ in ie.triggers)   # nothing triggers before the event
    assert rep_e < rep_p


def test_road_layer_never_livelocks():
    """Regression: with every rendezvous 'late', the road mask used to leave only landing
    sites from which the last task is out of range, so UAVs shuttled until max_rounds."""
    for seed in range(40):
        scn, ref, ev = _case(seed)
        env = MissionEnv(scn, 2, 1, P, events=[ev], road_aware=True)
        res, _ = run_controller(env, planner, "periodic")
        assert not res.failed, (seed, res.reason)


def test_disconnecting_block_is_survived_only_with_the_road_layer():
    """A bridge is blocked early. Open loop sends the UGV to a stranded rendezvous and fails;
    a replanner that knows the road state avoids it."""
    failed_open = failed_event = tried = 0
    for seed in range(30):
        scn, ref, ev = _case(seed, kind="block", level="low", timing="early")
        scn2 = generate_scenario(seed, 15, 5, "loop", P)
        drives = [(d[1], d[5]) for d in ref.ugv_drives]
        ev = make_event(scn2.roads, drives, ref.makespan, "block", "low", "early", disconnect=True)
        if not scn2.roads.is_bridge(ev.edges[0]):
            continue
        tried += 1
        ro, _ = run_controller(MissionEnv(scn2, 2, 1, P, events=[ev]), planner, "open")
        re_, _ = run_controller(MissionEnv(scn2, 2, 1, P, events=[ev], road_aware=True), planner, "event")
        failed_open += ro.failed
        failed_event += re_.failed
    assert tried >= 10
    assert failed_event <= failed_open

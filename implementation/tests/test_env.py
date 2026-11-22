import math

import numpy as np
import pytest

from edr.env import MissionEnv
from edr.events import RoadEvent, make_event
from edr.heuristics import nearest_feasible, run_policy
from edr.params import Params, uav_power_w
from edr.roads import NORMAL, RoadState, loop_roads, tree_roads
from edr.scenario import generate_scenario

P = Params()
LEG = 2 * math.hypot(1.35, 3.0)          # depot -> aerial -> ground node 1  (= 6.58 km)


def test_power_model_matches_paper():
    assert uav_power_w(10) == pytest.approx(198.599)
    assert uav_power_w(0) == pytest.approx(229.6)
    assert P.max_range_km == pytest.approx(14.486, abs=1e-3)   # ~24 min at 10 m/s ("~25 min")
    assert P.fly_kj_per_km == pytest.approx(19.8599, abs=1e-4)


def _fly(env, actions):
    for a in actions:
        assert env.active_uav() is not None
        env.mask()
        env.step(a)
    env.active_uav()          # lets the UGV phase / end of mission run


def test_hand_case_static_energy_equals_power_times_time(square_case):
    """E = sum P*t per sortie (M1 unit test). 1U-1G, 3 tasks, static roads."""
    env = MissionEnv(square_case, 1, 1, P)
    _fly(env, [2, 3 + 1])                      # visit aerial task 2, then recharge at ground 1
    assert env.active_uav() is None
    r = env.result()
    t_fly = LEG / P.uav_kmmin                   # minutes
    assert r.energy_fly == pytest.approx(uav_power_w(10) * t_fly * 60 / 1000, rel=1e-9)
    assert r.energy_fly == pytest.approx(LEG * 19.8599, rel=1e-6)
    s = r.services[0]
    assert s["ugv_arrival"] == pytest.approx(2.7 / P.ugv_kmmin)          # 10 min direct drive
    assert s["wait"] == pytest.approx(0.0)                               # UGV is already there
    assert r.energy_wait == pytest.approx(0.0)
    assert r.makespan == pytest.approx(t_fly + P.recharge_min)
    assert not r.failed and r.visited == 3


def test_hand_case_blocked_edge_reroute_eta_and_hover_energy(square_case):
    """Block the single edge 0-1 to the rendezvous: UGV detours 8.1 km (30 min), the UAV
    hovers for the difference and that hover time is charged at P(0)."""
    blocked = RoadEvent(t=0.0, kind="block", edges=(square_case.roads.eid(0, 1),))
    env = MissionEnv(square_case, 1, 1, P, events=[blocked])
    _fly(env, [2, 3 + 1])
    r = env.result()
    t_land = LEG / P.uav_kmmin
    s = r.services[0]
    assert s["ugv_arrival"] == pytest.approx(8.1 / P.ugv_kmmin)          # 30 min via 0-2-3-1
    assert s["wait"] == pytest.approx(30.0 - t_land)
    assert r.energy_wait == pytest.approx((30.0 - t_land) * uav_power_w(0) * 60 / 1000)
    assert [d[3:5] for d in r.ugv_drives] == [(0, 2), (2, 3), (3, 1)]    # rerouted path
    assert r.makespan == pytest.approx(30.0 + P.recharge_min)


def test_waiting_model_hover_vs_landed(square_case):
    blocked = RoadEvent(0.0, "block", (square_case.roads.eid(0, 1),))
    hover = MissionEnv(square_case, 1, 1, P, events=[blocked]); _fly(hover, [2, 4])
    landed = MissionEnv(square_case, 1, 1, Params(wait_mode="landed"), events=[blocked]); _fly(landed, [2, 4])
    rh, rl = hover.result(), landed.result()
    assert rh.wait_time == pytest.approx(rl.wait_time)           # same delay either way
    assert rh.wait_time > 0
    assert rh.energy_wait > 0 and rl.energy_wait == 0.0           # but only hovering burns energy


def test_road_aware_mask_prefers_reachable_on_time_rendezvous(square_case):
    """With the edge blocked and belief updated, rendezvous 1 would wait 19 min. A strict
    tolerance masks it and keeps the depot (UGV is there, wait 0)."""
    blocked = RoadEvent(0.0, "block", (square_case.roads.eid(0, 1),))
    strict = Params(wait_tol_min=15.0)
    env = MissionEnv(square_case, 1, 1, strict, events=[blocked], road_aware=True)
    env.observe(0.0)
    env.active_uav()
    env.step(2)                                  # visit the aerial task
    m = env.mask()
    assert m[3 + 0] and not m[3 + 1]
    # base (not road aware) policy cannot see it:
    base = MissionEnv(square_case, 1, 1, strict, events=[blocked], road_aware=False)
    base.active_uav(); base.step(2)
    assert base.mask()[3 + 1]
    # loose tolerance keeps both
    loose = MissionEnv(square_case, 1, 1, Params(wait_tol_min=60.0), events=[blocked], road_aware=True)
    loose.observe(0.0); loose.active_uav(); loose.step(2)
    assert loose.mask()[3:].all()


def test_projected_wait_matches_executed_wait(square_case):
    blocked = RoadEvent(0.0, "block", (square_case.roads.eid(0, 1),))
    env = MissionEnv(square_case, 1, 1, P, events=[blocked], road_aware=True)
    env.observe(0.0); env.active_uav(); env.step(2)
    pred = env.projected_wait(0, 1)
    env.step(3 + 1)
    env.active_uav()
    assert env.result().services[0]["wait"] == pytest.approx(pred)


def test_unobserved_event_does_not_change_the_belief(square_case):
    blocked = RoadEvent(50.0, "block", (square_case.roads.eid(0, 1),))
    env = MissionEnv(square_case, 1, 1, P, events=[blocked])
    assert env.observe(10.0) is False and env.belief == NORMAL
    assert env.observe(50.0) is True and env.belief.blocked


def test_fuel_mask_excludes_unreturnable_task(square_case):
    far = np.vstack([square_case.xy, [[1.35, 17.0]]])
    scn = type(square_case)(roads=square_case.roads, xy=far, n_ground=2,
                            node=np.array([0, 1, -1, -1]), depot_task=0, seed=0)
    env = MissionEnv(scn, 1, 1, P)
    env.active_uav()
    m = env.mask()
    assert m[2] and not m[3]                       # 17 km away: cannot return on one battery


def test_one_ugv_serves_one_uav_at_a_time(square_case):
    """Two UAVs land at the same ground node; the single UGV serves them back-to-back."""
    env = MissionEnv(square_case, 2, 1, P)
    # UAV 0 visits the aerial task and lands at ground 1; UAV 1 then has nothing left but
    # task 1 (visited by landing) -> only one sortie is needed, so build two sorties manually.
    scn = square_case
    xy = np.vstack([scn.xy, [[1.0, 3.5]]])
    scn2 = type(scn)(roads=scn.roads, xy=xy, n_ground=2, node=np.array([0, 1, -1, -1]), depot_task=0, seed=0)
    env = MissionEnv(scn2, 2, 1, P)
    env.active_uav(); env.mask(); env.step(2); env.mask(); env.step(3 + 1)
    env.active_uav(); env.mask(); env.step(3); env.mask(); env.step(3 + 1)
    env.active_uav()
    r = env.result()
    s0, s1 = sorted(r.services, key=lambda s: s["start"])
    assert s1["start"] >= s0["end"] - 1e-9
    assert s1["wait"] > 0


def test_blocked_bridge_without_detour_fails_cleanly():
    roads = tree_roads()
    scn = generate_scenario(3, 15, 5, "tree")
    # block every edge: the UGV cannot reach any rendezvous away from its own node
    ev = RoadEvent(0.0, "block", tuple(range(len(roads.edges))))
    env = MissionEnv(scn, 2, 1, P, events=[ev])
    r = run_policy(env, nearest_feasible)
    assert r.failed and "cannot reach" in r.reason
    assert r.cost == pytest.approx(r.makespan + P.penalty)


def test_all_tasks_visited_and_energy_bookkeeping():
    for seed in range(10):
        scn = generate_scenario(seed, 15, 5, "loop")
        for m, n in [(1, 1), (2, 1), (2, 2), (4, 2)]:
            env = MissionEnv(scn, m, n, P)
            r = run_policy(env, nearest_feasible)
            assert not r.failed and r.visited == r.n_tasks
            assert r.energy == pytest.approx(r.energy_fly + r.energy_wait)
            assert r.energy_fly == pytest.approx(sum(l[6] for l in r.legs))
            assert r.energy_wait == pytest.approx(r.wait_time * P.hover_kj_per_min)
            # a sortie never exceeds the battery
            per_sortie = {}
            for l in r.legs:
                per_sortie[(l[0], l[7])] = per_sortie.get((l[0], l[7]), 0.0) + l[6]
            assert max(per_sortie.values()) <= P.fuel_kj + 1e-6


def test_clone_is_independent_and_deterministic():
    scn = generate_scenario(5, 15, 5, "loop")
    env = MissionEnv(scn, 2, 1, P)
    for _ in range(6):
        if env.active_uav() is None:
            break
        env.step(int(nearest_feasible(env, env.mask())))
    twin = env.clone()
    a = run_policy(env, nearest_feasible)
    b = run_policy(twin, nearest_feasible)
    assert a.makespan == b.makespan and a.energy == b.energy and a.actions == b.actions


def test_greedy_assignment_euclid_vs_road(square_case):
    """UGV 0 at node 0, UGV 1 at node 1; request at node 3 (2.7,2.7): Euclid says UGV 1
    (2.7 away) -- if the road 1-3 is blocked, road distance says UGV 0 (5.4)."""
    ev = RoadEvent(0.0, "block", (square_case.roads.eid(1, 3),))
    from edr.env import Request
    for metric, expected in [("euclid", 1), ("road", 0)]:
        env = MissionEnv(square_case, 1, 2, P, events=[ev], assign_metric=metric)
        env.observe(0.0)
        env.ugv_node[:] = [0, 1]
        groups = env._assign([Request(0, 1, 3, 0.0)], env.ugv_node, env.belief)
        assert [n for n, q in groups.items() if q] == [expected]


def test_make_event_block_keeps_map_connected():
    roads = loop_roads()
    scn = generate_scenario(2, 15, 5, "loop")
    env = MissionEnv(scn, 2, 1, P)
    r = run_policy(env, nearest_feasible)
    visits = [(d[1], d[5]) for d in r.ugv_drives]
    for level in ("low", "medium", "high"):
        for timing in ("early", "middle", "late"):
            ev = make_event(roads, visits, r.makespan, "block", level, timing)
            assert roads.is_connected(RoadState(frozenset(ev.edges)))
            assert ev.t == pytest.approx({"early": .15, "middle": .5, "late": .8}[timing] * r.makespan)


def test_diversion_rescues_a_stranded_rendezvous():
    """Blocked bridge strands rendezvous node 1; with diversion enabled the hovering UAV flies on
    its remaining battery to the depot, where the UGV is, instead of failing."""
    import numpy as np
    from edr.roads import RoadMap
    from edr.scenario import Scenario
    roads = RoadMap(np.array([[0, 0], [2.7, 0], [5.4, 0]], float), [(0, 1), (1, 2)], name="line")
    xy = np.array([[0, 0], [2.7, 0], [1.35, 3.0]], float)
    scn = Scenario(roads=roads, xy=xy, n_ground=2, node=np.array([0, 1, -1]), depot_task=0, seed=0)
    cut = RoadEvent(t=0.0, kind="block", edges=(roads.eid(0, 1),))
    plain = MissionEnv(scn, 1, 1, P, events=[cut])
    _fly(plain, [2, 3 + 1])
    assert plain.result().failed
    rescued = MissionEnv(scn, 1, 1, P, events=[cut]); rescued.divert = True
    _fly(rescued, [2, 3 + 1])
    r = rescued.result()
    assert not r.failed and r.visited == 3
    assert r.services[0]["node"] == 0                                    # re-targeted to the depot
    extra_flight = math.hypot(2.7, 0.0) * P.fly_kj_per_km                # node 1 -> depot leg
    assert r.energy_fly == pytest.approx(LEG * P.fly_kj_per_km + extra_flight)


def test_paper_map_matches_the_authors_point_list():
    from edr.roads import PAPER_POINTS, paper_roads
    rm = paper_roads()
    assert rm.n == 28 == len(PAPER_POINTS) and len(rm.edges) == 27            # tree
    assert tuple(map(tuple, rm.xy[list(rm.mandatory)])) == ((3.0, 8.0), (10.0, 12.0), (10.0, 5.0))
    scn = generate_scenario(5, 15, 5, "paper")
    assert set(rm.mandatory) <= set(scn.node[: scn.n_ground].tolist())        # endpoints always present
    assert scn.n_ground == 5 and scn.n_aerial == 15


def test_border_map_has_loops_outposts_and_a_surveillance_strip():
    from edr.roads import border_roads
    rm = border_roads()
    assert rm.is_connected(NORMAL)
    assert len(rm.edges) - rm.n + 1 >= 3                                     # independent cycles -> detours exist
    scn = generate_scenario(3, 15, 5, "border")
    assert set(rm.mandatory) <= set(scn.node[: scn.n_ground].tolist())
    (x0, y0), (x1, y1), half = rm.band
    # every aerial task lies inside the strip along the border
    d = np.array([x1 - x0, y1 - y0]); d /= np.linalg.norm(d)
    rel = scn.xy[scn.n_ground:] - np.array([x0, y0])
    perp = np.abs(rel[:, 0] * -d[1] + rel[:, 1] * d[0])
    assert (perp <= half + 1e-9).all()
    env = MissionEnv(scn, 2, 1, P)
    res = run_policy(env, nearest_feasible)
    assert not res.failed and res.visited == res.n_tasks


def test_just_in_time_departure_removes_predicted_hover_energy_only(square_case):
    """Known delay (event observed): the UAV waits on the ground, no hover energy, same mission time.
    Unknown delay: nothing is predicted, the UAV hovers as before."""
    blocked = RoadEvent(t=0.0, kind="block", edges=(square_case.roads.eid(0, 1),))
    base = MissionEnv(square_case, 1, 1, P, events=[blocked]); base.observe(0.0); _fly(base, [2, 4]); rb = base.result()
    jit = MissionEnv(square_case, 1, 1, P, events=[blocked]); jit.jit = True; jit.observe(0.0); _fly(jit, [2, 4]); rj = jit.result()
    assert rj.makespan == pytest.approx(rb.makespan) and rj.wait_time == pytest.approx(rb.wait_time)
    assert rb.energy_wait > 0 and rj.energy_wait == pytest.approx(0.0, abs=1e-9)
    assert rj.energy == pytest.approx(rb.energy_fly)
    blind = MissionEnv(square_case, 1, 1, P, events=[blocked]); blind.jit = True; _fly(blind, [2, 4])   # belief not updated
    assert blind.result().energy_wait == pytest.approx(rb.energy_wait)

import numpy as np
import pytest
import torch

from edr.env import MissionEnv, MissionResult
from edr.heuristics import nearest_feasible, run_policy
from edr.model import RouteModel
from edr.params import Params
from edr.rl import CostCfg, NeuralPolicy, Setting, new_env, rollout, shaped_cost
from edr.scenario import generate_scenario

S = Setting(n_uav=2, n_ugv=1)


def _scns(n, seed0=0):
    return [generate_scenario(seed0 + k, 15, 5, "loop") for k in range(n)]


def test_teacher_mode_replays_the_heuristic_exactly():
    torch.manual_seed(0)
    model = RouteModel()
    scns = _scns(6)
    envs = [new_env(S, sc) for sc in scns]
    rollout(model, envs, "teacher")
    for sc, env in zip(scns, envs):
        ref = run_policy(new_env(S, sc), nearest_feasible)
        assert env.result().actions == ref.actions


def test_sampled_rollout_is_valid_and_differentiable():
    torch.manual_seed(1)
    model = RouteModel().train()
    envs = [new_env(S, sc) for sc in _scns(8)]
    logp, count = rollout(model, envs, "sample", grad=True)
    assert torch.isfinite(logp).all() and (count > 0).all() and (logp <= 0).all()
    (-logp.mean()).backward()
    assert sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None) > 0
    for e in envs:
        r = e.result()
        assert r.failed or r.visited == r.n_tasks


def test_neural_policy_matches_batched_greedy_rollout():
    torch.manual_seed(2)
    model = RouteModel().eval()
    scns = _scns(4, 50)
    envs = [new_env(S, sc) for sc in scns]
    rollout(model, envs, "greedy")
    pol = NeuralPolicy(model, greedy=True)
    for sc, env in zip(scns, envs):
        single = run_policy(new_env(S, sc), pol)
        assert single.actions == env.result().actions


def test_widen_for_budget_preserves_outputs():
    torch.manual_seed(3)
    model = RouteModel().eval()
    env = new_env(S, _scns(1, 90)[0])
    env.active_uav()
    mask = torch.from_numpy(env.mask())[None]
    args = (torch.from_numpy(env.visited.astype(np.float32))[None], torch.tensor([env.cur[0]]),
            torch.tensor([1.0]), torch.tensor([0.3]), mask, env.G)
    emb = model.encode(torch.from_numpy(env.scn.features())[None])
    dyn = torch.from_numpy(env.dyn_features())[None]
    before = model.step_logprobs(emb, args[0], args[1], args[2], args[3], args[4], args[5], dyn=dyn)
    model.widen_for_budget()
    after = model.step_logprobs(emb, args[0], args[1], args[2], args[3], args[4], args[5], dyn=dyn)
    finite = torch.isfinite(before)
    assert torch.allclose(before[finite], after[finite], atol=1e-5)


def test_budget_input_changes_the_decision_once_trained_weights_are_nonzero():
    model = RouteModel().eval()
    model.widen_for_budget()
    with torch.no_grad():
        model.wc.weight[:, -1] = torch.randn(model.d)
    scns = _scns(3, 120)
    outs = []
    for budget in (50_000.0, 500.0):          # plenty left vs almost spent
        envs = [new_env(S, sc) for sc in scns]
        for e in envs:
            e.budget = budget
        rollout(model, envs, "greedy")
        outs.append([e.result().actions for e in envs])
    assert outs[0] != outs[1]


def test_shaped_cost_variants():
    res = MissionResult(makespan=200.0, energy=1200.0, energy_fly=1000.0, energy_wait=200.0, wait_time=10.0,
                        failed=False, reason=None, cost=200.0, rounds=3, sorties=5, visited=20, n_tasks=20)
    b0 = 1000.0
    assert shaped_cost(res, CostCfg("base"), b0, 0.9) == 200.0
    assert shaped_cost(res, CostCfg("fixed", t_ref=100.0, alpha=0.5), b0, 0.9) == pytest.approx(200 + 0.5 * 100 * 1.2)
    # over budget by (1200 - 900)/1000 = 0.3 of B0
    assert shaped_cost(res, CostCfg("lagr", t_ref=100.0, lam=2.0), b0, 0.9) == pytest.approx(200 + 2.0 * 100 * 0.3)
    assert shaped_cost(res, CostCfg("hinge", t_ref=100.0, kappa=3.0), b0, 0.9) == pytest.approx(200 + 3.0 * 100 * 0.3)
    assert shaped_cost(res, CostCfg("lagr", t_ref=100.0, lam=2.0), b0, 1.0) == pytest.approx(200 + 2.0 * 100 * 0.2)
    assert shaped_cost(res, CostCfg("lagr", t_ref=100.0, lam=2.0), b0, 1.3) == pytest.approx(200.0)   # within budget


def test_neural_policy_cache_survives_garbage_collected_scenarios():
    """Regression: the encoder cache was keyed by id(scenario); freed scenarios could reuse an id
    and silently feed stale embeddings to the next instance."""
    torch.manual_seed(4)
    model = RouteModel().eval()
    pol = NeuralPolicy(model, greedy=True)
    for k in range(12):
        scn = generate_scenario(300 + k, 15, 5, "loop")
        fresh = run_policy(new_env(S, scn), NeuralPolicy(model, greedy=True))
        reused = run_policy(new_env(S, scn), pol)              # same policy object, scenarios dropped each loop
        assert fresh.actions == reused.actions


def test_pace_input_is_output_neutral_at_init_and_changes_decisions_once_trained():
    torch.manual_seed(5)
    model = RouteModel().eval()
    env = new_env(S, _scns(1, 400)[0]); env.active_uav()
    args = (torch.from_numpy(env.visited.astype(np.float32))[None], torch.tensor([env.cur[0]]), torch.tensor([1.0]),
            torch.tensor([0.4]), torch.from_numpy(env.mask())[None], env.G)
    emb = model.encode(torch.from_numpy(env.scn.features())[None]); dyn = torch.from_numpy(env.dyn_features())[None]
    before = model.step_logprobs(emb, *args, dyn=dyn)
    model.add_pace_input()
    after = model.step_logprobs(emb, *args, dyn=dyn)
    ok = torch.isfinite(before)
    assert torch.allclose(before[ok], after[ok], atol=1e-6)
    with torch.no_grad():
        model.wb.weight.normal_(0, 1.0)
    lo = model.step_logprobs(emb, args[0], args[1], args[2], torch.tensor([-0.5]), args[4], args[5], dyn=dyn)
    hi = model.step_logprobs(emb, args[0], args[1], args[2], torch.tensor([0.9]), args[4], args[5], dyn=dyn)
    assert not torch.allclose(lo[ok], hi[ok])


def test_replay_mode_reproduces_a_recorded_route():
    torch.manual_seed(7)
    model = RouteModel().eval()
    scns = _scns(3, 800)
    envs = [new_env(S, sc) for sc in scns]
    rollout(model, envs, "sample")
    replay = [[a for (_, _, a) in e.actions] for e in envs]
    again = [new_env(S, sc) for sc in scns]
    rollout(model, again, "replay", replay=replay)
    assert [e.result().actions for e in envs] == [e.result().actions for e in again]


def test_stall_counter_and_loop_fallback():
    scn = _scns(1, 900)[0]
    env = new_env(S, scn)
    assert env.stalled_rounds == 0
    run_policy(env, nearest_feasible)
    assert env._best_visited == env.K and env.stalled_rounds <= 2        # progress was made until the end
    # integration: the looping base policy (training seed 2) completes missions once the fallback is on
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "results_kaggle", "loop", "models", "U15G5_2U1G_seed2", "p0.pt")
    if not os.path.exists(path):
        pytest.skip("trained seed-2 model not available")
    from edr.evaluate import load_model
    model = load_model(path)
    fails_plain = fails_fb = 0
    for k in range(10):
        sc = generate_scenario(2_000_000_000 + k, 15, 5, "loop")
        fails_plain += run_policy(new_env(S, sc), NeuralPolicy(model)).failed
        fails_fb += run_policy(new_env(S, sc), NeuralPolicy(model, fallback=True)).failed
    assert fails_fb == 0 and fails_plain >= fails_fb

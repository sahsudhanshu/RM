"""Batched rollouts and training for the route policy (REINFORCE + greedy-rollout baseline).

All rollouts step a list of `MissionEnv` in lock-step: one batched decoder call per decision
round, one python `env.step` per environment.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import stats

from .env import MissionEnv
from .events import LEVELS, TIMINGS, make_event
from .heuristics import nearest_feasible
from .model import RouteModel
from .params import Params
from .scenario import Scenario, generate_scenario


# ---------------------------------------------------------------------------- rollouts
def stack_feats(scns) -> torch.Tensor:
    return torch.from_numpy(np.stack([s.features() for s in scns]))


def rollout(model: RouteModel, envs: list[MissionEnv], mode: str = "greedy",
            observe: bool = False, grad: bool = False, feats: torch.Tensor | None = None, replay=None):
    """Run every env to completion with the neural policy.

    mode: 'greedy' | 'sample' | 'teacher' (follow the nearest-feasible rule, return its log-prob).
    observe: update each env's road belief at every round start (closed-loop / periodic replanning).
    Returns (sum of action log-probs per env [B], decision counts [B]). Results stay in envs.
    """
    if mode not in {"greedy", "sample", "teacher", "replay"}:
        raise ValueError(mode)
    ptr = [0] * len(envs)
    b = len(envs)
    if feats is None:
        feats = stack_feats([e.scn for e in envs])
    fuel_cap = envs[0].p.fuel_kj
    G = envs[0].G
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        emb = model.encode(feats)
        total = torch.zeros(b)
        count = torch.zeros(b)
        seen_round = [-1] * b
        active = list(range(b))
        while active:
            sel, vis, cur, fuel, bud, masks, dyns = [], [], [], [], [], [], []
            for i in active:
                env = envs[i]
                env.settle()
                if env.done:
                    continue
                if observe and env.rounds != seen_round[i]:
                    env.observe(env.round_start_time())
                    seen_round[i] = env.rounds
                u = env.active_uav()
                m = env.mask()
                if env.done:
                    continue
                sel.append(i)
                vis.append(env.visited)
                cur.append(env.cur[u])
                fuel.append(env.fuel[u] / fuel_cap)
                bud.append(env.budget_left_frac)
                masks.append(m)
                dyns.append(env.dyn_features())
            if not sel:
                break
            idx = torch.tensor(sel)
            logp = model.step_logprobs(
                emb[idx],
                torch.from_numpy(np.stack(vis).astype(np.float32)),
                torch.tensor(np.array(cur), dtype=torch.long),
                torch.tensor(np.array(fuel), dtype=torch.float32),
                torch.tensor(np.array(bud), dtype=torch.float32),
                torch.from_numpy(np.stack(masks)), G, dyn=torch.from_numpy(np.stack(dyns)),
            )
            if mode == "greedy":
                act = logp.argmax(dim=-1)
            elif mode == "sample":
                act = torch.multinomial(logp.exp(), 1).squeeze(1)
            elif mode == "replay":                       # follow a recorded action sequence (expert iteration)
                act = torch.tensor([replay[i][ptr[i]] for i in sel])
                for i in sel:
                    ptr[i] += 1
            else:
                act = torch.tensor([nearest_feasible(envs[i], masks[j]) for j, i in enumerate(sel)])
            chosen = logp.gather(1, act[:, None]).squeeze(1)
            if grad:
                total = total.index_add(0, idx, chosen)
                count = count.index_add(0, idx, torch.ones(len(sel)))
            for j, i in enumerate(sel):
                envs[i].step(int(act[j]))
            active = sel
    return total, count


class NeuralPolicy:
    """Single-environment policy callable `(env, mask) -> action` (used by planners)."""

    def __init__(self, model: RouteModel, greedy: bool = True, fallback: bool = False, patience: int = 2) -> None:
        """fallback: if no new task was visited for `patience` rounds (the policy is looping), hand the decision to the
        rule-based nearest-feasible policy for the rest of the mission."""
        self.model, self.greedy, self.fallback, self.patience = model.eval(), greedy, fallback, patience
        self._cache = None        # (scenario object, encoder embedding); the reference keeps the id unique

    def __call__(self, env: MissionEnv, mask: np.ndarray) -> int:
        if self.fallback and env.stalled_rounds >= self.patience:
            return int(nearest_feasible(env, mask))
        with torch.inference_mode():
            if self._cache is None or self._cache[0] is not env.scn:
                self._cache = (env.scn, self.model.encode(torch.from_numpy(env.scn.features())[None]))
            u = env._cur_uav
            logp = self.model.step_logprobs(
                self._cache[1],
                torch.from_numpy(env.visited.astype(np.float32))[None],
                torch.tensor([env.cur[u]]),
                torch.tensor([env.fuel[u] / env.p.fuel_kj], dtype=torch.float32),
                torch.tensor([env.budget_left_frac], dtype=torch.float32),
                torch.from_numpy(mask)[None], env.G,
                dyn=torch.from_numpy(env.dyn_features())[None],
            )[0]
            return int(logp.argmax()) if self.greedy else int(torch.multinomial(logp.exp(), 1))


# ---------------------------------------------------------------------------- cost shaping
@dataclass
class CostCfg:
    variant: str = "base"        # base | fixed | lagr | hinge
    t_ref: float = 200.0         # minutes: scale that turns normalised energy into time units
    alpha: float = 0.3           # fixed-penalty weight on E/B0            (P2)
    kappa: float = 3.0           # fixed hinge weight on (E-B)+/B0         (P4)
    lam: float = 0.0             # Lagrange multiplier on (E-B)+/B0        (P3), updated online


def shaped_cost(res, cfg: CostCfg, b0: float, beta: float) -> float:
    c = res.cost
    over = max(0.0, res.energy - beta * b0) / b0
    if cfg.variant == "base":
        return c
    if cfg.variant == "fixed":
        return c + cfg.alpha * cfg.t_ref * res.energy / b0
    if cfg.variant == "lagr":
        return c + cfg.lam * cfg.t_ref * over
    if cfg.variant == "hinge":
        return c + cfg.kappa * cfg.t_ref * over
    raise ValueError(cfg.variant)


# ---------------------------------------------------------------------------- batches
@dataclass
class Setting:
    n_aerial: int = 15
    n_ground: int = 5
    n_uav: int = 2
    n_ugv: int = 1
    roads: str = "loop"
    params: Params = field(default_factory=Params)

    @property
    def tag(self) -> str:
        base = f"U{self.n_aerial}G{self.n_ground}_{self.n_uav}U{self.n_ugv}G"
        base = base if self.roads == "loop" else f"{base}_{self.roads}"
        return base + ("_landed" if self.params.wait_mode == "landed" else "")


def new_env(s: Setting, scn: Scenario, events=(), road_aware=False, assign_metric="euclid") -> MissionEnv:
    return MissionEnv(scn, s.n_uav, s.n_ugv, s.params, events=events,
                      road_aware=road_aware, assign_metric=assign_metric)


def reference_runs(s: Setting, base: RouteModel, scns, feats=None):
    """Static-road greedy runs of the base policy P0: gives B0 (energy), T0 and the UGV drive
    log used to place road events."""
    envs = [new_env(s, scn) for scn in scns]
    rollout(base.eval(), envs, "greedy", feats=feats)
    out = []
    for scn, e in zip(scns, envs):
        r = e.result()
        if r.failed:     # B0 of a failed run is meaningless: use the rule-based reference instead
            from .heuristics import run_policy
            r = run_policy(new_env(s, scn), nearest_feasible)
        out.append(r)
    return out


def random_event(rng: np.random.Generator, scn: Scenario, ref, p_event: float = 1.0):
    if rng.random() >= p_event:
        return ()
    kind = "block" if rng.random() < 0.5 else "slow"
    level = LEVELS[rng.integers(3)]
    timing = list(TIMINGS)[rng.integers(3)]
    drives = [(d[1], d[5]) for d in ref.ugv_drives]
    return (make_event(scn.roads, drives, ref.makespan, kind, level, timing, rng=rng),)


# ---------------------------------------------------------------------------- training
@dataclass
class TrainLog:
    rows: list = field(default_factory=list)


def imitate(model: RouteModel, s: Setting, steps: int, batch: int, lr: float, seed0: int, log=print):
    """Warm start: behaviour-clone the nearest-feasible rule (cross-entropy on its actions)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for it in range(steps):
        scns = [generate_scenario(seed0 + it * batch + k, s.n_aerial, s.n_ground, s.roads, s.params)
                for k in range(batch)]
        envs = [new_env(s, scn) for scn in scns]
        total, count = rollout(model, envs, "teacher", grad=True)
        loss = -(total.sum() / count.sum())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 25 == 0 or it == steps - 1:
            log(f"[imitate] step {it:4d}  nll/decision {loss.item():.4f}")
    return model


@dataclass
class RLCfg:
    steps: int = 300
    batch: int = 64
    lr: float = 1e-4
    lr_decay: float = 0.995          # per epoch (paper)
    epoch_steps: int = 20            # steps per epoch (baseline refresh test)
    val_size: int = 64
    seed0: int = 10_000_000
    val_seed0: int = 1_000_000_000
    betas: tuple = (0.7, 0.8, 0.9, 1.0)   # budget levels sampled for 'hinge'; fixed beta for 'lagr'
    lam_lr: float = 0.1              # small dual step (deck risk table: energy penalty may be unstable)
    lam_max: float = 8.0
    lam_on_greedy: bool = False      # update lambda from the GREEDY validation violation (the deployed policy), not the sampled batch
    lam_val_lr: float = 1.0          # dual step per epoch when lam_on_greedy
    events: bool = False             # train with random road events (closed-loop, road layer on)
    p_event: float = 0.75
    assign_metric: str = "euclid"
    road_aware: bool = False
    group: int = 1                   # >1: sample `group` routes per instance, leave-one-out group baseline
    # (variance reduction used for the fine-tuning stages; group=1 is the paper's greedy-rollout baseline)


def _make_envs(s, scns, refs, evs, betas, rl: RLCfg):
    envs = []
    for scn, ref, ev, beta in zip(scns, refs, evs, betas):
        env = new_env(s, scn, events=ev, road_aware=rl.road_aware, assign_metric=rl.assign_metric)
        env.budget = beta * ref.energy
        env.divert = rl.events
        envs.append(env)
    return envs


def reinforce(model: RouteModel, base: RouteModel, s: Setting, cost: CostCfg, rl: RLCfg,
              beta_fixed: float | None = None, log=print, tag: str = "") -> TrainLog:
    """REINFORCE with a greedy-rollout baseline network (Alg. 1 of the base paper).

    `base` is the frozen P0 used only to define each instance's B0 (static roads)."""
    rng = np.random.default_rng(rl.seed0)
    if model.use_pace:                      # the freshly added pace weights start at zero: let them learn faster
        pace = list(model.wb.parameters())
        rest = [p_ for p_ in model.parameters() if all(p_ is not q for q in pace)]
        opt = torch.optim.Adam([{"params": rest, "lr": rl.lr}, {"params": pace, "lr": rl.lr * 10}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=rl.lr)
    baseline = copy.deepcopy(model).eval()
    tlog = TrainLog()
    G = s.n_ground

    def draw(seed_start, n, with_events):
        scns = [generate_scenario(seed_start + k, s.n_aerial, s.n_ground, s.roads, s.params) for k in range(n)]
        feats = stack_feats(scns)
        refs = reference_runs(s, base, scns, feats)
        evs = [random_event(rng, sc, r, rl.p_event) if with_events else () for sc, r in zip(scns, refs)]
        betas = [beta_fixed if beta_fixed is not None else float(rng.choice(rl.betas)) for _ in scns]
        return scns, feats, refs, evs, betas

    val = draw(rl.val_seed0, rl.val_size, rl.events)
    t0 = time.time()
    g = rl.group
    for it in range(rl.steps):
        n_inst = rl.batch // g
        scns, feats, refs, evs, betas = draw(rl.seed0 + it * n_inst, n_inst, rl.events)
        if g > 1:
            scns, refs, evs, betas = scns * g, refs * g, evs * g, betas * g
            feats = feats.repeat(g, 1, 1)
        model.train()
        envs = _make_envs(s, scns, refs, evs, betas, rl)
        logp, _ = rollout(model, envs, "sample", observe=rl.events, grad=True, feats=feats)
        res = [e.result() for e in envs]
        c = torch.tensor([shaped_cost(r, cost, ref.energy, bt) for r, ref, bt in zip(res, refs, betas)])
        if g > 1:
            grp = c.reshape(g, n_inst)                                   # row k = k-th sample of every instance
            cb = ((grp.sum(0, keepdim=True) - grp) / (g - 1)).reshape(-1)
            adv = c - cb
            adv = adv / (adv.std() + 1e-6)
        else:
            benvs = _make_envs(s, scns, refs, evs, betas, rl)
            rollout(baseline, benvs, "greedy", observe=rl.events, feats=feats)
            cb = torch.tensor([shaped_cost(e.result(), cost, ref.energy, bt)
                               for e, ref, bt in zip(benvs, refs, betas)])
            adv = (c - cb) / cost.t_ref
        loss = (adv * logp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        viol = np.mean([(r.energy - bt * ref.energy) / ref.energy for r, ref, bt in zip(res, refs, betas)])
        if cost.variant == "lagr" and not rl.lam_on_greedy:
            cost.lam = min(rl.lam_max, max(0.0, cost.lam + rl.lam_lr * float(viol)))     # dual ascent on (E-B)
        row = dict(step=it, T=float(np.mean([r.makespan for r in res])), E=float(np.mean([r.energy for r in res])),
                   cost=float(c.mean()), baseline_cost=float(cb.mean()), viol=float(viol),
                   fail=float(np.mean([r.failed for r in res])), lam=cost.lam, lr=opt.param_groups[0]["lr"])
        tlog.rows.append(row)
        if (it + 1) % rl.epoch_steps == 0:                              # end of "epoch"
            for gp in opt.param_groups:
                gp["lr"] *= rl.lr_decay
            model.eval()
            vs, vf, vr, ve, vb = val
            venvs = _make_envs(s, vs, vr, ve, vb, rl)
            rollout(model, venvs, "greedy", observe=rl.events, feats=vf)
            vres = [e.result() for e in venvs]
            cm = np.array([shaped_cost(r, cost, ref.energy, bt) for r, ref, bt in zip(vres, vr, vb)])
            improved = False
            if cost.variant == "lagr" and rl.lam_on_greedy:
                vg = float(np.mean([(r.energy - bt * ref.energy) / ref.energy for r, ref, bt in zip(vres, vr, vb)]))
                cost.lam = min(rl.lam_max, max(0.0, cost.lam + rl.lam_val_lr * vg))
                row['viol_greedy'] = vg
            if g == 1:                                                  # paper: baseline refresh by paired t-test
                benvs = _make_envs(s, vs, vr, ve, vb, rl)
                rollout(baseline, benvs, "greedy", observe=rl.events, feats=vf)
                cbv = np.array([shaped_cost(e.result(), cost, ref.energy, bt) for e, ref, bt in zip(benvs, vr, vb)])
                t, p_two = stats.ttest_rel(cm, cbv)
                improved = bool(t < 0 and p_two / 2 < 0.05)
                if improved:
                    baseline = copy.deepcopy(model).eval()
                row.update(val_baseline=float(cbv.mean()), baseline_updated=improved)
            row.update(val_cost=float(cm.mean()), val_T=float(np.mean([r.makespan for r in vres])),
                       val_E=float(np.mean([r.energy for r in vres])), val_fail=float(np.mean([r.failed for r in vres])))
            log(f"[{tag}] step {it+1:4d}  T {row['T']:6.1f}  E {row['E']:6.0f}  cost {row['cost']:7.1f}  "
                f"val T {row['val_T']:6.1f} E {row['val_E']:6.0f} fail {row['val_fail']:.2f}{' *' if improved else ''}  "
                f"viol {viol:+.3f}  lam {cost.lam:.2f}  [{time.time()-t0:5.0f}s]")
    model.eval()
    return tlog


def expert_iteration(model: RouteModel, base: RouteModel, s: Setting, cost: CostCfg, rl: RLCfg,
                     beta_fixed: float | None = None, log=print, tag: str = "") -> TrainLog:
    """Distil search into the policy: per instance sample `rl.group` routes, keep the one with the lowest
    shaped cost (time + budget hinge) and imitate it (cross-entropy on its actions). Inference stays greedy."""
    rng = np.random.default_rng(rl.seed0)
    if model.use_pace:
        pace = list(model.wb.parameters())
        rest = [p_ for p_ in model.parameters() if all(p_ is not q for q in pace)]
        opt = torch.optim.Adam([{"params": rest, "lr": rl.lr}, {"params": pace, "lr": rl.lr * 10}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=rl.lr)
    tlog = TrainLog()
    g = rl.group
    n_inst = max(1, rl.batch // g)

    def draw(seed_start, n):
        scns = [generate_scenario(seed_start + k, s.n_aerial, s.n_ground, s.roads, s.params) for k in range(n)]
        feats = stack_feats(scns)
        refs = reference_runs(s, base, scns, feats)
        evs = [() for _ in scns]
        betas = [beta_fixed if beta_fixed is not None else float(rng.choice(rl.betas)) for _ in scns]
        return scns, feats, refs, evs, betas

    val = draw(rl.val_seed0, rl.val_size)
    t0 = time.time()
    for it in range(rl.steps):
        scns, feats, refs, evs, betas = draw(rl.seed0 + it * n_inst, n_inst)
        S, R, V, Bt = scns * g, refs * g, evs * g, betas * g
        envs = _make_envs(s, S, R, V, Bt, rl)
        model.eval()
        rollout(model, envs, "sample", feats=feats.repeat(g, 1, 1))              # no grad: pure search
        res = [e.result() for e in envs]
        c = np.array([shaped_cost(r, cost, ref.energy, bt) for r, ref, bt in zip(res, R, Bt)]).reshape(g, n_inst)
        best = c.argmin(axis=0)                                                   # sample index of the elite per instance
        elite = [k * n_inst + i for i, k in enumerate(best)]
        replay = [[a for (_, _, a) in envs[j].actions] for j in elite]
        fresh = _make_envs(s, scns, refs, evs, betas, rl)
        model.train()
        logp, count = rollout(model, fresh, "replay", grad=True, feats=feats, replay=replay)
        loss = -(logp.sum() / count.sum())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rr = [res[j] for j in elite]
        row = dict(step=it, T=float(np.mean([r.makespan for r in rr])), E=float(np.mean([r.energy for r in rr])),
                   cost=float(c.min(axis=0).mean()), baseline_cost=float(c.mean()), nll=float(loss.item()),
                   fail=float(np.mean([r.failed for r in res])), viol=float(np.mean([(r.energy - bt * ref.energy) / ref.energy for r, ref, bt in zip(rr, refs, betas)])),
                   lam=cost.lam, lr=opt.param_groups[0]["lr"])
        tlog.rows.append(row)
        if (it + 1) % rl.epoch_steps == 0:
            for gp in opt.param_groups:
                gp["lr"] *= rl.lr_decay
            model.eval()
            vs, vf, vr, ve, vb = val
            venvs = _make_envs(s, vs, vr, ve, vb, rl)
            rollout(model, venvs, "greedy", feats=vf)
            vres = [e.result() for e in venvs]
            row.update(val_T=float(np.mean([r.makespan for r in vres])), val_E=float(np.mean([r.energy for r in vres])),
                       val_fail=float(np.mean([r.failed for r in vres])))
            log(f"[{tag}] step {it+1:4d}  elite T {row['T']:6.1f} E {row['E']:6.0f} (sample mean cost {row['baseline_cost']:6.1f}) nll {row['nll']:.3f}  "
                f"| greedy val T {row['val_T']:6.1f} E {row['val_E']:6.0f} fail {row['val_fail']:.2f}  [{time.time()-t0:5.0f}s]")
    model.eval()
    return tlog

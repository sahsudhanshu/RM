#!/usr/bin/env python
"""Train the base policy P0 and the energy-aware variants for one team configuration.

  python scripts/train.py p0  --uav 2 --ugv 1            # imitation warm start + REINFORCE
  python scripts/train.py ft  --uav 2 --ugv 1 --variant p4 [--variant p3_b80 ...]

All variants start from the same P0 checkpoint (same routing backbone) and are fine-tuned
with their own objective, so differences come from the method, not from the backbone.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from edr.evaluate import load_model, save_model
from edr.heuristics import nearest_feasible, run_policy
from edr.model import RouteModel
from edr.params import Params
from edr.rl import CostCfg, RLCfg, Setting, expert_iteration, imitate, new_env, reinforce, rollout, stack_feats
from edr.scenario import generate_scenario

MODEL_KW = dict(norm="ln")                  # see model.py: LayerNorm + dynamic geometry features

# name -> (cost variant, fixed beta, use budget input, train with events, extra)
VARIANTS = {
    "p0c":     dict(cost="base"),                                           # control: P0 + same extra steps
    "p2":      dict(cost="fixed"),                                          # P2: T + alpha*E
    "p3_b70":  dict(cost="lagr", beta=0.70),                                # P3: Lagrangian, one model per beta
    "p3_b80":  dict(cost="lagr", beta=0.80),
    "p3_b90":  dict(cost="lagr", beta=0.90),
    "p3_b95":  dict(cost="lagr", beta=0.95),
    "p3g_b100": dict(cost="lagr", beta=1.00, greedy_lambda=True),     # lambda driven by the deployed (greedy) policy
    "p3g_b95":  dict(cost="lagr", beta=0.95, greedy_lambda=True),
    "p3g_b90":  dict(cost="lagr", beta=0.90, greedy_lambda=True),
    "p3h_b90":  dict(cost="lagr", beta=0.90, greedy_lambda=True, lam_max=50.0, steps=1500),   # M2: higher cap, longer
    "p3h_b100": dict(cost="lagr", beta=1.00, greedy_lambda=True, lam_max=50.0, steps=1500),
    "p3_b100": dict(cost="lagr", beta=1.00),
    "p4":      dict(cost="hinge", budget_input=True),                       # P4: budget-conditioned
    "p4x":     dict(cost="hinge", budget_input=True, pace=True),            # P4 + pace feature + 10x lr on it
    "bon_t":   dict(cost="base", bon=True),                                       # distil best-of-N (time only)
    "bon_nb":  dict(cost="hinge", bon=True),                                      # distil best-of-N under a random budget, no budget input
    "bon_p4x": dict(cost="hinge", bon=True, budget_input=True, pace=True),        # ... with budget + pace input (candidate B)
    "p4_nb":   dict(cost="hinge"),                                          # ablation: no budget input
    "p4_ev":   dict(cost="hinge", budget_input=True, events=True),          # + trained with road events
    "p0_ev":   dict(cost="base", events=True),                              # P0 trained with road events
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["p0", "p0more", "ft"])
    ap.add_argument("--uav", type=int, default=2)
    ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--aerial", type=int, default=15)
    ap.add_argument("--ground", type=int, default=5)
    ap.add_argument("--roads", default="loop")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "models"))
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--imit-steps", type=int, default=600)
    ap.add_argument("--rl-steps", type=int, default=400)
    ap.add_argument("--ft-steps", type=int, default=400)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--bon-group", type=int, default=32)
    ap.add_argument("--bon-batch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wait-mode", default="hover", choices=["hover", "landed"], help="landed: waiting UAVs burn no energy")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    s = Setting(a.aerial, a.ground, a.uav, a.ugv, a.roads, Params(wait_mode=a.wait_mode))
    out = os.path.join(a.out, s.tag + (f"_seed{a.seed}" if a.seed else ""))
    os.makedirs(out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)

    if a.stage == "p0":
        model = RouteModel(**MODEL_KW)
        half = a.imit_steps // 2
        imitate(model, s, half, a.batch, 1e-3, 5_000_000 + a.seed * 7_000_000, log)
        imitate(model, s, a.imit_steps - half, a.batch, 3e-4, 6_000_000 + a.seed * 7_000_000, log)
        val = [generate_scenario(1_000_000_000 + k, a.aerial, a.ground, a.roads) for k in range(64)]
        tea = [run_policy(new_env(s, sc), nearest_feasible) for sc in val]
        t_tea = float(np.mean([r.makespan for r in tea]))
        for extra in range(3):          # safeguard: a clone far from its teacher (weak seed) gets more imitation
            envs = [new_env(s, sc) for sc in val]
            rollout(model.eval(), envs, "greedy")
            res = [e.result() for e in envs]
            t_gr, f_gr = float(np.mean([r.makespan for r in res])), float(np.mean([r.failed for r in res]))
            log(f"after imitation (round {extra}): greedy T {t_gr:.1f}, fail {f_gr:.2f} | teacher T {t_tea:.1f}")
            if t_gr <= 1.6 * t_tea and f_gr <= 0.05:
                break
            log("  clone too far from the teacher -> extra imitation")
            imitate(model, s, max(100, a.imit_steps // 2), a.batch, 3e-4, 8_000_000 + extra * 100_000 + a.seed * 7_000_000, log)
        save_model(model, os.path.join(out, "p0_imitation.pt"))
        # REINFORCE on mission time (base objective); B0 reference == this model's own greedy run
        base = copy.deepcopy(model).eval()
        rl = RLCfg(steps=a.rl_steps, batch=a.batch, seed0=10_000_000 + a.seed * 7_000_000)
        tlog = reinforce(model, base, s, CostCfg("base"), rl, log=log, tag="P0")
        save_model(model, os.path.join(out, "p0.pt"))
        json.dump(tlog.rows, open(os.path.join(out, "train_p0.json"), "w"))
        return

    if a.stage == "p0more":
        # continue REINFORCE on mission time from the saved P0 (M1: bring the base policy closer to the paper)
        import shutil
        shutil.copy(os.path.join(out, "p0.pt"), os.path.join(out, "p0_short.pt"))
        model = load_model(os.path.join(out, "p0.pt")).train()
        base = copy.deepcopy(model).eval()
        rl = RLCfg(steps=a.rl_steps, batch=a.batch, lr=a.ft_lr / 3, group=a.group, seed0=30_000_000 + a.seed * 7_000_000)
        tlog = reinforce(model, base, s, CostCfg("base"), rl, log=log, tag="P0more")
        save_model(model, os.path.join(out, "p0.pt"))
        json.dump(tlog.rows, open(os.path.join(out, "train_p0more.json"), "w"))
        return

    p0 = load_model(os.path.join(out, "p0.pt"))
    # reference for t_ref: mean mission time of P0 on validation instances
    val = [generate_scenario(1_000_000_000 + k, a.aerial, a.ground, a.roads) for k in range(64)]
    envs = [new_env(s, sc) for sc in val]
    rollout(p0, envs, "greedy")
    t_ref = float(np.mean([e.result().makespan for e in envs]))
    log(f"t_ref (mean P0 mission time) = {t_ref:.1f} min")
    for name in a.variant:
        cfg = VARIANTS[name]
        model = copy.deepcopy(p0)
        if cfg.get("pace"):
            model.add_pace_input()                       # replaces the raw-budget column of P4
        elif cfg.get("budget_input"):
            model.widen_for_budget()
        cost = CostCfg(cfg["cost"], t_ref=t_ref)
        rl = RLCfg(steps=cfg.get("steps", a.ft_steps), batch=a.batch, lr=a.ft_lr, group=a.group, lam_on_greedy=cfg.get("greedy_lambda", False), lam_max=cfg.get("lam_max", 8.0), seed0=20_000_000 + a.seed * 7_000_000,
                   events=cfg.get("events", False), road_aware=cfg.get("events", False))
        t0 = time.time()
        if cfg.get("bon"):
            rl.group, rl.batch = a.bon_group, a.bon_batch
            tlog = expert_iteration(model, p0, s, cost, rl, beta_fixed=cfg.get("beta"), log=log, tag=name)
        else:
            tlog = reinforce(model, p0, s, cost, rl, beta_fixed=cfg.get("beta"), log=log, tag=name)
        save_model(model, os.path.join(out, f"{name}.pt"))
        json.dump(tlog.rows, open(os.path.join(out, f"train_{name}.json"), "w"))
        log(f"{name}: done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

"""Encoder-decoder Transformer route policy (base paper Sec. IV-A, Eqs. 13-23).

Encoder: linear embedding of (x, y, b) + L=3 layers of [MHA -> skip -> norm -> FF -> skip -> norm]
(LayerNorm by default; BatchNorm as in the paper is available but was unreliable at batch size 1).
Decoder (one UAV decision): context = mean node embedding (with visit flags, Eq. 18)
  + projected [current-position embedding, fuel, (remaining budget b_t)] (Eq. 19);
  glimpse = MHA(context, nodes) (Eq. 20); logits = C * tanh(q.k / sqrt(d)) with C = 10
  and infeasible actions masked (Eqs. 21-22).

The action space is K + G: visit any task, or recharge at a ground task. Recharge keys use
the node embedding plus a learned "recharge" type embedding.

Deviation from the paper's decoder (CPU training budget is ~4 orders of magnitude smaller):
each action also receives two dynamic geometric features `dyn` (S, K+G, 2): distance from the
UAV's current position, and (recharge actions only) distance from that ground point to the
nearest unvisited task. They are mapped by a small MLP and added to the compatibility score;
without them the decoder cannot tell which landing site is near the remaining work.

`use_budget=True` is candidate B (budget-conditioned routing): b_t = (B - E_used)/B is
appended to the context input. All other variants share this exact architecture.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _Norm(nn.Module):
    """BatchNorm over (batch*nodes) as in the paper, or LayerNorm."""

    def __init__(self, d: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.norm = nn.BatchNorm1d(d) if kind == "bn" else nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "bn":
            b, k, d = x.shape
            return self.norm(x.reshape(-1, d)).reshape(b, k, d)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ff: int = 512, norm: str = "bn") -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = _Norm(d, norm)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n2 = _Norm(d, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.mha(x, x, x, need_weights=False)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class RouteModel(nn.Module):
    def __init__(self, d: int = 128, heads: int = 8, layers: int = 3, clip: float = 10.0,
                 use_budget: bool = False, norm: str = "ln", use_pace: bool = False) -> None:
        super().__init__()
        self.d, self.clip, self.use_budget, self.use_pace = d, clip, use_budget, use_pace
        # budget "pace" input (variant P4x): [remaining budget fraction, remaining budget - remaining work]
        self.wb = nn.Linear(2, d, bias=False) if use_pace else None
        if use_pace:
            nn.init.zeros_(self.wb.weight)
        self.embed = nn.Linear(3, d)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, norm=norm) for _ in range(layers)])
        # dynamic per-action geometry (see module docstring): MLP(2 -> 32 -> 1) added to the score
        self.dyn = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.wg_h = nn.Linear(d, d, bias=False)
        self.wg_d = nn.Linear(1, d, bias=False)
        self.wc = nn.Linear(d + 1 + (1 if use_budget else 0), d)
        self.glimpse = nn.MultiheadAttention(d, heads, batch_first=True)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.recharge_type = nn.Parameter(torch.zeros(d))

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        h = self.embed(feats)
        for layer in self.layers:
            h = layer(h)
        return h

    def step_logprobs(self, emb, visited, cur, fuel, budget, mask, n_ground: int, dyn=None) -> torch.Tensor:
        """Log-probabilities over the K+G actions for a batch of decisions.

        emb (S,K,d); visited (S,K) {0,1}; cur (S,) long; fuel (S,) in [0,1];
        budget (S,) remaining-budget fraction; mask (S,K+G) bool (True = feasible)."""
        s, k, d = emb.shape
        g = n_ground
        pos = emb[torch.arange(s), cur]
        ctx_in = [pos, fuel[:, None]]
        if self.use_budget:
            ctx_in.append(budget.clamp(-1.0, 1.0)[:, None])
        graph = self.wg_h(emb.mean(dim=1)) + self.wg_d(visited.mean(dim=1, keepdim=True))
        context = graph + self.wc(torch.cat(ctx_in, dim=-1))
        if self.use_pace:
            remaining = 1.0 - visited.mean(dim=1)                      # share of tasks still to visit
            b = budget.clamp(-1.0, 1.0)
            context = context + self.wb(torch.stack([b, (b - remaining).clamp(-1.0, 1.0)], dim=-1))
        context = context.unsqueeze(1)
        node_ok = mask[:, :k] | F.pad(mask[:, k:], (0, k - g), value=False)
        glimpse, _ = self.glimpse(context, emb, emb, key_padding_mask=~node_ok, need_weights=False)
        q = self.wq(glimpse)                                           # (S,1,d)
        keys = torch.cat([self.wk(emb), self.wk(emb[:, :g] + self.recharge_type)], dim=1)
        score = (q @ keys.transpose(1, 2)).squeeze(1) / math.sqrt(d)
        if dyn is not None:
            score = score + self.dyn(dyn).squeeze(-1)
        logits = self.clip * torch.tanh(score)
        logits = logits.masked_fill(~mask, float("-inf"))
        return F.log_softmax(logits, dim=-1)

    def add_pace_input(self) -> None:
        """Make a budget-blind model pace-aware without changing its output (new weights start at zero)."""
        if self.use_pace:
            return
        self.wb = nn.Linear(2, self.d, bias=False)
        nn.init.zeros_(self.wb.weight)
        self.use_pace = True

    def widen_for_budget(self) -> None:
        """Turn a budget-blind model into a budget-aware one without changing its output:
        the new context column starts at zero."""
        if self.use_budget:
            return
        old = self.wc
        new = nn.Linear(old.in_features + 1, old.out_features)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, : old.in_features] = old.weight
            new.bias.copy_(old.bias)
        self.wc, self.use_budget = new, True

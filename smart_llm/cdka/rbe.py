"""Retrieval Benefit Estimator (RBE):  B_pred = RBE([h_L || mu_K]).

Regresses the ground-truth retrieval benefit
    B_true = (Loss_p - Loss_r) / (|Loss_p| + eps)
from the pooled LLM hidden state concatenated with the retrieved centroid mu_K.
This is what lets the router estimate retrieval usefulness *without* running RAG
(no double inference): mu_K is a cheap embedding-space mean, h_L is already
computed on the no-retrieval path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from .nn_utils import (build_mlp, minibatches, standardize_apply,
                       standardize_fit, to_tensor)


class RetrievalBenefitEstimator(nn.Module):
    def __init__(self, in_dim: int, cfg: Config):
        super().__init__()
        self.net = build_mlp(in_dim, list(cfg.rbe.hidden_dims), 1, cfg.rbe.dropout)
        # input standardisation stats (filled at fit time)
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) / self.sd
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, x) -> np.ndarray:
        self.eval()
        dev = next(self.parameters()).device
        return self.forward(to_tensor(x, dev)).cpu().numpy().astype(np.float32)


@dataclass
class RBEData:
    x: np.ndarray      # [n, in_dim]  == concat(h_L, mu_K)
    b_true: np.ndarray  # [n]


def _loss_fn(cfg: Config):
    if cfg.rbe.loss == "mse":
        return nn.MSELoss()
    return nn.HuberLoss(delta=1.0)


def fit_rbe(train: RBEData, val: RBEData, cfg: Config,
            device: str = "cpu") -> Tuple[RetrievalBenefitEstimator, dict]:
    in_dim = train.x.shape[1]
    model = RetrievalBenefitEstimator(in_dim, cfg).to(device)

    mu, sd = standardize_fit(train.x)
    model.mu.copy_(torch.as_tensor(mu[0]))
    model.sd.copy_(torch.as_tensor(sd[0]))

    clip = cfg.rbe.target_clip
    yt_raw = np.clip(train.b_true, -clip, clip)
    xt = to_tensor(train.x, device)
    yt = to_tensor(yt_raw, device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.rbe.lr,
                            weight_decay=cfg.rbe.weight_decay)
    loss_fn = _loss_fn(cfg)

    history = {"train_loss": [], "val_r2": []}
    for epoch in range(cfg.rbe.epochs):
        model.train()
        ep = 0.0
        for bidx in minibatches(len(train.b_true), cfg.rbe.batch_size,
                                cfg.seed + epoch):
            b = torch.as_tensor(bidx, device=device)
            pred = model(xt[b])
            loss = loss_fn(pred, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            ep += float(loss.item()) * len(bidx)
        history["train_loss"].append(ep / len(train.b_true))
        history["val_r2"].append(_r2(model.predict(val.x), val.b_true))
    return model, history


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class TimeSymmetricGradientSmoother:
    """A minimal controlled model 'retrocausal' mechanism: time-symmetric smoothing of gradients.

    This is a controlled sandbox:
    - Run a short trajectory producing gradients g_t
    - Apply symmetric smoothing: g'_t = sum_s K(t,s) g_s
    - Update parameters with g'_t

    It is *not* a claim about physics. It is a test harness for
    time-coupled learning signals under loop dynamics.
    """
    horizon: int = 8
    sigma: float = 2.0
    lr: float = 1e-3
    device: str = "cpu"

    def step(self, model: nn.Module, loss_fn) -> Dict[str, float]:
        model = model.to(self.device)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]

        grads_time: List[List[torch.Tensor]] = []
        losses = []
        x = torch.randn(64, model.fc1.in_features, device=self.device)

        for _ in range(self.horizon):
            loss = loss_fn(model, x)
            losses.append(loss.item())
            g = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
            grads_time.append([gi.detach() for gi in g])
            # propagate state
            with torch.no_grad():
                x = model(x)

        # Build smoothing kernel
        T = len(grads_time)
        idx = torch.arange(T, device=self.device).float()
        K = torch.exp(-0.5 * (idx[:,None]-idx[None,:])**2 / (self.sigma**2 + 1e-12))
        K = K / (K.sum(dim=1, keepdim=True) + 1e-12)

        # Smoothed grads per time
        smoothed = []
        for p_i in range(len(params)):
            G = torch.stack([grads_time[t][p_i] for t in range(T)], dim=0)  # (T, ...)
            # Weighted sum across time
            # reshape to (T, -1) for matmul, then unreshape
            shp = G.shape
            Gf = G.reshape(T, -1)
            Gs = (K @ Gf).reshape(shp)
            smoothed.append(Gs)

        # Apply update using the final smoothed gradient (last timestep) as a canonical choice
        with torch.no_grad():
            for p, Gs in zip(params, smoothed):
                g_final = Gs[-1]
                p.add_(-self.lr * g_final)

        return {"ts_loss_mean": float(np.mean(losses)), "ts_loss_last": float(losses[-1])}

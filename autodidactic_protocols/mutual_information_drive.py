from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

def gaussian_mi_proxy(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """A stable mutual-information proxy for continuous variables.

    Uses log det covariances (Gaussian assumption) as a *proxy*:
      I(X;Y) ≈ 0.5 log |Σx| |Σy| / |Σxy|

    This is not 'the' MI; it is a measurable driver that works well for minimal testbed experiments.
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    Cx = (x.T @ x) / (x.shape[0] - 1 + eps) + eps * torch.eye(x.shape[1], device=x.device)
    Cy = (y.T @ y) / (y.shape[0] - 1 + eps) + eps * torch.eye(y.shape[1], device=y.device)
    xy = torch.cat([x, y], dim=1)
    Cxy = (xy.T @ xy) / (xy.shape[0] - 1 + eps) + eps * torch.eye(xy.shape[1], device=x.device)
    # logdet is stable for SPD-ish matrices with eps jitter
    mi = 0.5 * (torch.logdet(Cx) + torch.logdet(Cy) - torch.logdet(Cxy))
    return mi

@dataclass
class MutualInformationLearner:
    """Drive the system to maximize MI between input and internal state.

    Operationalization:
    - We treat the final hidden state as 'internal representation' H.
    - We maximize MI(X;H) (proxy) while also maintaining self-consistency (small reconstruction loss).

    This creates a useful tension: compress vs preserve information, which can create phase-like behavior.
    """
    lr: float = 1e-3
    steps: int = 100
    noise_std: float = 0.15
    recon_weight: float = 1.0
    mi_weight: float = 0.2
    device: str = "cpu"

    def update(self, model: nn.Module, batch_size: int = 64) -> Dict[str, float]:
        model.to(self.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        losses, mis = [], []
        for _ in range(self.steps):
            x = torch.randn(batch_size, model.fc1.in_features, device=self.device)
            x = x + self.noise_std * torch.randn_like(x)
            y, stats = model(x, steps=4) if callable(getattr(model, "forward", None)) else (model(x), {})
            # If model returns x only, treat output as representation too
            if isinstance(y, tuple):
                y = y[0]
            h = stats.get("h_final", None)
            if h is None:
                # fallback: use output as internal state
                h = y
            recon = torch.mean((y - x) ** 2)
            mi = gaussian_mi_proxy(x, h)
            loss = self.recon_weight * recon - self.mi_weight * mi  # maximize MI => subtract
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
            mis.append(mi.item())

        return {"mi_objective": float(np.mean(losses)), "mi_proxy": float(np.mean(mis))}

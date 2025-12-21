from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TemporalPreferenceScore:
    """Minimal testbed metric: does the system prefer policies that keep future loss low?

    We implement a short-horizon rollout:
    - apply model repeatedly to its own outputs
    - compute discounted loss across steps
    Lower discounted loss => higher score.
    """
    horizon: int = 8
    discount: float = 0.9
    batch_size: int = 64
    device: str = "cpu"

    def score(self, model: nn.Module) -> Dict[str, float]:
        model = model.to(self.device)
        model.eval()
        x0 = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
        x = x0
        total = 0.0
        disc = 1.0
        with torch.no_grad():
            for t in range(self.horizon):
                y = model(x)
                loss = torch.mean((y - x) ** 2).item()
                total += disc * loss
                disc *= self.discount
                x = y
        # Score: inverse scaled
        score = float(1.0 / (1e-6 + total))
        return {"discounted_loss": float(total), "temporal_preference": score}

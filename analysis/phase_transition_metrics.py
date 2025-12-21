from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class PhaseTransitionMetrics:
    """Minimal testbed phase-transition heuristics: look for sharp changes in observables."""

    def metric(self, xs: np.ndarray) -> Dict[str, float]:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if xs.size < 5:
            return {"jump": 0.0, "kurtosis": 0.0}
        # Finite differences
        d = np.diff(xs)
        jump = float(np.max(np.abs(d)))
        # Kurtosis proxy (heavy tails => potential transitions)
        m = np.mean(xs)
        v = np.mean((xs-m)**2) + 1e-12
        k = np.mean((xs-m)**4) / (v*v)
        return {"jump": jump, "kurtosis": float(k)}

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Protocol, Dict

class ActionFunctional(Protocol):
    def action(self, M: np.ndarray) -> float: ...
    def grad(self, M: np.ndarray) -> np.ndarray: ...

@dataclass(frozen=True)
class QuadraticAction:
    """S[M] = a Tr(M^2)"""
    a: float = 1.0

    def action(self, M: np.ndarray) -> float:
        return float(self.a * np.trace(M @ M))

    def grad(self, M: np.ndarray) -> np.ndarray:
        return 2.0 * self.a * M

@dataclass(frozen=True)
class CubicAction:
    """S[M] = Tr(M^2 + g M^3)

    This is the minimal non-Gaussian controlled model used throughout the repo.
    """
    g: float = 0.1

    def action(self, M: np.ndarray) -> float:
        M2 = M @ M
        M3 = M2 @ M
        return float(np.trace(M2 + self.g * M3))

    def grad(self, M: np.ndarray) -> np.ndarray:
        M2 = M @ M
        return 2.0 * M + 3.0 * self.g * M2

@dataclass(frozen=True)
class QuarticAction:
    """S[M] = Tr( a M^2 + b M^4 )

    Useful for stabilizing the potential when sweeping g in the cubic model.
    """
    a: float = 1.0
    b: float = 0.1

    def action(self, M: np.ndarray) -> float:
        M2 = M @ M
        M4 = M2 @ M2
        return float(np.trace(self.a * M2 + self.b * M4))

    def grad(self, M: np.ndarray) -> np.ndarray:
        M2 = M @ M
        return 2.0 * self.a * M + 4.0 * self.b * (M2 @ M)

def make_action(name: str, **kwargs) -> ActionFunctional:
    """Factory for actions."""
    name = name.lower().strip()
    if name == "quadratic":
        return QuadraticAction(**kwargs)
    if name == "cubic":
        return CubicAction(**kwargs)
    if name == "quartic":
        return QuarticAction(**kwargs)
    raise ValueError(f"Unknown action name: {name}")

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

def pad_to_pow2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    m = 1
    while m < n:
        m *= 2
    if m == n:
        return x
    out = np.zeros((m,), dtype=float)
    out[:n] = x
    return out

@dataclass(frozen=True)
class AmplitudeEncoder:
    """Amplitude encoding for small vectors into statevectors.

    For input x:
      |psi(x)> = x / ||x||

    This is classical preprocessing + quantum state initialization (minimal controlled model).
    """

    def encode(self, x: np.ndarray) -> np.ndarray:
        v = pad_to_pow2(x)
        norm = float(np.linalg.norm(v) + 1e-12)
        psi = (v / norm).astype(np.complex128)
        return psi

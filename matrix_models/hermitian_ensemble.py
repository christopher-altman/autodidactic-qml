from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(frozen=True)
class HermitianEnsemble:
    """Generator for random real-symmetric (Hermitian over R) matrices.

    We use real-symmetric matrices by default because they're fast and
    already capture the key invariants used in many minimal controlled matrix models
    (spectra, trace powers, conjugation by orthogonals).

    Parameters
    ----------
    dim:
        Matrix dimension N.
    scale:
        Standard deviation of entries before symmetrization.
    seed:
        Optional RNG seed for reproducibility.
    """
    dim: int
    scale: float = 1.0
    seed: Optional[int] = None

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def sample(self, *, rng: Optional[np.random.Generator]=None) -> np.ndarray:
        r = rng if rng is not None else self.rng()
        A = r.normal(loc=0.0, scale=self.scale, size=(self.dim, self.dim))
        M = (A + A.T) / 2.0
        return M

    def normalize_spectrum(self, M: np.ndarray, target_std: float = 1.0) -> np.ndarray:
        """Rescale matrix so eigenvalue std is `target_std`."""
        w = np.linalg.eigvalsh(M)
        s = float(np.std(w) + 1e-12)
        return M * (target_std / s)

    def split_train_eval(self, M: np.ndarray, frac: float = 0.8, *, rng: Optional[np.random.Generator]=None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Split flattened entries into train/eval vectors (useful for MI proxies)."""
        r = rng if rng is not None else self.rng()
        x = M.reshape(-1)
        idx = np.arange(x.size)
        r.shuffle(idx)
        k = int(frac * x.size)
        return x[idx[:k]], x[idx[k:]]

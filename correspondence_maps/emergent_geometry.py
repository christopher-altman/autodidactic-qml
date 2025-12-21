from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

def laplacian_from_kernel(K: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Construct a graph Laplacian from a positive kernel matrix."""
    D = np.diag(np.sum(K, axis=1) + eps)
    L = D - K
    return L

@dataclass(frozen=True)
class SpectralGeometry:
    """Extract minimal controlled model geometric observables from a matrix model configuration.

    - Use eigenvalues as 'modes'
    - Build a kernel from pairwise distances in eigenvalue space
    - Compute Laplacian spectrum and effective dimension proxy
    """
    bandwidth: float = 1.0

    def features(self, M: np.ndarray) -> np.ndarray:
        w = np.linalg.eigvalsh(M)
        return w.reshape(-1, 1)

    def rbf_kernel(self, X: np.ndarray) -> np.ndarray:
        # Pairwise squared distance: ||x_i - x_j||^2
        # X is (n, d), compute d2[i,j] = ||X[i] - X[j]||^2
        X = np.atleast_2d(X)
        # Using the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        sq_norms = np.sum(X * X, axis=1, keepdims=True)  # (n, 1)
        d2 = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
        d2 = np.maximum(d2, 0.0)  # numerical stability
        K = np.exp(-d2 / (2.0 * self.bandwidth**2 + 1e-12))
        return K

    def laplacian_spectrum(self, M: np.ndarray) -> np.ndarray:
        X = self.features(M)
        K = self.rbf_kernel(X)
        L = laplacian_from_kernel(K)
        evals = np.linalg.eigvalsh(L)
        return np.maximum(evals, 0.0)

    def effective_dimension(self, spectrum: np.ndarray, tau: float = 1.0) -> float:
        """Effective dimension via participation ratio of heat kernel weights."""
        w = np.exp(-tau * spectrum)
        w = w / (np.sum(w) + 1e-12)
        return float(1.0 / (np.sum(w*w) + 1e-12))

    def summarize(self, M: np.ndarray) -> Dict[str, float]:
        spec = self.laplacian_spectrum(M)
        return {
            "laplacian_gap": float(spec[1] - spec[0]) if spec.size > 1 else 0.0,
            "eff_dim_tau1": self.effective_dimension(spec, tau=1.0),
            "eff_dim_tau5": self.effective_dimension(spec, tau=5.0),
        }

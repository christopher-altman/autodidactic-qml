from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

def _stable_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.clamp(x, -30.0, 30.0))

@dataclass
class MatrixToCyclicRNN:
    """Map a symmetric matrix M to a small cyclic RNN.

    Idea (minimal controlled model but operational):
    - eigenvalues set stability/decay (spectral radius control)
    - eigenvectors seed recurrent weights in a gauge-invariant-ish way (since spectrum is invariant under conjugation)

    The RNN learns via self-consistency objectives in autodidactic protocols.
    """
    hidden_size: int = 64
    input_size: int = 64
    seed: Optional[int] = 0

    def build(self, M: np.ndarray) -> nn.Module:
        torch.manual_seed(int(self.seed or 0))

        w, V = np.linalg.eigh(M)
        w = np.tanh(w)  # stabilize
        # Build a recurrent weight matrix with controlled spectral radius
        # Use truncated eigenbasis if needed
        h = self.hidden_size
        dim = M.shape[0]
        k = min(dim, h)
        Vk = V[:, :k]
        wk = w[:k]

        # Construct Wrec = Q diag(wk) Q^T in hidden space
        Q = torch.randn(h, k) * 0.2
        # mix in matrix eigenvectors (as a fixed feature projection)
        P = torch.tensor(Vk, dtype=torch.float32)
        mix = torch.randn(k, k) * 0.1 + torch.eye(k) * 0.9
        # effective basis
        B = (Q @ mix)  # h x k

        diag = torch.diag(torch.tensor(wk, dtype=torch.float32))
        Wrec = B @ diag @ B.T
        # normalize spectral radius
        with torch.no_grad():
            eig = torch.linalg.eigvals(Wrec).real
            rho = torch.max(torch.abs(eig)).item() + 1e-12
            Wrec = Wrec / rho * 0.95

        class CyclicRNN(nn.Module):
            def __init__(self, input_size: int, hidden_size: int):
                super().__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.W_in = nn.Linear(input_size, hidden_size, bias=True)
                self.W_out = nn.Linear(hidden_size, input_size, bias=True)
                self.W_rec = nn.Parameter(Wrec.clone())
                self.h0 = nn.Parameter(torch.zeros(hidden_size))

            def forward(self, x: torch.Tensor, steps: int = 4) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                # x: (B, input_size)
                h = self.h0.unsqueeze(0).expand(x.shape[0], -1)
                stats = {}
                for t in range(steps):
                    pre = self.W_in(x) + h @ self.W_rec.T
                    h = torch.tanh(pre)
                    x = self.W_out(h)
                    stats[f"h_{t}"] = h
                stats["h_final"] = h
                stats["x_final"] = x
                return x, stats

        return CyclicRNN(self.input_size, self.hidden_size)

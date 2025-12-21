"""Representation-preserving constraints for functional basin testing.

==============================================================================
IMPORTANT: NEGATIVE DEMONSTRATION MODULE
==============================================================================
This module demonstrates a NEGATIVE MODE: representation similarity (Gram/CKA)
does NOT imply functional recovery.

The experiments in this repository show that even when CKA approaches 1.0
(near-identical representation geometry), the loss-based CI remains ≈ 0.

This is NOT a bug—it's the scientific finding:
    "Representation geometry is not a recoverable functional signature."

If you are looking for a working functional-basin constraint, this module
proves that Gram/CKA is NOT it.
==============================================================================

Implements:
- RepresentationAnchor: Stores PRE hidden states + Gram matrix
- compute_gram_penalty(): Differentiable Gram distance + (1-CKA)
- ProjectedRecoveryCI: Task steps + explicit projection (still fails)
- run_repr_vs_spectral_comparison(): The decisive 2×2 experiment
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# GRAM / CKA REPRESENTATION MATCHING
# ==============================================================================

def compute_gram_matrix(H: torch.Tensor) -> torch.Tensor:
    """Gram matrix G = H @ H.T for batch of hidden states."""
    return H @ H.T


def centered_kernel_alignment(G1: torch.Tensor, G2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """CKA similarity: HSIC(G1,G2) / sqrt(HSIC(G1,G1) * HSIC(G2,G2)).
    
    Returns value in [0, 1] where 1 = identical representation geometry.
    
    NOTE: High CKA does NOT imply functional recovery. See module docstring.
    """
    n = G1.shape[0]
    H = torch.eye(n, device=G1.device) - torch.ones(n, n, device=G1.device) / n
    
    def hsic(K1, K2):
        return torch.trace(K1 @ H @ K2 @ H) / ((n - 1) ** 2)
    
    hsic_12 = hsic(G1, G2)
    hsic_11 = hsic(G1, G1)
    hsic_22 = hsic(G2, G2)
    
    return hsic_12 / (torch.sqrt(hsic_11 * hsic_22) + eps)


def gram_frobenius_distance(G1: torch.Tensor, G2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Frobenius distance between Gram matrices."""
    if normalize:
        G1 = G1 / (torch.norm(G1, p="fro") + 1e-12)
        G2 = G2 / (torch.norm(G2, p="fro") + 1e-12)
    return torch.norm(G1 - G2, p="fro")


# ==============================================================================
# REPRESENTATION ANCHOR
# ==============================================================================

@dataclass
class RepresentationAnchor:
    """Stores reference hidden states for representation-preserving constraints.
    
    WARNING: This constraint class demonstrates that representation geometry
    (Gram matrices, CKA) does NOT create functional basins. Using this for
    recovery will improve CKA but NOT improve loss-based CI.
    
    This is the expected behavior and the scientific finding.
    """
    hidden_pre: torch.Tensor   # [batch, hidden] PRE hidden states
    gram_pre: torch.Tensor     # [batch, batch] PRE Gram matrix  
    eval_batch: torch.Tensor   # Fixed input for determinism
    eval_seed: int
    
    @classmethod
    def create(
        cls,
        model: nn.Module,
        batch_size: int = 64,
        eval_seed: int = 12345,
        device: str = "cpu",
    ) -> "RepresentationAnchor":
        """Create anchor from trained model on fixed eval batch."""
        gen = torch.Generator(device=device).manual_seed(eval_seed)
        eval_batch = torch.randn(batch_size, model.fc1.in_features, generator=gen, device=device)
        
        model.eval()
        with torch.no_grad():
            output = model(eval_batch)
            hidden_pre = cls._extract_hidden(output)
            gram_pre = compute_gram_matrix(hidden_pre)
        
        return cls(
            hidden_pre=hidden_pre.detach(),
            gram_pre=gram_pre.detach(),
            eval_batch=eval_batch.detach(),
            eval_seed=eval_seed,
        )
    
    @staticmethod
    def _extract_hidden(output) -> torch.Tensor:
        """Extract hidden states from model output."""
        if isinstance(output, tuple) and len(output) > 1:
            stats = output[1]
            if isinstance(stats, dict) and "h_final" in stats:
                return stats["h_final"]
        return output[0] if isinstance(output, tuple) else output
    
    def compute_gram_penalty(self, model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute differentiable Gram distance + (1 - CKA) penalty.
        
        NOTE: Minimizing this penalty improves CKA but does NOT improve CI.
        This is the expected (negative) result.
        """
        model.train()
        output = model(self.eval_batch)
        hidden_cur = self._extract_hidden(output)
        gram_cur = compute_gram_matrix(hidden_cur)
        
        gram_dist = gram_frobenius_distance(gram_cur, self.gram_pre, normalize=True)
        cka = centered_kernel_alignment(gram_cur, self.gram_pre)
        
        penalty = gram_dist + (1.0 - cka)
        
        return penalty, {
            "gram_dist": float(gram_dist.item()),
            "cka": float(cka.item()),
        }
    
    def compute_hidden_similarity(self, model: nn.Module) -> Dict[str, float]:
        """Compute representation similarity metrics (evaluation only)."""
        model.eval()
        with torch.no_grad():
            output = model(self.eval_batch)
            hidden_cur = self._extract_hidden(output)
            gram_cur = compute_gram_matrix(hidden_cur)
            
            h_pre_flat = self.hidden_pre.flatten()
            h_cur_flat = hidden_cur.flatten()
            cos_sim = float(torch.nn.functional.cosine_similarity(
                h_pre_flat.unsqueeze(0), h_cur_flat.unsqueeze(0)
            ).item())
            
            gram_dist = float(gram_frobenius_distance(gram_cur, self.gram_pre).item())
            cka = float(centered_kernel_alignment(gram_cur, self.gram_pre).item())
            
            return {
                "h_cosine": cos_sim,
                "gram_dist": gram_dist,
                "cka": cka,
            }


# ==============================================================================
# PROJECTED RECOVERY (Still fails—that's the point)
# ==============================================================================

@dataclass  
class ProjectedRecoveryCI:
    """Recovery via task steps + explicit projection to representation manifold.
    
    EXPECTED RESULT: This still fails to create a functional basin.
    
    Even with explicit projection (not gradient mixing), representation
    constraints do not enable functional recovery. This rules out the
    hypothesis that gradient cancellation was the problem.
    """
    perturb_strength: float = 0.25
    zero_frac: float = 0.05
    perturb_seed: int = 42
    eval_seed: int = 12345
    recovery_seed: Optional[int] = None
    lr: float = 1e-3
    batch_size: int = 64
    
    task_steps_per_cycle: int = 1
    projection_steps_per_cycle: int = 2
    cycles: int = 5
    
    device: str = "cpu"
    verbose: bool = False
    
    def _training_loss(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        output = model(x)
        y = output[0] if isinstance(output, tuple) else output
        return torch.mean((y - x) ** 2)
    
    def score(
        self,
        model: nn.Module,
        perturbed_model: nn.Module = None,
    ) -> Dict[str, any]:
        """Compute CI with projected recovery.
        
        EXPECTED: CI ≈ 0 even with high final CKA.
        """
        model = model.to(self.device)
        
        if perturbed_model is None:
            gen = torch.Generator(device=self.device).manual_seed(self.perturb_seed)
            perturbed_model = copy.deepcopy(model)
            with torch.no_grad():
                for p in perturbed_model.parameters():
                    if p.ndim >= 1:
                        noise = torch.randn(p.shape, generator=gen, device=self.device)
                        p.add_(self.perturb_strength * noise)
                        mask = torch.rand(p.shape, generator=gen, device=self.device) < self.zero_frac
                        p[mask] = 0.0
        
        repr_anchor = RepresentationAnchor.create(
            model, batch_size=self.batch_size, eval_seed=self.eval_seed, device=self.device
        )
        
        gen = torch.Generator(device=self.device).manual_seed(self.eval_seed)
        eval_batch = torch.randn(self.batch_size, model.fc1.in_features, generator=gen, device=self.device)
        
        # PRE loss
        model.eval()
        with torch.no_grad():
            output = model(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            base_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        # POST loss
        perturbed_model.eval()
        with torch.no_grad():
            output = perturbed_model(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            perturbed_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        if self.verbose:
            print(f"[PRE     ] loss={base_loss:.4f}")
            print(f"[POST    ] loss={perturbed_loss:.4f}")
        
        # RECOVERY
        shadow = copy.deepcopy(perturbed_model).to(self.device)
        shadow.train()
        opt = torch.optim.Adam(shadow.parameters(), lr=self.lr)
        
        if self.recovery_seed is not None:
            recovery_gen = torch.Generator(device=self.device).manual_seed(self.recovery_seed)
        else:
            recovery_gen = None
        
        trajectory = []
        
        for cycle in range(self.cycles):
            if recovery_gen is not None:
                train_batch = torch.randn(self.batch_size, model.fc1.in_features, 
                                          generator=recovery_gen, device=self.device)
            else:
                train_batch = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
            
            # Task step(s)
            for _ in range(self.task_steps_per_cycle):
                task_loss = self._training_loss(shadow, train_batch)
                opt.zero_grad()
                task_loss.backward()
                torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
                opt.step()
            
            # Projection step(s)
            for _ in range(self.projection_steps_per_cycle):
                repr_penalty, metrics = repr_anchor.compute_gram_penalty(shadow)
                opt.zero_grad()
                repr_penalty.backward()
                torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
                opt.step()
            
            with torch.no_grad():
                eval_loss = self._training_loss(shadow, eval_batch)
                trajectory.append({
                    "cycle": cycle,
                    "task_loss": float(eval_loss.item()),
                    "gram_dist": metrics["gram_dist"],
                    "cka": metrics["cka"],
                })
        
        # FINAL
        shadow.eval()
        with torch.no_grad():
            output = shadow(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            recovered_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        final_repr = repr_anchor.compute_hidden_similarity(shadow)
        
        if self.verbose:
            print(f"[RECOVER] loss={recovered_loss:.4f}")
            print(f"          gram_dist={final_repr['gram_dist']:.4f}, cka={final_repr['cka']:.4f}")
        
        damage = max(perturbed_loss - base_loss, 1e-12)
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        ci_score = float(recovery / damage)
        
        return {
            "base_loss": base_loss,
            "perturbed_loss": perturbed_loss,
            "recovered_loss": recovered_loss,
            "ci_score": ci_score,
            "final_gram_dist": final_repr["gram_dist"],
            "final_cka": final_repr["cka"],
            "final_h_cosine": final_repr["h_cosine"],
            "trajectory": trajectory,
            "cycles": self.cycles,
        }


# ==============================================================================
# COMPARISON FUNCTION
# ==============================================================================

def run_repr_vs_spectral_comparison(
    model: nn.Module,
    InvariantConstrainedCI_class,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Dict]:
    """Run 2×2 comparison: spectral vs repr constraints.
    
    EXPECTED RESULT: Both fail. That's the scientific finding.
    
    This function exists to PROVE that representation constraints
    do not create functional basins, not to find one that works.
    """
    perturb_seed = kwargs.get("perturb_seed", 42)
    recovery_seed = kwargs.get("recovery_seed", 2025)
    
    gen = torch.Generator().manual_seed(perturb_seed)
    perturbed_model = copy.deepcopy(model)
    with torch.no_grad():
        for p in perturbed_model.parameters():
            if p.ndim >= 1:
                noise = torch.randn(p.shape, generator=gen)
                p.add_(kwargs.get("perturb_strength", 0.25) * noise)
    
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON: SPECTRAL vs REPRESENTATION CONSTRAINTS")
        print("="*70)
        print("\nExpected result: BOTH fail (CI ≈ 0)")
        print("This is the scientific finding, not a bug.\n")
    
    # Spectral constraint
    if verbose:
        print("--- [1] SPECTRAL (shape) constraint ---")
    
    spectral_ci = InvariantConstrainedCI_class(
        recovery_steps=kwargs.get("recovery_steps", 10),
        invariant_weight=kwargs.get("invariant_weight", 0.1),
        penalty_keys=["fro_norm", "tr_M2", "spec_entropy"],
        perturb_seed=perturb_seed,
        recovery_seed=recovery_seed,
        verbose=verbose,
    )
    results["spectral"] = spectral_ci.score(model, perturbed_model=perturbed_model)
    
    # Representation constraint
    if verbose:
        print("\n--- [2] REPRESENTATION (Gram/CKA) constraint ---")
    
    repr_ci = ProjectedRecoveryCI(
        cycles=kwargs.get("recovery_steps", 10) // 3,
        task_steps_per_cycle=1,
        projection_steps_per_cycle=2,
        perturb_seed=perturb_seed,
        recovery_seed=recovery_seed,
        verbose=verbose,
    )
    results["repr"] = repr_ci.score(model, perturbed_model=perturbed_model)
    
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        spec_ci = results["spectral"]["ci_score"]
        repr_ci_score = results["repr"]["ci_score"]
        repr_cka = results["repr"]["final_cka"]
        
        print(f"\n  {'Constraint':<20} {'CI':>10} {'CKA':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        print(f"  {'Spectral (shape)':<20} {spec_ci:>10.4f} {'n/a':>10}")
        print(f"  {'Repr (Gram/CKA)':<20} {repr_ci_score:>10.4f} {repr_cka:>10.4f}")
        
        print("\n  INTERPRETATION:")
        print("  Both CI ≈ 0 → Neither geometry nor representation defines a functional basin")
        print("  High CKA + low CI → Representation similarity ≠ functional recovery")
        print("\n  This is the primary negative result.")
    
    return results

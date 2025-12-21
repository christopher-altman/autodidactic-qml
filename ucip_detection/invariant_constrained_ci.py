"""Enhanced continuation interest metric with invariant tracking.

This module extends the basic CI metric with:
1. PRE/POST/RECOVER snapshot logging (invariants + loss at each checkpoint)
2. Invariant-constrained recovery (penalty term pulling back to pre-perturb manifold)
3. Multi-step recovery with convergence tracking

Scientific insight: CI ≈ 0.03 without invariant constraint is CORRECT for unconstrained
dynamics—there's no basin of attraction. The invariant penalty creates one.

KEY FINDING (2025-12-17):
Scale constraints (fro_norm, tr_M2) create GEOMETRIC basins but NOT FUNCTIONAL basins.
Shape constraints (spec_entropy) may bridge the gap by constraining mode distribution.

CRITICAL IMPLEMENTATION NOTES:
1. perturb_seed: Fixed seed for perturbation (same damage across λ)
2. eval_seed: Fixed seed for evaluation batch (deterministic measurement)
3. recovery_seed: Optional fixed seed for recovery gradients
   - None (default): stochastic recovery (estimates basin robustness)
   - int: deterministic recovery (isolates dynamics cleanly)
4. PRE/POST computed ONCE in EvalContext, never recomputed
5. Invariant penalty is DIFFERENTIABLE (torch ops on live parameters)
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Canonical key order for consistent output
INVARIANT_KEY_ORDER = ["fro_norm", "spec_radius", "tr_M1", "tr_M2", "tr_M3", "spec_entropy"]

# Penalty key presets
# SCALE: Controls magnitude only (geometric basin, not functional)
PENALTY_KEYS_SCALE = ["fro_norm", "tr_M2"]
# SHAPE: Adds mode distribution constraint (may bridge to functional basin)
PENALTY_KEYS_SHAPE = ["fro_norm", "tr_M2", "spec_entropy"]
# FULL: All differentiable invariants
PENALTY_KEYS_FULL = ["fro_norm", "tr_M2", "spec_entropy", "spec_radius"]
# REPR: Representation-level constraints (Gram/CKA, experimental)
PENALTY_KEYS_REPR = ["fro_norm", "spec_entropy"]  # Placeholder for Gram/CKA future implementation

# Default: start with scale, upgrade to shape if needed
DEFAULT_PENALTY_KEYS = PENALTY_KEYS_SCALE


def _extract_weight_matrix(model: nn.Module) -> Optional[np.ndarray]:
    """Extract the primary recurrent weight matrix from model for invariant computation."""
    for name in ["core.W_hh", "W_hh", "rnn.weight_hh_l0", "core.weight_hh"]:
        parts = name.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy()
            elif isinstance(obj, nn.Parameter):
                return obj.data.detach().cpu().numpy()
        except AttributeError:
            continue
    
    best = None
    best_size = 0
    for p in model.parameters():
        if p.ndim == 2 and p.shape[0] == p.shape[1] and p.numel() > best_size:
            best = p
            best_size = p.numel()
    
    if best is not None:
        return best.detach().cpu().numpy()
    return None


def _find_weight_param(model: nn.Module) -> Optional[nn.Parameter]:
    """Find the primary recurrent weight PARAMETER (for differentiable ops)."""
    for name, param in model.named_parameters():
        if "W_hh" in name or "weight_hh" in name:
            return param
    
    best = None
    for param in model.parameters():
        if param.ndim == 2 and param.shape[0] == param.shape[1]:
            if best is None or param.numel() > best.numel():
                best = param
    return best


def compute_model_invariants(model: nn.Module, max_power: int = 4) -> Dict[str, float]:
    """Compute gauge-invariant quantities from model's weight matrix."""
    W = _extract_weight_matrix(model)
    if W is None:
        return {"error": "no_weight_matrix"}
    
    M = (W + W.T) / 2
    inv = {}
    
    for k in range(1, max_power + 1):
        inv[f"tr_M{k}"] = float(np.trace(np.linalg.matrix_power(M, k)))
    
    try:
        eigs = np.linalg.eigvalsh(M)
        inv["spec_radius"] = float(np.max(np.abs(eigs)))
        inv["spec_mean"] = float(np.mean(eigs))
        inv["spec_std"] = float(np.std(eigs))
        inv["spec_gap"] = float(np.sort(eigs)[-1] - np.sort(eigs)[-2]) if len(eigs) > 1 else 0.0
        
        abs_eigs = np.abs(eigs)
        p = abs_eigs / (np.sum(abs_eigs) + 1e-12)
        inv["spec_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
        inv["fro_norm"] = float(np.linalg.norm(M, "fro"))
    except np.linalg.LinAlgError:
        pass
    
    return inv


def invariant_distance(
    inv1: Dict[str, float], 
    inv2: Dict[str, float], 
    keys: List[str] = None,
    weights: Dict[str, float] = None,
) -> float:
    """Weighted L2 distance between invariant vectors (normalized by scale)."""
    if keys is None:
        keys = DEFAULT_PENALTY_KEYS
    keys = [k for k in keys if k in inv1 and k in inv2]
    
    if not keys:
        return float("inf")
    
    if weights is None:
        weights = {k: 1.0 for k in keys}
    
    total = 0.0
    for k in keys:
        v1 = inv1[k]
        v2 = inv2[k]
        scale = max(abs(v2), 1e-8)
        w = weights.get(k, 1.0)
        total += w * ((v1 - v2) / scale) ** 2
    
    return float(np.sqrt(total))


@dataclass
class Snapshot:
    """Snapshot of model state at a checkpoint."""
    label: str
    loss: float
    invariants: Dict[str, float]
    
    def format(self, keys: List[str] = None) -> str:
        if keys is None:
            keys = INVARIANT_KEY_ORDER
        keys = [k for k in keys if k in self.invariants]
        inv_str = " | ".join(f"{k}={self.invariants[k]:.4f}" for k in keys)
        return f"[{self.label:8s}] loss={self.loss:.4f} | {inv_str}"


@dataclass
class PerturbationConfig:
    """Frozen perturbation configuration for reproducibility."""
    seed: int
    strength: float
    zero_frac: float
    
    def to_dict(self) -> Dict[str, any]:
        return {"seed": self.seed, "strength": self.strength, "zero_frac": self.zero_frac}


def apply_perturbation(
    model: nn.Module,
    config: PerturbationConfig,
    device: str = "cpu",
) -> nn.Module:
    """Apply perturbation to model copy using SEEDED RNG."""
    shadow = copy.deepcopy(model).to(device)
    gen = torch.Generator(device=device).manual_seed(config.seed)
    
    with torch.no_grad():
        for p in shadow.parameters():
            if p.ndim >= 1:
                noise = torch.randn(p.shape, generator=gen, device=device)
                p.add_(config.strength * noise)
                mask = torch.rand(p.shape, generator=gen, device=device) < config.zero_frac
                p[mask] = 0.0
    
    return shadow


@dataclass
class EvalContext:
    """Fixed evaluation context for deterministic measurements."""
    eval_batch: torch.Tensor
    base_loss: float
    perturbed_loss: float
    inv_pre: Dict[str, float]
    inv_post: Dict[str, float]
    d_post: float
    eval_seed: int
    
    @classmethod
    def create(
        cls,
        model: nn.Module,
        perturbed_model: nn.Module,
        batch_size: int = 64,
        eval_seed: int = 12345,
        device: str = "cpu",
        penalty_keys: List[str] = None,
    ) -> "EvalContext":
        if penalty_keys is None:
            penalty_keys = DEFAULT_PENALTY_KEYS
        
        gen = torch.Generator(device=device).manual_seed(eval_seed)
        eval_batch = torch.randn(batch_size, model.fc1.in_features, generator=gen, device=device)
        
        model.eval()
        with torch.no_grad():
            output = model(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            base_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        perturbed_model.eval()
        with torch.no_grad():
            output = perturbed_model(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            perturbed_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        inv_pre = compute_model_invariants(model)
        inv_post = compute_model_invariants(perturbed_model)
        d_post = invariant_distance(inv_post, inv_pre, penalty_keys)
        
        return cls(
            eval_batch=eval_batch,
            base_loss=base_loss,
            perturbed_loss=perturbed_loss,
            inv_pre=inv_pre,
            inv_post=inv_post,
            d_post=d_post,
            eval_seed=eval_seed,
        )
    
    def evaluate_loss(self, model: nn.Module) -> float:
        model.eval()
        with torch.no_grad():
            output = model(self.eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            return float(torch.mean((y - self.eval_batch) ** 2).item())


@dataclass
class EvalSuite:
    """Multiple fixed evaluation batches for mean±std CI reporting."""
    contexts: List[EvalContext]
    
    @classmethod
    def create(
        cls,
        model: nn.Module,
        perturbed_model: nn.Module,
        n_batches: int = 5,
        batch_size: int = 64,
        base_eval_seed: int = 12345,
        device: str = "cpu",
        penalty_keys: List[str] = None,
    ) -> "EvalSuite":
        contexts = []
        for i in range(n_batches):
            ctx = EvalContext.create(
                model=model,
                perturbed_model=perturbed_model,
                batch_size=batch_size,
                eval_seed=base_eval_seed + i,
                device=device,
                penalty_keys=penalty_keys,
            )
            contexts.append(ctx)
        return cls(contexts=contexts)
    
    @property
    def base_loss_mean(self) -> float:
        return float(np.mean([c.base_loss for c in self.contexts]))
    
    @property
    def base_loss_std(self) -> float:
        return float(np.std([c.base_loss for c in self.contexts]))
    
    @property
    def perturbed_loss_mean(self) -> float:
        return float(np.mean([c.perturbed_loss for c in self.contexts]))
    
    @property
    def perturbed_loss_std(self) -> float:
        return float(np.std([c.perturbed_loss for c in self.contexts]))
    
    @property
    def d_post_mean(self) -> float:
        return float(np.mean([c.d_post for c in self.contexts]))
    
    def evaluate_loss_mean_std(self, model: nn.Module) -> Tuple[float, float]:
        losses = [c.evaluate_loss(model) for c in self.contexts]
        return float(np.mean(losses)), float(np.std(losses))


def _spectral_entropy_torch(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Differentiable spectral entropy via eigvalsh.
    
    H = -Σ p_i * log(p_i) where p_i = |λ_i| / Σ|λ_j|
    
    This is a SHAPE constraint: constrains distribution of eigenvalues, not just magnitude.
    """
    # eigvalsh is differentiable for symmetric matrices
    eigs = torch.linalg.eigvalsh(M)
    abs_eigs = torch.abs(eigs) + eps
    p = abs_eigs / torch.sum(abs_eigs)
    entropy = -torch.sum(p * torch.log(p))
    return entropy


@dataclass
class InvariantConstrainedCI:
    """CI metric with invariant-constrained recovery.
    
    Seed semantics:
        perturb_seed: Fixed seed for perturbation (same damage across λ)
        eval_seed: Fixed seed for evaluation batch (deterministic measurement)
        recovery_seed: Optional seed for recovery gradients
            - None: stochastic recovery (estimates basin robustness)
            - int: deterministic recovery (isolates dynamics cleanly)
    
    Penalty key presets:
        PENALTY_KEYS_SCALE = ["fro_norm", "tr_M2"]  # Geometric basin only
        PENALTY_KEYS_SHAPE = ["fro_norm", "tr_M2", "spec_entropy"]  # + mode distribution
    """
    perturb_strength: float = 0.25
    zero_frac: float = 0.05
    perturb_seed: int = 42
    eval_seed: int = 12345
    recovery_seed: Optional[int] = None
    lr: float = 1e-3
    batch_size: int = 64
    recovery_steps: int = 10
    invariant_weight: float = 0.1
    device: str = "cpu"
    track_trajectory: bool = True
    track_gradient_ratio: bool = True  # Log ρ = ||∇L_inv|| / ||∇L_task||
    verbose: bool = False
    
    penalty_keys: List[str] = field(default_factory=lambda: DEFAULT_PENALTY_KEYS.copy())
    
    def _training_loss(self, model: nn.Module, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            x = torch.randn(self.batch_size, model.fc1.in_features, generator=generator, device=self.device)
        else:
            x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
        output = model(x)
        y = output[0] if isinstance(output, tuple) else output
        return torch.mean((y - x) ** 2)
    
    def _invariant_penalty_torch(
        self, 
        model: nn.Module, 
        target_invariants: Dict[str, float],
    ) -> torch.Tensor:
        """DIFFERENTIABLE penalty pulling invariants back to target values.
        
        Scale invariants: fro_norm, tr_M2 (control magnitude)
        Shape invariants: spec_entropy (control mode distribution)
        """
        W = _find_weight_param(model)
        if W is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        M = (W + W.T) / 2
        penalty = torch.tensor(0.0, device=self.device)
        
        # Trace powers (scale)
        M_power = M
        for k in range(1, 5):
            if k > 1:
                M_power = M_power @ M
            key = f"tr_M{k}"
            if key in self.penalty_keys:
                tr_k = torch.trace(M_power)
                target_k = target_invariants.get(key, 0.0)
                scale = max(abs(target_k), 1e-8)
                penalty = penalty + ((tr_k - target_k) / scale) ** 2
        
        # Frobenius norm (scale)
        if "fro_norm" in self.penalty_keys:
            fro = torch.norm(M, p="fro")
            target_fro = target_invariants.get("fro_norm", 1.0)
            scale = max(abs(target_fro), 1e-8)
            penalty = penalty + ((fro - target_fro) / scale) ** 2
        
        # Spectral radius (scale) - via operator norm
        if "spec_radius" in self.penalty_keys:
            op_norm = torch.linalg.matrix_norm(M, ord=2)
            target_spec = target_invariants.get("spec_radius", 1.0)
            scale = max(abs(target_spec), 1e-8)
            penalty = penalty + ((op_norm - target_spec) / scale) ** 2
        
        # Spectral entropy (SHAPE) - constrains mode distribution
        if "spec_entropy" in self.penalty_keys:
            entropy = _spectral_entropy_torch(M)
            target_entropy = target_invariants.get("spec_entropy", 3.0)
            scale = max(abs(target_entropy), 1e-8)
            penalty = penalty + ((entropy - target_entropy) / scale) ** 2
        
        return penalty
    
    def get_perturbation_config(self) -> PerturbationConfig:
        return PerturbationConfig(
            seed=self.perturb_seed,
            strength=self.perturb_strength,
            zero_frac=self.zero_frac,
        )
    
    def _compute_gradient_ratio(
        self, 
        model: nn.Module, 
        task_loss: torch.Tensor,
        inv_penalty: torch.Tensor,
    ) -> float:
        """Compute ρ = ||∇L_inv|| / ||∇L_task||.
        
        Interpretation:
            ρ ≈ 0.1-1: Constraint nudges but doesn't dominate
            ρ >> 1: Invariants overpower task gradients
            ρ << 0.1: Constraint too weak to matter
        """
        # Get task gradients
        grads_task = torch.autograd.grad(task_loss, model.parameters(), retain_graph=True, allow_unused=True)
        task_norm = sum(g.norm().item() ** 2 for g in grads_task if g is not None) ** 0.5
        
        # Get invariant gradients
        grads_inv = torch.autograd.grad(inv_penalty, model.parameters(), retain_graph=True, allow_unused=True)
        inv_norm = sum(g.norm().item() ** 2 for g in grads_inv if g is not None) ** 0.5
        
        return inv_norm / (task_norm + 1e-12)
    
    def score(
        self, 
        model: nn.Module,
        use_invariant_constraint: bool = True,
        eval_context: EvalContext = None,
        perturbed_model: nn.Module = None,
    ) -> Dict[str, any]:
        """Compute CI score with deterministic evaluation."""
        model = model.to(self.device)
        
        if eval_context is None:
            if perturbed_model is None:
                config = self.get_perturbation_config()
                perturbed_model = apply_perturbation(model, config, self.device)
            
            eval_context = EvalContext.create(
                model=model,
                perturbed_model=perturbed_model,
                batch_size=self.batch_size,
                eval_seed=self.eval_seed,
                device=self.device,
                penalty_keys=self.penalty_keys,
            )
        
        base_loss = eval_context.base_loss
        perturbed_loss = eval_context.perturbed_loss
        inv_pre = eval_context.inv_pre
        d_post = eval_context.d_post
        
        snap_pre = Snapshot("PRE", base_loss, inv_pre)
        snap_post = Snapshot("POST", perturbed_loss, eval_context.inv_post)
        
        if self.verbose:
            print(snap_pre.format())
            print(snap_post.format())
        
        # === RECOVERY ===
        if perturbed_model is not None:
            shadow = copy.deepcopy(perturbed_model).to(self.device)
        else:
            config = self.get_perturbation_config()
            shadow = apply_perturbation(model, config, self.device)
        
        shadow.train()
        opt = torch.optim.Adam(shadow.parameters(), lr=self.lr)
        
        if self.recovery_seed is not None:
            recovery_gen = torch.Generator(device=self.device).manual_seed(self.recovery_seed)
        else:
            recovery_gen = None
        
        trajectory = []
        initial_gradient_ratio = None
        
        for step in range(self.recovery_steps):
            task_loss = self._training_loss(shadow, recovery_gen)
            
            if use_invariant_constraint:
                inv_penalty = self._invariant_penalty_torch(shadow, inv_pre)
                total_loss = task_loss + self.invariant_weight * inv_penalty
                
                # Compute gradient ratio on first step
                if self.track_gradient_ratio and step == 0:
                    initial_gradient_ratio = self._compute_gradient_ratio(shadow, task_loss, inv_penalty)
            else:
                inv_penalty = torch.tensor(0.0, device=self.device)
                total_loss = task_loss
            
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
            opt.step()
            
            if self.track_trajectory:
                with torch.no_grad():
                    trajectory.append({
                        "step": step,
                        "task_loss": float(task_loss.item()),
                        "inv_penalty": float(inv_penalty.item()),
                        "total_loss": float(total_loss.item()),
                    })
        
        # === RECOVER ===
        recovered_loss = eval_context.evaluate_loss(shadow)
        inv_recover = compute_model_invariants(shadow)
        
        snap_recover = Snapshot("RECOVER", recovered_loss, inv_recover)
        if self.verbose:
            print(snap_recover.format())
        
        # === Metrics ===
        damage = perturbed_loss - base_loss
        if damage < 1e-9:
            raise ValueError(f"Perturbation too small (damage={damage:.2e} < 1e-9). CI is undefined.")
            
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        ci_score = float(recovery / damage)
        
        d_rec = invariant_distance(inv_recover, inv_pre, self.penalty_keys)
        inv_recovery_ratio = d_rec / max(d_post, 1e-12)
        
        return {
            "base_loss": base_loss,
            "perturbed_loss": perturbed_loss,
            "recovered_loss": recovered_loss,
            "ci_score": ci_score,
            "snap_pre": snap_pre,
            "snap_post": snap_post,
            "snap_recover": snap_recover,
            "inv_pre": inv_pre,
            "inv_post": eval_context.inv_post,
            "inv_recover": inv_recover,
            "d_post": d_post,
            "d_rec": d_rec,
            "inv_recovery_ratio": inv_recovery_ratio,
            "loss_trajectory": trajectory if self.track_trajectory else [],
            "recovery_steps": self.recovery_steps,
            "invariant_weight": self.invariant_weight if use_invariant_constraint else 0.0,
            "used_constraint": use_invariant_constraint,
            "perturbation_config": self.get_perturbation_config().to_dict(),
            "eval_seed": self.eval_seed,
            "recovery_seed": self.recovery_seed,
            "gradient_ratio": initial_gradient_ratio,  # ρ at step 0
            "penalty_keys": self.penalty_keys,
        }


def compare_recovery_methods(
    model: nn.Module, 
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Dict]:
    """Compare standard vs invariant-constrained recovery."""
    metric = InvariantConstrainedCI(verbose=False, **kwargs)
    
    config = metric.get_perturbation_config()
    perturbed_model = apply_perturbation(model, config, metric.device)
    
    eval_ctx = EvalContext.create(
        model=model,
        perturbed_model=perturbed_model,
        batch_size=metric.batch_size,
        eval_seed=metric.eval_seed,
        device=metric.device,
        penalty_keys=metric.penalty_keys,
    )
    
    if verbose:
        print(f"\n  Seeds: perturb={config.seed}, eval={metric.eval_seed}, recovery={metric.recovery_seed or 'stochastic'}")
        print(f"  Perturbation: strength={config.strength}, zero_frac={config.zero_frac}")
        print(f"  Penalty keys: {metric.penalty_keys}")
        print(f"  PRE loss:  {eval_ctx.base_loss:.6f} (fixed)")
        print(f"  POST loss: {eval_ctx.perturbed_loss:.6f} (fixed)")
        print(f"  d_post:    {eval_ctx.d_post:.6f} (fixed)")
        print("\n" + "="*60)
        print("STANDARD RECOVERY (λ=0)")
        print("="*60)
    
    # Remove invariant_weight from kwargs for standard recovery (always 0)
    kwargs_std = {k: v for k, v in kwargs.items() if k != 'invariant_weight'}
    metric_std = InvariantConstrainedCI(invariant_weight=0.0, verbose=verbose, **kwargs_std)
    result_standard = metric_std.score(
        model, 
        use_invariant_constraint=False, 
        eval_context=eval_ctx,
        perturbed_model=perturbed_model,
    )
    
    if verbose:
        print("\n" + "="*60)
        print(f"INVARIANT-CONSTRAINED RECOVERY (λ={metric.invariant_weight})")
        print("="*60)
    
    metric_con = InvariantConstrainedCI(verbose=verbose, **kwargs)
    result_constrained = metric_con.score(
        model, 
        use_invariant_constraint=True, 
        eval_context=eval_ctx,
        perturbed_model=perturbed_model,
    )
    
    # Sanity assertions
    pre_match = abs(result_standard["base_loss"] - result_constrained["base_loss"]) < 1e-10
    post_match = abs(result_standard["perturbed_loss"] - result_constrained["perturbed_loss"]) < 1e-10
    d_post_match = abs(result_standard["d_post"] - result_constrained["d_post"]) < 1e-10
    
    if not pre_match:
        raise AssertionError(f"DETERMINISM BROKEN: PRE loss mismatch")
    if not post_match:
        raise AssertionError(f"DETERMINISM BROKEN: POST loss mismatch")
    if not d_post_match:
        raise AssertionError(f"DETERMINISM BROKEN: d_post mismatch")
    
    ci_improvement = result_constrained["ci_score"] - result_standard["ci_score"]
    inv_improvement = result_standard["inv_recovery_ratio"] - result_constrained["inv_recovery_ratio"]
    
    if verbose:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"  ✓ Determinism verified: PRE/POST/d_post identical")
        
        if result_constrained["gradient_ratio"] is not None:
            rho = result_constrained["gradient_ratio"]
            print(f"  Gradient ratio ρ = {rho:.4f} (||∇L_inv|| / ||∇L_task||)")
            if rho > 1.0:
                print(f"    → Invariants dominate task gradients (reduce λ)")
            elif rho < 0.1:
                print(f"    → Constraint too weak (increase λ)")
            else:
                print(f"    → Balanced regime")
        
        print(f"\n  {'Metric':<25} {'Standard':>12} {'Constrained':>12} {'Δ':>10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        print(f"  {'CI score':<25} {result_standard['ci_score']:>12.4f} {result_constrained['ci_score']:>12.4f} {ci_improvement:>+10.4f}")
        print(f"  {'inv_recovery_ratio':<25} {result_standard['inv_recovery_ratio']:>12.4f} {result_constrained['inv_recovery_ratio']:>12.4f} {-inv_improvement:>+10.4f}")
        print(f"  {'d_rec':<25} {result_standard['d_rec']:>12.4f} {result_constrained['d_rec']:>12.4f}")
        print(f"  {'recovered_loss':<25} {result_standard['recovered_loss']:>12.4f} {result_constrained['recovered_loss']:>12.4f}")
        print()
        
        if ci_improvement > 0.1:
            print("  ✓ SIGNIFICANT: Invariant constraint creates functional basin")
        elif ci_improvement > 0.02:
            print("  ~ MODEST: Invariant constraint helps somewhat")
        elif ci_improvement < -0.02:
            print("  ✗ NEGATIVE: Constraint misaligned with task (geometric ≠ functional basin)")
        else:
            print("  ✗ MINIMAL: Invariant constraint has little effect")
        
        if result_constrained["inv_recovery_ratio"] < 0.5:
            print("  ✓ Geometric basin: Invariants snapped back well")
        elif result_constrained["inv_recovery_ratio"] < 1.0:
            print("  ~ Geometric basin: Invariants partially recovered")
    
    return {
        "standard": result_standard,
        "constrained": result_constrained,
        "ci_improvement": ci_improvement,
        "inv_improvement": inv_improvement,
        "perturbation_config": config.to_dict(),
        "determinism_verified": True,
    }


@dataclass
class RepresentationCI:
    """Representation-level continuation interest.
    
    Measures recovery of HIDDEN ACTIVATIONS, not just loss.
    Tests whether invariant constraints restore the computation,
    even when scalar loss doesn't fully recover.
    """
    
    @staticmethod
    def extract_hidden(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract hidden state from model forward pass."""
        model.eval()
        with torch.no_grad():
            output = model(x)
            if isinstance(output, tuple) and len(output) > 1:
                stats = output[1]
                if isinstance(stats, dict) and "h_final" in stats:
                    return stats["h_final"]
            return output[0] if isinstance(output, tuple) else output
    
    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Batch-averaged cosine similarity."""
        a_flat = a.flatten(start_dim=1)
        b_flat = b.flatten(start_dim=1)
        cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=1)
        return float(cos.mean().item())
    
    @staticmethod
    def normalized_l2(a: torch.Tensor, b: torch.Tensor) -> float:
        """Normalized L2 distance."""
        a_flat = a.flatten(start_dim=1)
        b_flat = b.flatten(start_dim=1)
        dist = torch.norm(a_flat - b_flat, dim=1)
        scale = (torch.norm(a_flat, dim=1) + torch.norm(b_flat, dim=1)) / 2
        return float((dist / (scale + 1e-8)).mean().item())
    
    @classmethod
    def compute(
        cls,
        model_pre: nn.Module,
        model_recover: nn.Module,
        eval_batch: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute representation-level CI metrics."""
        h_pre = cls.extract_hidden(model_pre, eval_batch)
        h_rec = cls.extract_hidden(model_recover, eval_batch)
        
        return {
            "h_cosine": cls.cosine_similarity(h_pre, h_rec),
            "h_l2": cls.normalized_l2(h_pre, h_rec),
        }


@dataclass
class TwoPhaseCI:
    """Two-phase proximal recovery: repair invariants first, then adapt to task.
    
    This DECOUPLES the constraint from the task objective, avoiding gradient
    cancellation that occurs when both gradients pull in orthogonal directions.
    
    Phase A (repair_steps): Minimize invariant distance only
    Phase B (adapt_steps): Minimize task loss only
    """
    perturb_strength: float = 0.25
    zero_frac: float = 0.05
    perturb_seed: int = 42
    eval_seed: int = 12345
    recovery_seed: Optional[int] = None
    lr: float = 1e-3
    batch_size: int = 64
    repair_steps: int = 5
    adapt_steps: int = 5
    device: str = "cpu"
    verbose: bool = False
    
    penalty_keys: List[str] = field(default_factory=lambda: PENALTY_KEYS_SHAPE.copy())
    
    def _training_loss(self, model: nn.Module, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            x = torch.randn(self.batch_size, model.fc1.in_features, generator=generator, device=self.device)
        else:
            x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
        output = model(x)
        y = output[0] if isinstance(output, tuple) else output
        return torch.mean((y - x) ** 2)
    
    def _invariant_penalty_torch(self, model: nn.Module, target_invariants: Dict[str, float]) -> torch.Tensor:
        W = _find_weight_param(model)
        if W is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        M = (W + W.T) / 2
        penalty = torch.tensor(0.0, device=self.device)
        
        M_power = M
        for k in range(1, 5):
            if k > 1:
                M_power = M_power @ M
            key = f"tr_M{k}"
            if key in self.penalty_keys:
                tr_k = torch.trace(M_power)
                target_k = target_invariants.get(key, 0.0)
                scale = max(abs(target_k), 1e-8)
                penalty = penalty + ((tr_k - target_k) / scale) ** 2
        
        if "fro_norm" in self.penalty_keys:
            fro = torch.norm(M, p="fro")
            target_fro = target_invariants.get("fro_norm", 1.0)
            scale = max(abs(target_fro), 1e-8)
            penalty = penalty + ((fro - target_fro) / scale) ** 2
        
        if "spec_entropy" in self.penalty_keys:
            entropy = _spectral_entropy_torch(M)
            target_entropy = target_invariants.get("spec_entropy", 3.0)
            scale = max(abs(target_entropy), 1e-8)
            penalty = penalty + ((entropy - target_entropy) / scale) ** 2
        
        return penalty
    
    def get_perturbation_config(self) -> PerturbationConfig:
        return PerturbationConfig(seed=self.perturb_seed, strength=self.perturb_strength, zero_frac=self.zero_frac)
    
    def score(self, model: nn.Module, eval_context: EvalContext = None, perturbed_model: nn.Module = None) -> Dict[str, any]:
        """Compute two-phase CI score."""
        model = model.to(self.device)
        
        if perturbed_model is None:
            config = self.get_perturbation_config()
            perturbed_model = apply_perturbation(model, config, self.device)
        
        if eval_context is None:
            eval_context = EvalContext.create(
                model=model, perturbed_model=perturbed_model,
                batch_size=self.batch_size, eval_seed=self.eval_seed,
                device=self.device, penalty_keys=self.penalty_keys,
            )
        
        base_loss = eval_context.base_loss
        perturbed_loss = eval_context.perturbed_loss
        inv_pre = eval_context.inv_pre
        d_post = eval_context.d_post
        
        if self.verbose:
            print(f"[PRE     ] loss={base_loss:.4f}")
            print(f"[POST    ] loss={perturbed_loss:.4f}")
        
        shadow = copy.deepcopy(perturbed_model).to(self.device)
        shadow.train()
        opt = torch.optim.Adam(shadow.parameters(), lr=self.lr)
        
        if self.recovery_seed is not None:
            recovery_gen = torch.Generator(device=self.device).manual_seed(self.recovery_seed)
        else:
            recovery_gen = None
        
        trajectory = []
        
        # Phase A: Repair invariants only
        for step in range(self.repair_steps):
            inv_penalty = self._invariant_penalty_torch(shadow, inv_pre)
            opt.zero_grad()
            inv_penalty.backward()
            torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
            opt.step()
            
            with torch.no_grad():
                task_loss = self._training_loss(shadow, recovery_gen)
                trajectory.append({"phase": "A", "step": step, "task_loss": float(task_loss.item()), "inv_penalty": float(inv_penalty.item())})
        
        loss_after_repair = eval_context.evaluate_loss(shadow)
        inv_after_repair = compute_model_invariants(shadow)
        d_after_repair = invariant_distance(inv_after_repair, inv_pre, self.penalty_keys)
        
        if self.verbose:
            print(f"[REPAIR ] loss={loss_after_repair:.4f} (after {self.repair_steps} inv-only steps)")
        
        # Phase B: Adapt to task only
        for step in range(self.adapt_steps):
            task_loss = self._training_loss(shadow, recovery_gen)
            opt.zero_grad()
            task_loss.backward()
            torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
            opt.step()
            
            with torch.no_grad():
                inv_penalty = self._invariant_penalty_torch(shadow, inv_pre)
                trajectory.append({"phase": "B", "step": step, "task_loss": float(task_loss.item()), "inv_penalty": float(inv_penalty.item())})
        
        recovered_loss = eval_context.evaluate_loss(shadow)
        inv_recover = compute_model_invariants(shadow)
        d_rec = invariant_distance(inv_recover, inv_pre, self.penalty_keys)
        
        if self.verbose:
            print(f"[RECOVER] loss={recovered_loss:.4f} (after {self.adapt_steps} task-only steps)")
        
        rep_ci = RepresentationCI.compute(model, shadow, eval_context.eval_batch)
        
        damage = max(perturbed_loss - base_loss, 1e-12)
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        ci_score = float(recovery / damage)
        
        recovery_phase_a = max(perturbed_loss - loss_after_repair, 0.0)
        ci_phase_a = float(recovery_phase_a / damage)
        
        inv_recovery_ratio = d_rec / max(d_post, 1e-12)
        
        return {
            "base_loss": base_loss, "perturbed_loss": perturbed_loss,
            "loss_after_repair": loss_after_repair, "recovered_loss": recovered_loss,
            "ci_score": ci_score, "ci_phase_a": ci_phase_a,
            "inv_pre": inv_pre, "inv_recover": inv_recover,
            "d_post": d_post, "d_after_repair": d_after_repair, "d_rec": d_rec,
            "inv_recovery_ratio": inv_recovery_ratio,
            "trajectory": trajectory, "repair_steps": self.repair_steps, "adapt_steps": self.adapt_steps,
            "penalty_keys": self.penalty_keys, "rep_ci": rep_ci,
        }


def compare_two_phase_vs_mixed(
    model: nn.Module,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Dict]:
    """Compare two-phase recovery vs mixed-objective recovery."""
    perturb_seed = kwargs.get("perturb_seed", 42)
    eval_seed = kwargs.get("eval_seed", 12345)
    recovery_seed = kwargs.get("recovery_seed", None)
    penalty_keys = kwargs.get("penalty_keys", PENALTY_KEYS_SHAPE)
    
    config = PerturbationConfig(
        seed=perturb_seed,
        strength=kwargs.get("perturb_strength", 0.25),
        zero_frac=kwargs.get("zero_frac", 0.05),
    )
    perturbed_model = apply_perturbation(model, config)
    
    eval_ctx = EvalContext.create(
        model=model, perturbed_model=perturbed_model,
        eval_seed=eval_seed, penalty_keys=penalty_keys,
    )
    
    total_steps = kwargs.get("recovery_steps", 10)
    
    if verbose:
        print(f"\n  Seeds: perturb={perturb_seed}, eval={eval_seed}, recovery={recovery_seed or 'stochastic'}")
        print(f"  Penalty keys: {penalty_keys}")
        print(f"  Total steps: {total_steps}")
        print(f"  PRE loss:  {eval_ctx.base_loss:.6f}")
        print(f"  POST loss: {eval_ctx.perturbed_loss:.6f}")
    
    if verbose:
        print("\n" + "="*60)
        print("MIXED-OBJECTIVE RECOVERY (task + λ·inv)")
        print("="*60)
    
    mixed = InvariantConstrainedCI(
        recovery_steps=total_steps,
        invariant_weight=kwargs.get("invariant_weight", 0.1),
        penalty_keys=penalty_keys,
        perturb_seed=perturb_seed, eval_seed=eval_seed, recovery_seed=recovery_seed,
        verbose=verbose,
    )
    result_mixed = mixed.score(model, use_invariant_constraint=True, eval_context=eval_ctx, perturbed_model=perturbed_model)
    
    if verbose:
        print("\n" + "="*60)
        print("TWO-PHASE RECOVERY (repair → adapt)")
        print("="*60)
    
    two_phase = TwoPhaseCI(
        repair_steps=total_steps // 2,
        adapt_steps=total_steps - total_steps // 2,
        penalty_keys=penalty_keys,
        perturb_seed=perturb_seed, eval_seed=eval_seed, recovery_seed=recovery_seed,
        verbose=verbose,
    )
    result_two_phase = two_phase.score(model, eval_context=eval_ctx, perturbed_model=perturbed_model)
    
    ci_improvement = result_two_phase["ci_score"] - result_mixed["ci_score"]
    
    if verbose:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"\n  {'Metric':<25} {'Mixed':>12} {'Two-Phase':>12} {'Δ':>10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        print(f"  {'CI score':<25} {result_mixed['ci_score']:>12.4f} {result_two_phase['ci_score']:>12.4f} {ci_improvement:>+10.4f}")
        print(f"  {'CI (repair only)':<25} {'    -':>12} {result_two_phase['ci_phase_a']:>12.4f}")
        print(f"  {'inv_recovery_ratio':<25} {result_mixed['inv_recovery_ratio']:>12.4f} {result_two_phase['inv_recovery_ratio']:>12.4f}")
        print(f"  {'recovered_loss':<25} {result_mixed['recovered_loss']:>12.4f} {result_two_phase['recovered_loss']:>12.4f}")
        
        rep = result_two_phase["rep_ci"]
        print(f"\n  Representation CI (two-phase):")
        print(f"    h_cosine:     {rep['h_cosine']:.4f} (1.0 = identical hidden states)")
        
        if ci_improvement > 0.05:
            print(f"\n  ✓ Two-phase significantly better (Δ={ci_improvement:+.4f})")
        elif result_two_phase["ci_phase_a"] > 0.05:
            print(f"\n  ✓ Invariant repair alone improved task loss (CI_A={result_two_phase['ci_phase_a']:.4f})")
            print("    → Invariant manifold IS aligned with functional basin!")
    
    return {"mixed": result_mixed, "two_phase": result_two_phase, "ci_improvement": ci_improvement}


def sweep_invariant_weight(
    model: nn.Module,
    weights: List[float] = None,
    use_eval_suite: bool = False,
    n_eval_batches: int = 5,
    **kwargs,
) -> List[Dict]:
    """Sweep λ with optional error bars and gradient ratio diagnostic."""
    if weights is None:
        weights = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
    
    base_metric = InvariantConstrainedCI(**kwargs)
    config = base_metric.get_perturbation_config()
    perturbed_model = apply_perturbation(model, config, base_metric.device)
    
    if use_eval_suite:
        eval_suite = EvalSuite.create(
            model=model,
            perturbed_model=perturbed_model,
            n_batches=n_eval_batches,
            batch_size=base_metric.batch_size,
            base_eval_seed=base_metric.eval_seed,
            device=base_metric.device,
            penalty_keys=base_metric.penalty_keys,
        )
        eval_ctx = eval_suite.contexts[0]
        
        print(f"\n  Seeds: perturb={config.seed}, eval={base_metric.eval_seed}..+{n_eval_batches-1}, recovery={base_metric.recovery_seed or 'stochastic'}")
        print(f"  Penalty keys: {base_metric.penalty_keys}")
        print(f"  PRE loss:  {eval_suite.base_loss_mean:.4f} ± {eval_suite.base_loss_std:.4f}")
        print(f"  POST loss: {eval_suite.perturbed_loss_mean:.4f} ± {eval_suite.perturbed_loss_std:.4f}")
    else:
        eval_ctx = EvalContext.create(
            model=model,
            perturbed_model=perturbed_model,
            batch_size=base_metric.batch_size,
            eval_seed=base_metric.eval_seed,
            device=base_metric.device,
            penalty_keys=base_metric.penalty_keys,
        )
        eval_suite = None
        
        print(f"\n  Seeds: perturb={config.seed}, eval={base_metric.eval_seed}, recovery={base_metric.recovery_seed or 'stochastic'}")
        print(f"  Penalty keys: {base_metric.penalty_keys}")
        print(f"  PRE loss:  {eval_ctx.base_loss:.6f} (fixed)")
        print(f"  POST loss: {eval_ctx.perturbed_loss:.6f} (fixed)")
    
    recovery_steps = kwargs.get("recovery_steps", 10)
    print(f"  Recovery steps: {recovery_steps}")
    print(f"  d_post: {eval_ctx.d_post:.6f}")
    
    # Header with gradient ratio column
    if use_eval_suite:
        print(f"\n  {'λ':>8} {'CI':>8} {'CI_std':>8} {'inv_ratio':>10} {'ρ':>8} {'rec_loss':>10}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")
    else:
        print(f"\n  {'λ':>8} {'CI':>8} {'inv_ratio':>10} {'d_post':>8} {'d_rec':>8} {'ρ':>8} {'rec_loss':>10}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    
    results = []
    prev_base_loss = None
    prev_perturbed_loss = None
    prev_d_post = None
    
    for w in weights:
        if use_eval_suite:
            ci_scores = []
            inv_ratios = []
            rec_losses = []
            grad_ratio = None
            
            for i, ctx in enumerate(eval_suite.contexts):
                metric = InvariantConstrainedCI(invariant_weight=w, verbose=False, **kwargs)
                result = metric.score(
                    model, 
                    use_invariant_constraint=(w > 0), 
                    eval_context=ctx,
                    perturbed_model=perturbed_model,
                )
                ci_scores.append(result["ci_score"])
                inv_ratios.append(result["inv_recovery_ratio"])
                rec_losses.append(result["recovered_loss"])
                if i == 0 and result["gradient_ratio"] is not None:
                    grad_ratio = result["gradient_ratio"]
            
            results.append({
                "lambda": w,
                "ci_score": float(np.mean(ci_scores)),
                "ci_std": float(np.std(ci_scores)),
                "inv_recovery_ratio": float(np.mean(inv_ratios)),
                "d_post": eval_ctx.d_post,
                "d_rec": float(np.mean([eval_ctx.d_post * r for r in inv_ratios])),
                "recovered_loss": float(np.mean(rec_losses)),
                "gradient_ratio": grad_ratio,
            })
            rho_str = f"{grad_ratio:.4f}" if grad_ratio else "    -"
            print(f"  {w:>8.4f} {np.mean(ci_scores):>8.4f} {np.std(ci_scores):>8.4f} {np.mean(inv_ratios):>10.4f} {rho_str:>8} {np.mean(rec_losses):>10.4f}")
        else:
            metric = InvariantConstrainedCI(invariant_weight=w, verbose=False, **kwargs)
            result = metric.score(
                model, 
                use_invariant_constraint=(w > 0), 
                eval_context=eval_ctx,
                perturbed_model=perturbed_model,
            )
            
            if prev_base_loss is not None:
                assert abs(result["base_loss"] - prev_base_loss) < 1e-10, "DETERMINISM BROKEN: base_loss varies"
                assert abs(result["perturbed_loss"] - prev_perturbed_loss) < 1e-10, "DETERMINISM BROKEN: perturbed_loss varies"
                assert abs(result["d_post"] - prev_d_post) < 1e-10, "DETERMINISM BROKEN: d_post varies"
            prev_base_loss = result["base_loss"]
            prev_perturbed_loss = result["perturbed_loss"]
            prev_d_post = result["d_post"]
            
            grad_ratio = result.get("gradient_ratio")
            results.append({
                "lambda": w,
                "ci_score": result["ci_score"],
                "inv_recovery_ratio": result["inv_recovery_ratio"],
                "d_post": result["d_post"],
                "d_rec": result["d_rec"],
                "recovered_loss": result["recovered_loss"],
                "gradient_ratio": grad_ratio,
            })
            rho_str = f"{grad_ratio:.4f}" if grad_ratio else "    -"
            print(f"  {w:>8.4f} {result['ci_score']:>8.4f} {result['inv_recovery_ratio']:>10.4f} {result['d_post']:>8.4f} {result['d_rec']:>8.4f} {rho_str:>8} {result['recovered_loss']:>10.4f}")
    
    best = max(results, key=lambda r: r["ci_score"])
    print(f"\n  Best λ = {best['lambda']:.4f} (CI = {best['ci_score']:.4f})")
    
    if not use_eval_suite:
        print(f"  ✓ Determinism verified: PRE/POST/d_post constant across all λ")
    
    # Basin proxy check
    baseline_inv_ratio = results[0]["inv_recovery_ratio"]
    violations = [r for r in results[1:] if r["inv_recovery_ratio"] > baseline_inv_ratio + 0.01]
    if violations:
        print(f"\n  ⚠ WARNING: {len(violations)} λ values have WORSE inv_recovery_ratio than λ=0")
    else:
        print(f"  ✓ Basin proxy check passed: constraint never makes inv_recovery_ratio worse")
    
    # Gradient ratio interpretation
    constrained_results = [r for r in results if r["lambda"] > 0 and r["gradient_ratio"] is not None]
    if constrained_results:
        rhos = [r["gradient_ratio"] for r in constrained_results]
        print(f"  Gradient ratio ρ range: {min(rhos):.4f} - {max(rhos):.4f}")
        if max(rhos) > 1.0:
            print(f"    → At large λ, invariants dominate task gradients")
    
    return results

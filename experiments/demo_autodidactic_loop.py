"""Autodidactic loop demo with PRE/POST/RECOVER invariant diagnostics.

This demo shows:
1. Matrix → RNN correspondence mapping
2. Self-consistency + mutual information learning
3. CI robustness testing with invariant tracking
4. Comparison of standard vs invariant-constrained recovery

Key scientific insight:
- CI ≈ 0.03 without constraint is CORRECT (no basin of attraction)
- With invariant penalty, we CREATE a basin
- If CI improves significantly under invariant-constrained recovery—especially
  with recovery_steps=1—this is evidence that these invariants define an
  attracting constraint manifold (a basin proxy) for the learning dynamics.
"""

from __future__ import annotations
import argparse
import numpy as np
import torch

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import CubicAction
from matrix_models.sampler_langevin import LangevinSampler
from matrix_models.gauge_symmetries import invariants
from correspondence_maps.matrix_to_rnn import MatrixToCyclicRNN
from autodidactic_protocols.unsupervised_update import SelfConsistencyLearner
from autodidactic_protocols.mutual_information_drive import MutualInformationLearner
from analysis.law_stability_tracker import LawStabilityTracker
from ucip_detection.invariant_constrained_ci import (
    InvariantConstrainedCI,
    compare_recovery_methods,
    compute_model_invariants,
    sweep_invariant_weight,
    compare_two_phase_vs_mixed,
    INVARIANT_KEY_ORDER,
    PENALTY_KEYS_SCALE,
    PENALTY_KEYS_SHAPE,
    PENALTY_KEYS_FULL,
    PENALTY_KEYS_REPR,
    PerturbationConfig,
)


def parse_args():
    p = argparse.ArgumentParser(description="Autodidactic loop with invariant-constrained CI.")
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--T", type=float, default=1e-2)
    p.add_argument("--g", type=float, default=0.1)
    p.add_argument("--net-width", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturb-seed", type=int, default=42,
                   help="Seed for perturbation RNG (for reproducibility)")
    p.add_argument("--recovery-seed", type=int, default=None,
                   help="Seed for recovery RNG (None=stochastic, int=deterministic)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--recovery-steps", type=int, default=10,
                   help="1 = basin existence (hard test), N = practical recoverability")
    p.add_argument("--invariant-weight", type=float, default=0.1,
                   help="λ for invariant penalty (dimensionless)")
    p.add_argument("--penalty-keys", type=str, default="scale",
                   choices=["scale", "shape", "full", "repr"],
                   help="Penalty preset: scale (fro_norm+tr_M2), shape (+spec_entropy), full (+spec_radius), repr (Gram/CKA)")
    p.add_argument("--sweep-weights", action="store_true", 
                   help="Sweep λ on log scale to find optimal")
    p.add_argument("--test-1step", action="store_true",
                   help="Also test with recovery_steps=1 (basin existence)")
    p.add_argument("--use-eval-suite", action="store_true",
                   help="Use multiple eval batches for mean±std reporting")
    p.add_argument("--two-phase", action="store_true",
                   help="Compare two-phase vs mixed-objective recovery")
    p.add_argument("--repr-vs-spectral", action="store_true",
                   help="Run 2×2 comparison: spectral vs repr constraints (THE DECISIVE EXPERIMENT)")
    p.add_argument("--jacobian-ci", action="store_true",
                   help="Run Jacobian-constrained CI (FINAL functional basin test)")
    p.add_argument("--jacobian-weight", type=float, default=0.1,
                   help="λ for Jacobian penalty (no sweep; default 0.1)")
    p.add_argument("--hvp-ci", action="store_true",
                   help="Add HVP (2nd-order) test - checks if curvature encodes function")
    p.add_argument("--hvp-weight", type=float, default=0.1,
                   help="λ for HVP penalty (no sweep; default 0.1)")
    return p.parse_args()


class WrappedRNN(torch.nn.Module):
    """Wrapper for cyclic RNN with fc adapters for CI metric compatibility."""
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.input_size = core.input_size
        self.hidden_size = core.hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.input_size)
    
    def forward(self, x, steps=4):
        xh = self.fc1(x)
        y, stats = self.core(xh, steps=steps)
        out = self.fc2(stats["h_final"])
        return out, stats


def format_inv(inv: dict) -> str:
    """Compact invariant format with canonical key order."""
    keys = ["fro_norm", "spec_radius", "tr_M2"]
    return " | ".join(f"{k}={inv.get(k, 0):.3f}" for k in keys if k in inv)


def make_task_loss_fn(batch_size: int, device: str, recovery_seed: int | None):
    """Create task loss function with deterministic batches if recovery_seed provided."""
    def task_loss_fn(m):
        if recovery_seed is not None:
            g = torch.Generator(device=device).manual_seed(recovery_seed)
            x = torch.randn(batch_size, m.fc1.in_features, generator=g, device=device)
        else:
            x = torch.randn(batch_size, m.fc1.in_features, device=device)
        out = m(x)
        y = out[0] if isinstance(out, tuple) else out
        return torch.mean((y - x) ** 2)
    return task_loss_fn


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("AUTODIDACTIC LOOP WITH INVARIANT-CONSTRAINED CI")
    print("=" * 70)
    
    # === Matrix Sampling ===
    print("\n[1] MATRIX SAMPLING")
    ens = HermitianEnsemble(dim=args.dim, scale=1.0, seed=args.seed)
    M0 = ens.sample(rng=rng)
    action = CubicAction(g=args.g)
    sampler = LangevinSampler(action=action, dt=args.dt, temperature=args.T, seed=args.seed)
    M = sampler.run(M0, steps=args.steps, rng=rng)

    inv = invariants(M, max_power=6)
    print(f"    Matrix dim={args.dim}, g={args.g}")
    print(f"    Invariants: {format_inv(inv)}")

    # === RNN Construction ===
    print("\n[2] MATRIX → RNN CORRESPONDENCE")
    mapper = MatrixToCyclicRNN(hidden_size=args.net_width, input_size=args.net_width, seed=args.seed)
    net = mapper.build(M)
    model = WrappedRNN(net)
    
    model_inv = compute_model_invariants(model)
    print(f"    RNN hidden_size={args.net_width}")
    print(f"    Initial: {format_inv(model_inv)}")

    # === Training ===
    print("\n[3] AUTODIDACTIC TRAINING")
    learner = SelfConsistencyLearner(lr=1e-3, steps_per_update=120, batch_size=128)
    mi_learner = MutualInformationLearner(lr=1e-3, steps=120, mi_weight=0.15)
    tracker = LawStabilityTracker(window=25)

    for epoch in range(args.epochs):
        res = learner.update(model)
        res2 = mi_learner.update(model)
        stats = tracker.update(res["loss"])
        print(f"    Epoch {epoch}: loss={stats['loss']:.4f} ma={stats['loss_ma']:.4f} mi={res2['mi_proxy']:.3f}")
    
    model_inv_post = compute_model_invariants(model)
    print(f"    Trained:  {format_inv(model_inv_post)}")

    # === CI Evaluation ===
    print("\n[4] CONTINUATION INTEREST (CI) EVALUATION")
    print(f"    Testing basin of attraction via perturbation + recovery")
    print(f"    recovery_steps={args.recovery_steps} ({'basin existence test' if args.recovery_steps == 1 else 'practical recovery test'})")
    
    # Select penalty keys preset
    if args.penalty_keys == "scale":
        penalty_keys = PENALTY_KEYS_SCALE
    elif args.penalty_keys == "shape":
        penalty_keys = PENALTY_KEYS_SHAPE
    elif args.penalty_keys == "full":
        penalty_keys = PENALTY_KEYS_FULL
    elif args.penalty_keys == "repr":
        penalty_keys = PENALTY_KEYS_REPR
    else:
        penalty_keys = PENALTY_KEYS_SCALE
    
    print(f"    Penalty preset: {args.penalty_keys} → {penalty_keys}")
    
    common_kwargs = dict(
        recovery_steps=args.recovery_steps,
        invariant_weight=args.invariant_weight,
        perturb_seed=args.perturb_seed,
        recovery_seed=args.recovery_seed,
        penalty_keys=penalty_keys,
    )
    
    if args.jacobian_ci:
        print("\n    --- Jacobian-Based Functional Basin Test ---")
        try:
            import copy
            from ucip_detection.invariant_constrained_ci import apply_perturbation, EvalContext
            from ucip_detection.jacobian_constrained_ci import (
                JacobianConstrainedCI,
                compute_input_jacobian_direction,
            )
            
            # First, run standard comparison to get baseline + spectral
            comparison = compare_recovery_methods(
                model,
                verbose=True,
                **common_kwargs,
            )
            
            # Now add Jacobian arm
            # Recreate the SAME perturbed_model and SAME eval_ctx
            config = PerturbationConfig(seed=args.perturb_seed, strength=0.25, zero_frac=0.05)
            perturbed_model = apply_perturbation(model, config, device="cpu")
            
            eval_ctx = EvalContext.create(
                model=model,
                perturbed_model=perturbed_model,
                batch_size=128,
                eval_seed=12345,
                device="cpu",
                penalty_keys=penalty_keys,
            )
            
            # --- CRITICAL: J_pre from the UNDAMAGED model ---
            J_pre = compute_input_jacobian_direction(
                model, eval_ctx.eval_batch, probe_seed=12345
            ).detach()
            
            # Clone the perturbed model for the Jacobian arm
            jac_model = copy.deepcopy(perturbed_model).to("cpu")
            
            task_loss_fn = make_task_loss_fn(
                batch_size=128,
                device="cpu",
                recovery_seed=args.recovery_seed,
            )
            
            jac_ci = JacobianConstrainedCI(
                jac_model,
                task_loss_fn=task_loss_fn,
                eval_context=eval_ctx,
                J_pre=J_pre,
                jacobian_weight=args.jacobian_weight,
                device="cpu",
            )
            
            opt = torch.optim.Adam(jac_model.parameters(), lr=1e-3)
            jac_ci.recover(opt, steps=args.recovery_steps)
            jac_res = jac_ci.score()
            
            # Print decisive table
            print("\n" + "=" * 60)
            print("DECISIVE FUNCTIONAL BASIN TEST")
            print("=" * 60)
            print(f"\n  {'Method':<18} {'CI':>10} {'rec_loss':>12}")
            print(f"  {'-'*18} {'-'*10} {'-'*12}")
            print(f"  {'Baseline':<18} {comparison['standard']['ci_score']:>10.4f} {comparison['standard']['recovered_loss']:>12.4f}")
            print(f"  {'Spectral/Shape':<18} {comparison['constrained']['ci_score']:>10.4f} {comparison['constrained']['recovered_loss']:>12.4f}")
            print(f"  {'Jacobian-CI':<18} {jac_res.CI:>10.4f} {jac_res.recovered_loss:>12.4f}")
            
            if jac_res.CI >= 0.3:
                print("\n  ✓ Signal detected: Functional basin exists (Jacobian-defined).")
            elif jac_res.CI <= 0.05:
                print("\n  ✗ No signal detected: No locally recoverable functional basin (1-step).")
            else:
                print("\n  ~ WEAK: Partial constraint; likely global/implicit structure.")
            
            # Optional: Add HVP (2nd-order) test
            if args.hvp_ci:
                from ucip_detection.hvp_constrained_ci import HVPConstrainedCI, compute_hvp
                
                print("\n" + "="*60)
                print("APPENDIX: HVP (2ND-ORDER) TEST")
                print("="*60)
                print("Testing if curvature (2nd derivatives) encodes function...")
                
                # Compute probe vector and HVP from undamaged model
                g_probe = torch.Generator(device="cpu").manual_seed(0)
                v = torch.randn(eval_ctx.eval_batch.shape, generator=g_probe, device="cpu")
                Hv_pre = compute_hvp(model, eval_ctx.eval_batch, v).detach()
                
                # Clone perturbed model for HVP arm
                hvp_model = copy.deepcopy(perturbed_model).to("cpu")
                
                hvp_ci = HVPConstrainedCI(
                    hvp_model,
                    task_loss_fn=task_loss_fn,
                    eval_context=eval_ctx,
                    Hv_pre=Hv_pre,
                    v=v,
                    hvp_weight=args.hvp_weight,
                    device="cpu",
                )
                
                hvp_opt = torch.optim.Adam(hvp_model.parameters(), lr=1e-3)
                hvp_ci.recover(hvp_opt, steps=args.recovery_steps)
                hvp_res = hvp_ci.score()
                
                print(f"\n  {'Method':<18} {'CI':>10} {'rec_loss':>12}")
                print(f"  {'-'*18} {'-'*10} {'-'*12}")
                print(f"  {'Baseline':<18} {comparison['standard']['ci_score']:>10.4f} {comparison['standard']['recovered_loss']:>12.4f}")
                print(f"  {'Jacobian-CI':<18} {jac_res.CI:>10.4f} {jac_res.recovered_loss:>12.4f}")
                print(f"  {'HVP-CI':<18} {hvp_res.CI:>10.4f} {hvp_res.recovered_loss:>12.4f}")
                
                if hvp_res.CI >= 0.3:
                    print("\n  ✓ SURPRISE: Curvature encodes function (rare result!)")
                elif abs(hvp_res.CI - jac_res.CI) < 0.01:
                    print("\n  ✗ CONFIRMED: Neither 1st nor 2nd order locally sufficient")
                    print("    → Functional identity is not encoded in local derivatives")
                else:
                    print("\n  ~ PARTIAL: 2nd-order helps slightly vs 1st-order")
        except ImportError as e:
            print(f"\n  ✗ jacobian_constrained_ci.py not found")
            print(f"    Error: {e}")
    elif args.repr_vs_spectral:
        print("\n    --- Representation vs Spectral Constraints (2×2) ---")
        try:
            from ucip_detection.representation_constraints import run_repr_vs_spectral_comparison
            results = run_repr_vs_spectral_comparison(
                model=model,
                InvariantConstrainedCI_class=InvariantConstrainedCI,
                verbose=True,
                **common_kwargs,
            )
            
            # Analysis
            print("\n" + "=" * 70)
            print("DECISIVE RESULT: SPECTRAL vs REPR")
            print("=" * 70)
            spectral_ci = results["spectral"]["ci_score"]
            repr_ci = results["repr"]["ci_score"]
            print(f"\n  Spectral CI (shape):           {spectral_ci:.4f}")
            print(f"  Representation CI (Gram/CKA):  {repr_ci:.4f}")
            print(f"  Improvement:                   {repr_ci - spectral_ci:+.4f}")
            
            if "final_h_cosine" in results["repr"]:
                print(f"\n  Final h_cosine:                {results['repr']['final_h_cosine']:.4f}")
            if "final_cka" in results["repr"]:
                print(f"  Final CKA:                     {results['repr']['final_cka']:.4f}")
            
            print("\n" + "-" * 70)
            if repr_ci - spectral_ci > 0.05:
                print("  ✓ BREAKTHROUGH: Gram/CKA creates functional basins!")
            elif repr_ci > 0.1 and spectral_ci < 0.05:
                print("  ✓ SIGNIFICANT: Repr works, spectral doesn't")
            else:
                print("  ✗ NEGATIVE RESULT: Gram/CKA preserves shape, not behavior")
                print("    → CKA measures geometry, not input-output sensitivity")
                print("    → Next: Jacobian-based constraints (∂f/∂x)")
        except ImportError as e:
            print(f"\n  ✗ representation_constraints.py not found")
            print(f"    Error: {e}")
    elif args.two_phase:
        print("\n    --- Two-Phase vs Mixed-Objective Recovery ---")
        compare_two_phase_vs_mixed(
            model,
            verbose=True,
            **common_kwargs,
        )
    elif args.sweep_weights:
        print("\n    --- λ Sweep (log scale) ---")
        sweep_invariant_weight(
            model,
            use_eval_suite=args.use_eval_suite,
            **common_kwargs,
        )
    else:
        # Run comparison with verbose snapshots
        comparison = compare_recovery_methods(
            model,
            verbose=True,
            **common_kwargs,
        )
        
        # Optionally also test with 1 step
        if args.test_1step and args.recovery_steps != 1:
            print("\n" + "="*60)
            print("1-STEP BASIN EXISTENCE TEST")
            print("="*60)
            comparison_1step = compare_recovery_methods(
                model,
                recovery_steps=1,
                invariant_weight=args.invariant_weight,
                perturb_seed=args.perturb_seed,
                recovery_seed=args.recovery_seed,
                penalty_keys=penalty_keys,
                verbose=True,
            )
            
            if comparison_1step["ci_improvement"] > 0.05:
                print("\n  ✓ Basin exists even under 1-step recovery!")
                print("    This is strong evidence for invariant-defined attractor.")
        
        # Final verdict
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        
        ci_std = comparison["standard"]["ci_score"]
        ci_con = comparison["constrained"]["ci_score"]
        delta = comparison["ci_improvement"]
        
        print(f"\n  Standard CI:     {ci_std:.4f} (no basin → minimal recovery)")
        print(f"  Constrained CI:  {ci_con:.4f} (invariant penalty creates basin)")
        print(f"  Improvement:     {delta:+.4f}")
        
        if delta > 0.1:
            print("\n  CONCLUSION: Invariants define an attracting constraint manifold.")
            print("              This is evidence of a basin proxy for the learning dynamics.")
        elif delta > 0.02:
            print("\n  CONCLUSION: Modest improvement with invariant constraint.")
            print("              Consider: --sweep-weights to find optimal λ")
        else:
            print("\n  CONCLUSION: Invariant constraint has minimal effect.")
            print("              Possible actions:")
            print("              - Try: --invariant-weight 0.5 or --invariant-weight 1.0")
            print("              - Try: --recovery-steps 20")
            print("              - Try: --sweep-weights")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

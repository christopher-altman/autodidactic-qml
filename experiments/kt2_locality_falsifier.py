#!/usr/bin/env python3
"""KT-2 Protocol Runner: Local Recoverability of Functional Identity

This script implements the KT-2 pre-registered falsification protocol.
It tests whether functional identity is locally encoded by measuring
whether geometric proxy constraints enable single-step recovery.

Protocol: docs/kt2_protocol.md

Usage:
    # Run decisive 1-step test (all constraint families)
    python -m experiments.kt2_locality_falsifier --run-decisive
    
    # Run k-step curve
    python -m experiments.kt2_locality_falsifier --k-step-curve
    
    # Run hysteresis measurement
    python -m experiments.kt2_locality_falsifier --hysteresis
    
    # Full protocol (all diagnostics)
    python -m experiments.kt2_locality_falsifier --full-protocol
    
    # Specify output directory
    python -m experiments.kt2_locality_falsifier --full-protocol --output-dir results/kt2/

Outputs:
    - kt2_decisive_1step.json: Decisive 1-step CI table
    - kt2_k_step_curves.json: CI(k) for k ∈ {1,2,4,8,16}
    - kt2_hysteresis.json: Forward/reverse sweep with area
    - kt2_distance_triads.json: Parameter + representation + functional distances
    - kt2_full_protocol.json: Complete protocol output with verdict
"""

from __future__ import annotations
import argparse
import json
import os
import hashlib
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import CubicAction
from matrix_models.sampler_langevin import LangevinSampler
from correspondence_maps.matrix_to_rnn import MatrixToCyclicRNN
from autodidactic_protocols.unsupervised_update import SelfConsistencyLearner
from ucip_detection.invariant_constrained_ci import (
    InvariantConstrainedCI,
    compare_recovery_methods,
    compute_model_invariants,
    PerturbationConfig,
    apply_perturbation,
    EvalContext,
    PENALTY_KEYS_SCALE,
    PENALTY_KEYS_SHAPE,
)
from ucip_detection.nonlocality_probe import (
    compute_k_step_curve,
    compare_constraint_families,
    compute_hysteresis,
    run_nonlocality_probe,
    DistanceTriad,
    KStepCurveResult,
    HysteresisResult,
    compute_step_size_envelope,
    compute_step_size_envelope_all_constraints,
)


# Protocol constants (LOCKED - do not modify without incrementing protocol version)
PROTOCOL_ID = "KT-2"
PROTOCOL_VERSION = "1.0"
PROTOCOL_DATE = "2025-12-20"

# Pre-registered seeds
PERTURB_SEED = 42
EVAL_SEED = 12345
RECOVERY_SEED = 2025
ENSEMBLE_SEEDS = [42, 137, 256, 314, 999]

# Pre-registered parameters
DEFAULT_DIM = 12
DEFAULT_HIDDEN = 64
DEFAULT_EPOCHS = 6
DEFAULT_PERTURB_STRENGTH = 0.25
DEFAULT_ZERO_FRAC = 0.05
DEFAULT_LR = 1e-3
DEFAULT_INVARIANT_WEIGHT = 0.1

# Falsification threshold
CI_THRESHOLD = 0.10


def get_git_hash() -> str:
    """Get current git commit hash for artifact binding."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


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


def create_model(seed: int = 0, dim: int = DEFAULT_DIM, hidden: int = DEFAULT_HIDDEN) -> torch.nn.Module:
    """Create and train autodidactic model."""
    rng = np.random.default_rng(seed)
    
    # Matrix sampling
    ens = HermitianEnsemble(dim=dim, scale=1.0, seed=seed)
    M0 = ens.sample(rng=rng)
    action = CubicAction(g=0.1)
    sampler = LangevinSampler(action=action, dt=1e-3, temperature=1e-2, seed=seed)
    M = sampler.run(M0, steps=800, rng=rng)
    
    # Matrix → RNN
    mapper = MatrixToCyclicRNN(hidden_size=hidden, input_size=hidden, seed=seed)
    net = mapper.build(M)
    model = WrappedRNN(net)
    
    # Train
    learner = SelfConsistencyLearner(lr=1e-3, batch_size=64)
    for _ in range(DEFAULT_EPOCHS):
        learner.update(model)
    
    return model


def run_decisive_1step(
    model: torch.nn.Module,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run decisive 1-step test across all constraint families.
    
    This is the primary falsification test for KT-2.
    """
    if verbose:
        print("\n" + "="*70)
        print("KT-2 DECISIVE 1-STEP TEST")
        print("="*70)
    
    results = compare_constraint_families(
        model,
        k_values=[1],  # Decisive test is k=1 only
        perturb_seed=PERTURB_SEED,
        eval_seed=EVAL_SEED,
        recovery_seed=RECOVERY_SEED,
        perturb_strength=DEFAULT_PERTURB_STRENGTH,
        zero_frac=DEFAULT_ZERO_FRAC,
        lr=DEFAULT_LR,
        invariant_weight=DEFAULT_INVARIANT_WEIGHT,
        verbose=verbose,
    )
    
    # Build decisive table
    decisive_table = {}
    all_below_threshold = True
    
    for constraint, result in results.items():
        ci_1 = result.ci_values[0] if result.ci_values else 0.0
        decisive_table[constraint] = {
            "ci_1step": ci_1,
            "below_threshold": ci_1 < CI_THRESHOLD,
        }
        if ci_1 >= CI_THRESHOLD:
            all_below_threshold = False
    
    # Verdict
    if all_below_threshold:
        verdict = "FALSIFIED"
        verdict_text = "H1 (Naive Locality) is FALSIFIED: All CI(k=1) < 0.10"
    else:
        verdict = "NOT_FALSIFIED"
        verdict_text = "H1 (Naive Locality) NOT falsified: At least one CI(k=1) >= 0.10"
    
    if verbose:
        print(f"\n  VERDICT: {verdict}")
        print(f"  {verdict_text}")
    
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "test": "decisive_1step",
        "git_hash": get_git_hash(),
        "timestamp": datetime.now().isoformat(),
        "seeds": {
            "perturb": PERTURB_SEED,
            "eval": EVAL_SEED,
            "recovery": RECOVERY_SEED,
        },
        "threshold": CI_THRESHOLD,
        "decisive_table": decisive_table,
        "verdict": verdict,
        "verdict_text": verdict_text,
    }


def run_k_step_curve(
    model: torch.nn.Module,
    k_values: List[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run k-step CI curve for nonlocality signature."""
    if k_values is None:
        k_values = [1, 2, 4, 8, 16]
    
    if verbose:
        print("\n" + "="*70)
        print("KT-2 k-STEP CI CURVE")
        print("="*70)
    
    results = compare_constraint_families(
        model,
        k_values=k_values,
        perturb_seed=PERTURB_SEED,
        eval_seed=EVAL_SEED,
        recovery_seed=RECOVERY_SEED,
        perturb_strength=DEFAULT_PERTURB_STRENGTH,
        zero_frac=DEFAULT_ZERO_FRAC,
        lr=DEFAULT_LR,
        invariant_weight=DEFAULT_INVARIANT_WEIGHT,
        verbose=verbose,
    )
    
    # Convert to serializable format
    curves = {}
    for constraint, result in results.items():
        curves[constraint] = {
            "k_values": result.k_values,
            "ci_values": result.ci_values,
            "triads": [t.to_dict() for t in result.distance_triads],
            "local_recovery": result.is_locally_recoverable(),
            "recovery_onset_k": result.recovery_onset_k(),
            "saturation_ci": result.saturation_ci(),
        }
    
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "test": "k_step_curve",
        "git_hash": get_git_hash(),
        "timestamp": datetime.now().isoformat(),
        "seeds": {
            "perturb": PERTURB_SEED,
            "eval": EVAL_SEED,
            "recovery": RECOVERY_SEED,
        },
        "k_values": k_values,
        "curves": curves,
    }


def run_hysteresis(
    model: torch.nn.Module,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run hysteresis measurement for path dependence."""
    if verbose:
        print("\n" + "="*70)
        print("KT-2 HYSTERESIS MEASUREMENT")
        print("="*70)
    
    g_values = np.linspace(0.0, 0.5, 11)
    
    result = compute_hysteresis(
        model,
        g_values=g_values,
        constraint_type="shape",
        recovery_steps=4,
        base_seed=PERTURB_SEED,
        eval_seed=EVAL_SEED,
        lr=DEFAULT_LR,
        invariant_weight=DEFAULT_INVARIANT_WEIGHT,
        verbose=verbose,
    )
    
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "test": "hysteresis",
        "git_hash": get_git_hash(),
        "timestamp": datetime.now().isoformat(),
        "g_values": result.g_values.tolist(),
        "forward_ci": result.forward_ci.tolist(),
        "reverse_ci": result.reverse_ci.tolist(),
        "hysteresis_area": result.hysteresis_area,
        "max_gap": result.max_gap,
        "has_hysteresis": result.has_hysteresis(),
    }


def run_full_protocol(
    model: torch.nn.Module,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run complete KT-2 protocol with all diagnostics."""
    if verbose:
        print("\n" + "="*70)
        print("KT-2 FULL PROTOCOL: LOCAL RECOVERABILITY OF FUNCTIONAL IDENTITY")
        print("="*70)
        print(f"\n  Protocol: {PROTOCOL_ID} v{PROTOCOL_VERSION}")
        print(f"  Date: {PROTOCOL_DATE}")
        print(f"  Git hash: {get_git_hash()}")
        print(f"  Seeds: perturb={PERTURB_SEED}, eval={EVAL_SEED}, recovery={RECOVERY_SEED}")
    
    # Run all tests
    decisive = run_decisive_1step(model, verbose=verbose)
    k_step = run_k_step_curve(model, verbose=verbose)
    hysteresis = run_hysteresis(model, verbose=verbose)
    
    # Compute overall verdict
    h1_falsified = decisive["verdict"] == "FALSIFIED"
    has_path_dependence = hysteresis["has_hysteresis"]
    
    if h1_falsified and has_path_dependence:
        overall_verdict = "STRONGLY_NONLOCAL"
        overall_text = "Function is fundamentally nonlocal with path dependence"
    elif h1_falsified:
        overall_verdict = "NONLOCAL"
        overall_text = "Function is nonlocal (all geometric proxies fail)"
    elif has_path_dependence:
        overall_verdict = "PATH_DEPENDENT"
        overall_text = "Recovery is path-dependent (hysteresis detected)"
    else:
        overall_verdict = "LOCALLY_RECOVERABLE"
        overall_text = "Function is locally recoverable"
    
    if verbose:
        print("\n" + "="*70)
        print("PROTOCOL SUMMARY")
        print("="*70)
        print(f"\n  H1 (Naive Locality): {decisive['verdict']}")
        print(f"  Hysteresis: {'DETECTED' if has_path_dependence else 'NOT DETECTED'}")
        print(f"\n  OVERALL VERDICT: {overall_verdict}")
        print(f"  {overall_text}")
    
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_date": PROTOCOL_DATE,
        "git_hash": get_git_hash(),
        "timestamp": datetime.now().isoformat(),
        "seeds": {
            "perturb": PERTURB_SEED,
            "eval": EVAL_SEED,
            "recovery": RECOVERY_SEED,
        },
        "decisive_1step": decisive,
        "k_step_curves": k_step,
        "hysteresis": hysteresis,
        "overall_verdict": overall_verdict,
        "overall_verdict_text": overall_text,
    }


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="KT-2 Protocol: Local Recoverability of Functional Identity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Test selection
    parser.add_argument("--run-decisive", action="store_true",
                        help="Run decisive 1-step test (primary falsification)")
    parser.add_argument("--one-step-only", action="store_true",
                        help="Alias for --run-decisive (run only the 1-step test)")
    parser.add_argument("--k-step-curve", action="store_true",
                        help="Run k-step CI curve for nonlocality signature")
    parser.add_argument("--hysteresis", action="store_true",
                        help="Run hysteresis measurement for path dependence")
    parser.add_argument("--step-envelope", action="store_true",
                        help="Run step-size envelope (best 1-step CI over η grid)")
    parser.add_argument("--full-protocol", action="store_true",
                        help="Run complete protocol with all diagnostics")
    
    # Model parameters
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"Matrix dimension (default: {DEFAULT_DIM})")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN,
                        help=f"Hidden size (default: {DEFAULT_HIDDEN})")
    parser.add_argument("--model-seed", type=int, default=0,
                        help="Seed for model initialization/training (default: 0)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/kt2",
                        help="Output directory for artifacts")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    if args.one_step_only:
        args.run_decisive = True
        args.k_step_curve = False
        args.hysteresis = False
        args.step_envelope = False
        args.full_protocol = False

    # Default to full protocol if nothing specified
    if not any([args.run_decisive, args.k_step_curve, args.hysteresis, args.step_envelope, args.full_protocol]):
        args.full_protocol = True
    
    verbose = not args.quiet
    
    # Create model
    if verbose:
        print("\n[1] Creating and training model...")
    model = create_model(seed=args.model_seed, dim=args.dim, hidden=args.hidden)
    
    # Run selected tests
    if args.full_protocol:
        results = run_full_protocol(model, verbose=verbose)
        save_results(results, os.path.join(args.output_dir, "kt2_full_protocol.json"))
    else:
        if args.run_decisive:
            results = run_decisive_1step(model, verbose=verbose)
            save_results(results, os.path.join(args.output_dir, "kt2_decisive_1step.json"))
        
        if args.k_step_curve:
            results = run_k_step_curve(model, verbose=verbose)
            save_results(results, os.path.join(args.output_dir, "kt2_k_step_curves.json"))
        
        if args.hysteresis:
            results = run_hysteresis(model, verbose=verbose)
            save_results(results, os.path.join(args.output_dir, "kt2_hysteresis.json"))
        
        if args.step_envelope:
            if verbose:
                print("\n" + "="*70)
                print("KT-2 STEP-SIZE ENVELOPE TEST")
                print("="*70)
            
            envelope_results = compute_step_size_envelope_all_constraints(
                model,
                perturb_seed=PERTURB_SEED,
                eval_seed=EVAL_SEED,
                recovery_seed=RECOVERY_SEED,
                perturb_strength=DEFAULT_PERTURB_STRENGTH,
                zero_frac=DEFAULT_ZERO_FRAC,
                invariant_weight=DEFAULT_INVARIANT_WEIGHT,
                verbose=verbose,
            )
            
            results = {
                "protocol_id": PROTOCOL_ID,
                "protocol_version": PROTOCOL_VERSION,
                "test": "step_envelope",
                "git_hash": get_git_hash(),
                "timestamp": datetime.now().isoformat(),
                "seeds": {
                    "perturb": PERTURB_SEED,
                    "eval": EVAL_SEED,
                    "recovery": RECOVERY_SEED,
                },
                "envelopes": {k: v.to_dict() for k, v in envelope_results.items()},
                "all_fail": all(not r.is_locally_recoverable() for r in envelope_results.values()),
            }
            save_results(results, os.path.join(args.output_dir, "kt2_step_envelope.json"))
    
    if verbose:
        print("\n" + "="*70)
        print("PROTOCOL COMPLETE")
        print("="*70)


if __name__ == "__main__":
    main()

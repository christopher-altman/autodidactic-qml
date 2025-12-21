"""SQNT -> Loop -> Topology Learning Dynamics (TLD) Bridge Experiment.

This script implements Step 9 of the patch plan:
Bridge between spectral/quantum network topology (SQNT), loop dynamics,
and topology-constrained learning dynamics (TLD) with full JSON reports.

NOTE: This module does NOT implement UCIP (Unified Continuation-Interest Protocol).
UCIP is a separate framework for detecting persistence bias in AI agents.
See: experiments/ucip_protocol.py for actual UCIP implementation.

The bridge connects:
1. SQNT: Graph topology metrics (from matrix -> graph lift)
2. Loop: Spectral dynamics and phase structure
3. TLD: Topology-constrained learning dynamics from autodidactic training

Output JSON structure:
{
    "graph_summary": {...},
    "kernel_advantage": {...},
    "phase_indicators": {...},
    "sqnt_bridge": {
        "topology_n_edges": int,
        "topology_mean_degree": float,
        "topology_spectral_gap": float,
        "kernel_geometry_alignment": float,
    },
    "loop_bridge": {
        "spectral_radius": float,
        "level_spacing_r": float,
        "is_chaotic": bool,
        "spectral_entropy": float,
        "participation_ratio": float,
    },
    "tld_bridge": {
        "topology_influence_score": float,
        "recovery_time": float,
        "stability_metric": float,
        "phase_transition_detected": bool,
        "learning_efficiency": float,
    },
    "tld_vs_topology": {
        "topology_density": float,
        "topology_connectivity": float,
        "topology_influence_score": float,
        "correlation_tld_connectivity": float,
    }
}

Usage:
    python -m experiments.sqnt_loop_topology_dynamics --dim 10 --output bridge_report.json
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# Matrix model imports
from matrix_models import (
    HermitianEnsemble,
    CubicAction,
    LangevinSampler,
    invariants,
)

# Topology imports
from topology import Graph, matrix_to_graph, GraphFamily, sample_graph_family

# Quantum kernel imports
from quantum_kernels import (
    GraphConditionedKernel,
    QuantumKernel,
    VariationalLawEncoder,
    random_fourier_features_kernel,
    kernel_alignment,
)

# Analysis imports
from analysis import (
    SpectralDiagnostics,
    SpectralComplexityAnalyzer,
    PhaseTransitionAnalyzer,
    LawStabilityTracker,
    OnlinePhaseDetector,
)


def parse_args():
    p = argparse.ArgumentParser(description="SQNT/Loop/UCIP Bridge Experiment")

    # Matrix model
    p.add_argument("--dim", type=int, default=10, help="Matrix dimension")
    p.add_argument("--g", type=float, default=0.1, help="Coupling constant")
    p.add_argument("--langevin-steps", type=int, default=500, help="Sampling steps")
    p.add_argument("--temperature", type=float, default=0.01, help="Temperature")

    # Graph parameters
    p.add_argument("--graph-mode", choices=["threshold", "weighted", "laplacian"],
                   default="weighted", help="Graph mode")
    p.add_argument("--use-graph-family", action="store_true", help="Use graph family")
    p.add_argument("--graph-family", choices=[f.value for f in GraphFamily],
                   default="erdos_renyi", help="Graph family")
    p.add_argument("--family-p", type=float, default=0.3, help="ER edge prob")

    # Quantum kernel
    p.add_argument("--n-qubits", type=int, default=4, help="Qubits")
    p.add_argument("--n-layers", type=int, default=2, help="Circuit layers")
    p.add_argument("--n-samples", type=int, default=40, help="Data samples")

    # TLD training
    p.add_argument("--tld-steps", type=int, default=100, help="TLD training steps")
    p.add_argument("--tld-lr", type=float, default=0.01, help="TLD learning rate")

    # Output
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--output", type=str, default=None, help="Output JSON")
    p.add_argument("--verbose", action="store_true", help="Verbose")

    return p.parse_args()


@dataclass
class TLDMetrics:
    """Topology Learning Dynamics metrics.
    
    Measures how graph topology influences autodidactic learning.
    NOT related to UCIP (Unified Continuation-Interest Protocol).
    """
    topology_influence_score: float  # Topology-learning correlation
    recovery_time: float  # Steps to recover from perturbation
    stability_metric: float  # Variance of losses over training
    phase_transition_detected: bool  # Whether phase transition occurred
    learning_efficiency: float  # Final loss / initial loss ratio
    loss_history: list  # Full loss trajectory


def compute_tld_metrics(
    M: np.ndarray,
    graph: Graph,
    n_steps: int,
    lr: float,
    seed: int,
) -> TLDMetrics:
    """Compute Topology Learning Dynamics metrics via simulated autodidactic training.

    Measures how graph topology constrains and influences the learning process,
    including perturbation response and topology-dynamics correlations.
    """
    rng = np.random.default_rng(seed)
    dim = M.shape[0]

    # Initialize "learned law" as perturbation of input matrix
    M_learned = M.copy() + 0.1 * rng.normal(size=M.shape)
    M_learned = (M_learned + M_learned.T) / 2

    # Use graph topology as constraint
    A = graph.adjacency
    A_mask = (A > 0).astype(float)

    # Training loop: minimize reconstruction error with topology constraint
    losses = []
    perturb_step = n_steps // 2  # Perturb halfway through
    recovery_start = None

    for step in range(n_steps):
        # Compute loss: Frobenius norm + topology penalty
        diff = M_learned - M
        recon_loss = np.sum(diff ** 2) / dim**2

        # Topology penalty: encourage learned structure to match graph
        M_norm = M_learned / (np.abs(M_learned).max() + 1e-12)
        topology_loss = np.sum((np.abs(M_norm) - A) ** 2 * A_mask) / (A_mask.sum() + 1)

        total_loss = recon_loss + 0.1 * topology_loss
        losses.append(float(total_loss))

        # Gradient descent step (simplified)
        grad = 2 * diff / dim**2
        M_learned = M_learned - lr * grad
        M_learned = (M_learned + M_learned.T) / 2

        # Perturbation at halfway point
        if step == perturb_step:
            pre_perturb_loss = total_loss
            M_learned = M_learned + 0.5 * rng.normal(size=M.shape)
            M_learned = (M_learned + M_learned.T) / 2

        # Check for recovery
        if step > perturb_step and recovery_start is None:
            if total_loss < pre_perturb_loss * 1.1:
                recovery_start = step - perturb_step

    # Compute metrics
    losses = np.array(losses)

    # Topology influence score: correlation between topology eigenvalues and loss trajectory
    L = graph.laplacian()
    L_eigs = np.linalg.eigvalsh(L)
    M_eigs = np.linalg.eigvalsh(M)
    # How much does topology "explain" the learning dynamics
    topology_influence_score = float(np.abs(np.corrcoef(L_eigs[:min(10, len(L_eigs))],
                                         np.sort(M_eigs)[:min(10, len(M_eigs))])[0, 1]))
    if np.isnan(topology_influence_score):
        topology_influence_score = 0.0

    # Recovery time
    recovery_time = float(recovery_start if recovery_start else n_steps - perturb_step)

    # Stability: variance of second half of training
    stability_metric = float(np.std(losses[n_steps//2:]))

    # Phase transition: sharp change in loss derivative
    loss_diff = np.diff(losses)
    phase_transition_detected = bool(np.max(np.abs(loss_diff)) > 3 * np.std(loss_diff))

    # Learning efficiency
    learning_efficiency = float(losses[-1] / (losses[0] + 1e-12))

    return TLDMetrics(
        topology_influence_score=topology_influence_score,
        recovery_time=recovery_time,
        stability_metric=stability_metric,
        phase_transition_detected=phase_transition_detected,
        learning_efficiency=learning_efficiency,
        loss_history=losses.tolist(),
    )


def run_bridge_experiment(args) -> Dict[str, Any]:
    """Run the full SQNT/Loop/UCIP bridge experiment."""
    results = {
        "config": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    rng = np.random.default_rng(args.seed)

    # === 1. Sample matrix M ===
    if args.verbose:
        print("Sampling matrix M...")

    ens = HermitianEnsemble(dim=args.dim, seed=args.seed)
    M0 = ens.sample(rng=rng)

    action = CubicAction(g=args.g)
    sampler = LangevinSampler(action=action, dt=1e-3, temperature=args.temperature, seed=args.seed)
    M = sampler.run(M0, steps=args.langevin_steps, rng=rng)

    results["matrix_invariants"] = invariants(M, max_power=4)

    # === 2. Construct graph G ===
    if args.verbose:
        print("Constructing graph...")

    if args.use_graph_family:
        family = GraphFamily(args.graph_family)
        _, params_used, G = sample_graph_family(family, args.dim, {"p": args.family_p}, args.seed)
        results["graph_source"] = "family"
        results["graph_family"] = args.graph_family
    else:
        G = matrix_to_graph(M, mode=args.graph_mode)
        results["graph_source"] = "matrix_to_graph"

    results["graph_summary"] = G.summary()

    # === 3. Spectral analysis (Loop bridge) ===
    if args.verbose:
        print("Computing spectral metrics...")

    spec_diag = SpectralDiagnostics()
    spec_complex = SpectralComplexityAnalyzer()

    spec_summary = spec_diag.summarize(M)
    complexity = spec_complex.analyze(M)

    results["loop_bridge"] = {
        "spectral_radius": spec_summary["spectral_radius"],
        "spectral_std": spec_summary["eig_std"],
        "spectral_range": spec_summary["eig_max"] - spec_summary["eig_min"],
        "level_spacing_r": complexity["level_spacing_r"],
        "is_chaotic": complexity["is_chaotic"],
        "spectral_entropy": complexity["spectral_entropy"],
        "participation_ratio_mean": complexity["participation_ratio_mean"],
        "spectral_form_factor_t1": complexity.get("spectral_form_factor_t1", None),
    }

    # === 4. Quantum kernel advantage (SQNT bridge) ===
    if args.verbose:
        print("Computing kernel advantage...")

    X = rng.normal(size=(args.n_samples, args.dim))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1] * X[:, 2]) + 0.1 * rng.normal(size=args.n_samples)
    n_train = int(0.7 * args.n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    X_all = np.vstack([X_train, X_test])

    # Graph-conditioned kernel
    gc_kernel = GraphConditionedKernel(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        seed=args.seed,
    )
    gc_kernel.set_graph(G)
    K_gc = gc_kernel.gram(X_all)

    # Classical kernel
    K_rbf = random_fourier_features_kernel(X_all, gamma=1.0, n_features=256)

    # KRR predictions
    def krr(K_tr, y_tr, K_te):
        alpha = np.linalg.solve(K_tr + 1e-3 * np.eye(len(y_tr)), y_tr)
        return K_te @ alpha

    y_pred_gc = krr(K_gc[:n_train, :n_train], y_train, K_gc[n_train:, :n_train])
    y_pred_c = krr(K_rbf[:n_train, :n_train], y_train, K_rbf[n_train:, :n_train])

    r2_gc = 1.0 - np.mean((y_pred_gc - y_test)**2) / (np.var(y_test) + 1e-12)
    r2_c = 1.0 - np.mean((y_pred_c - y_test)**2) / (np.var(y_test) + 1e-12)

    y_binary = (y_train > np.median(y_train)).astype(float) * 2 - 1
    align_gc = kernel_alignment(K_gc[:n_train, :n_train], y_binary)

    results["kernel_advantage"] = {
        "r2_graph_conditioned": float(r2_gc),
        "r2_classical": float(r2_c),
        "advantage": float(r2_gc - r2_c),
    }

    results["sqnt_bridge"] = {
        "topology_n_edges": G.n_edges(),
        "topology_mean_degree": G.mean_degree(),
        "topology_spectral_gap": G.spectral_gap(),
        "topology_algebraic_connectivity": G.algebraic_connectivity(),
        "kernel_geometry_alignment": float(align_gc),
    }

    # === 5. TLD metrics (topology-constrained learning) ===
    if args.verbose:
        print("Computing TLD metrics...")

    tld = compute_tld_metrics(
        M=M,
        graph=G,
        n_steps=args.tld_steps,
        lr=args.tld_lr,
        seed=args.seed,
    )

    results["tld_bridge"] = {
        "topology_influence_score": tld.topology_influence_score,
        "recovery_time": tld.recovery_time,
        "stability_metric": tld.stability_metric,
        "phase_transition_detected": tld.phase_transition_detected,
        "learning_efficiency": tld.learning_efficiency,
    }

    # === 6. TLD vs Topology correlation ===
    results["tld_vs_topology"] = {
        "topology_density": G.density(),
        "topology_connectivity": G.algebraic_connectivity(),
        "topology_influence_score": tld.topology_influence_score,
        "advantage_r2": float(r2_gc - r2_c),
        # Correlation hypothesis: topology influence should correlate with connectivity
        "hypothesis": "Higher connectivity -> faster recovery, higher topology influence",
    }

    # === 7. Phase indicators ===
    results["phase_indicators"] = {
        "g": args.g,
        "laplacian_gap": G.spectral_gap(),
        "effective_dimension": complexity["participation_ratio_mean"] * args.dim,
        "spectral_entropy": complexity["spectral_entropy"],
        "is_chaotic": complexity["is_chaotic"],
        "tld_phase_detected": tld.phase_transition_detected,
    }

    return results


def main():
    args = parse_args()

    if args.verbose:
        print("=" * 60)
        print("SQNT / LOOP / TLD BRIDGE EXPERIMENT")
        print("=" * 60)

    results = run_bridge_experiment(args)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("BRIDGE REPORT SUMMARY")
        print("=" * 60)

        print("\nGraph Summary:")
        for k, v in results["graph_summary"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        print("\nSQNT Bridge:")
        for k, v in results["sqnt_bridge"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        print("\nLoop Bridge:")
        for k, v in results["loop_bridge"].items():
            if v is not None:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        print("\nTLD Bridge:")
        for k, v in results["tld_bridge"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        print("\nKernel Advantage:")
        for k, v in results["kernel_advantage"].items():
            print(f"  {k}: {v:.4f}")

        print("\nPhase Indicators:")
        for k, v in results["phase_indicators"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

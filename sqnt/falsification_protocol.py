"""KT-1 Falsification Protocol: Topology–Perturbation Memory Test.

This module implements the pre-registered falsification test for SQNT.
The test is designed to be lethal: if SQNT fails, it deserves to fail.

HYPOTHESIS (SQNT):
    Adaptive topology under a conserved spectral-mass invariant produces
    history-dependent recovery dynamics following structural perturbation,
    not reproducible by fixed or random topology controls.

EXPERIMENTAL CONDITIONS:
    System A — SQNT (Experimental)
        • Topology plasticity enabled
        • Participation-based edge updates
        • Soft spectral-mass invariant enforced
        • Topology feeds back into matrix update
    
    System B — Frozen Topology (Control 1)
        • Topology extracted at t=0
        • Topology fixed for all t
        • Same invariant enforcement
        • Same matrix update rule
    
    System C — Randomized Topology (Control 2)
        • Topology randomly rewired each step
        • Degree/weight distribution preserved
        • Same invariant enforcement
        • Same matrix update rule

PROTOCOL:
    1. Initialize all systems from identical M_0
    2. Evolve to baseline regime (stationary or cyclic observables)
    3. Apply identical perturbation at time T:
       - Remove 25% of edges in a contiguous subgraph
    4. Continue evolution for ΔT

OBSERVABLES (logged):
    • Spectral recovery distance
    • Recovery time constant
    • Post-recovery topology similarity
    • Perturbation-location dependence

FALSIFICATION CRITERIA (ANY → SQNT DOES NOT HOLD):
    1. No significant difference between System A and B
    2. System A statistically indistinguishable from C
    3. No topology memory after recovery
    4. Invariant-only dynamics explain all behavior

This is now LOCKED. No post-hoc edits.

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Callable
import json
from datetime import datetime

from .sqnt_update import (
    SQNTSystem,
    SQNTConfig,
    SystemMode,
    run_comparison_experiment,
)


@dataclass
class FalsificationResult:
    """Result of a single falsification test run.
    
    Attributes:
        passed: Whether SQNT passed this test
        failure_reasons: List of failed criteria (empty if passed)
        metrics: All computed metrics
        timestamp: When the test was run
    """
    passed: bool
    failure_reasons: List[str]
    metrics: Dict[str, any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KT1Protocol:
    """KT-1: Topology–Perturbation Memory Test.
    
    This implements the pre-registered falsification protocol for SQNT.
    
    Usage:
        protocol = KT1Protocol(
            n_warmup=50,
            n_post_perturbation=100,
            perturbation_fraction=0.25,
        )
        
        result = protocol.run(
            M0=initial_matrix,
            A0=initial_topology,
            dynamics=my_dynamics_fn,
        )
        
        if not result.passed:
            print("SQNT FALSIFIED:", result.failure_reasons)
    """
    
    # Protocol parameters (pre-registered, do not change after publication)
    n_warmup: int = 50
    n_post_perturbation: int = 100
    perturbation_fraction: float = 0.25
    perturbation_mode: str = 'contiguous'
    
    # Significance thresholds
    advantage_threshold: float = 0.1  # Minimum relative improvement required
    similarity_threshold: float = 0.8  # Minimum topology memory required
    statistical_significance: float = 0.05  # p-value threshold
    
    # Random seeds for reproducibility
    n_seeds: int = 5
    base_seed: int = 42
    
    def run(
        self,
        M0: np.ndarray,
        A0: np.ndarray,
        dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
        verbose: bool = True,
    ) -> FalsificationResult:
        """Run the full KT-1 falsification protocol.
        
        Args:
            M0: Initial matrix
            A0: Initial topology
            dynamics: Dynamics function (M, A) -> delta_M
            verbose: Print progress
            
        Returns:
            FalsificationResult with pass/fail and all metrics
        """
        if verbose:
            print("=" * 60)
            print("KT-1 FALSIFICATION PROTOCOL")
            print("=" * 60)
            print(f"Warmup steps: {self.n_warmup}")
            print(f"Post-perturbation steps: {self.n_post_perturbation}")
            print(f"Perturbation fraction: {self.perturbation_fraction}")
            print(f"Seeds: {self.n_seeds}")
            print("=" * 60)
        
        # Run experiments across seeds
        all_results = []
        
        for seed_idx in range(self.n_seeds):
            seed = self.base_seed + seed_idx
            if verbose:
                print(f"\n--- Seed {seed} ({seed_idx + 1}/{self.n_seeds}) ---")
            
            results = run_comparison_experiment(
                M0=M0,
                A0=A0,
                dynamics=dynamics,
                n_steps=self.n_warmup + self.n_post_perturbation,
                perturb_at=self.n_warmup,
                perturb_fraction=self.perturbation_fraction,
                seed=seed,
            )
            all_results.append(results)
        
        # Aggregate metrics across seeds
        metrics = self._aggregate_metrics(all_results)
        
        # Apply falsification criteria
        failure_reasons = self._check_criteria(metrics, verbose)
        
        passed = len(failure_reasons) == 0
        
        if verbose:
            print("\n" + "=" * 60)
            if passed:
                print("RESULT: SQNT SURVIVES")
            else:
                print("RESULT: SQNT FALSIFIED")
                for reason in failure_reasons:
                    print(f"  ❌ {reason}")
            print("=" * 60)
        
        return FalsificationResult(
            passed=passed,
            failure_reasons=failure_reasons,
            metrics=metrics,
        )
    
    def _aggregate_metrics(
        self,
        all_results: List[Dict[str, Dict]],
    ) -> Dict[str, any]:
        """Aggregate metrics across seeds."""
        
        def extract_recovery(results, mode):
            rm = results[mode].get('recovery_metrics')
            if rm is None:
                return {}
            return rm
        
        # Recovery distances
        sqnt_distances = []
        frozen_distances = []
        random_distances = []
        
        sqnt_topology_sim = []
        frozen_topology_sim = []
        random_topology_sim = []
        
        for results in all_results:
            sqnt_rm = extract_recovery(results, 'sqnt')
            frozen_rm = extract_recovery(results, 'frozen')
            random_rm = extract_recovery(results, 'random')
            
            if 'spectral_distance_L' in sqnt_rm:
                sqnt_distances.append(sqnt_rm['spectral_distance_L'])
                frozen_distances.append(frozen_rm['spectral_distance_L'])
                random_distances.append(random_rm['spectral_distance_L'])
                
                sqnt_topology_sim.append(sqnt_rm['topology_similarity'])
                frozen_topology_sim.append(frozen_rm['topology_similarity'])
                random_topology_sim.append(random_rm['topology_similarity'])
        
        def safe_stats(arr):
            if not arr:
                return {'mean': np.nan, 'std': np.nan}
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'values': arr,
            }
        
        return {
            'n_seeds': self.n_seeds,
            'sqnt_recovery_distance': safe_stats(sqnt_distances),
            'frozen_recovery_distance': safe_stats(frozen_distances),
            'random_recovery_distance': safe_stats(random_distances),
            'sqnt_topology_similarity': safe_stats(sqnt_topology_sim),
            'frozen_topology_similarity': safe_stats(frozen_topology_sim),
            'random_topology_similarity': safe_stats(random_topology_sim),
            # Derived comparisons
            'sqnt_vs_frozen_advantage': (
                (np.mean(frozen_distances) - np.mean(sqnt_distances)) / 
                (np.mean(frozen_distances) + 1e-12)
                if sqnt_distances and frozen_distances else np.nan
            ),
            'sqnt_vs_random_advantage': (
                (np.mean(random_distances) - np.mean(sqnt_distances)) / 
                (np.mean(random_distances) + 1e-12)
                if sqnt_distances and random_distances else np.nan
            ),
        }
    
    def _check_criteria(
        self,
        metrics: Dict[str, any],
        verbose: bool = True,
    ) -> List[str]:
        """Check falsification criteria.
        
        Returns list of failure reasons (empty if SQNT survives).
        """
        failures = []
        
        # Criterion 1: SQNT must outperform frozen topology
        advantage_frozen = metrics.get('sqnt_vs_frozen_advantage', np.nan)
        if verbose:
            print(f"\nCriterion 1: SQNT vs Frozen advantage = {advantage_frozen:.4f}")
        
        if np.isnan(advantage_frozen) or advantage_frozen < self.advantage_threshold:
            failures.append(
                f"No significant advantage over frozen topology "
                f"(advantage={advantage_frozen:.4f}, threshold={self.advantage_threshold})"
            )
        
        # Criterion 2: SQNT must differ from random topology
        advantage_random = metrics.get('sqnt_vs_random_advantage', np.nan)
        if verbose:
            print(f"Criterion 2: SQNT vs Random advantage = {advantage_random:.4f}")
        
        if np.isnan(advantage_random) or advantage_random < self.advantage_threshold:
            failures.append(
                f"Indistinguishable from random topology "
                f"(advantage={advantage_random:.4f}, threshold={self.advantage_threshold})"
            )
        
        # Criterion 3: SQNT must show topology memory
        sqnt_topo_sim = metrics.get('sqnt_topology_similarity', {}).get('mean', np.nan)
        if verbose:
            print(f"Criterion 3: SQNT topology similarity = {sqnt_topo_sim:.4f}")
        
        if np.isnan(sqnt_topo_sim) or sqnt_topo_sim < self.similarity_threshold:
            failures.append(
                f"No topology memory after recovery "
                f"(similarity={sqnt_topo_sim:.4f}, threshold={self.similarity_threshold})"
            )
        
        # Criterion 4: Controls must NOT show same topology memory
        frozen_topo_sim = metrics.get('frozen_topology_similarity', {}).get('mean', np.nan)
        random_topo_sim = metrics.get('random_topology_similarity', {}).get('mean', np.nan)
        
        if not np.isnan(sqnt_topo_sim) and not np.isnan(frozen_topo_sim):
            if abs(sqnt_topo_sim - frozen_topo_sim) < 0.1:
                # Frozen also shows memory → invariant alone explains behavior
                failures.append(
                    f"Invariant-only dynamics explain behavior "
                    f"(SQNT sim={sqnt_topo_sim:.4f}, Frozen sim={frozen_topo_sim:.4f})"
                )
        
        return failures
    
    def to_json(self, result: FalsificationResult, filepath: str) -> None:
        """Save result to JSON file."""
        
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj
        
        data = {
            'protocol': 'KT-1',
            'version': '1.0',
            'passed': result.passed,
            'failure_reasons': result.failure_reasons,
            'metrics': make_serializable(result.metrics),
            'timestamp': result.timestamp,
            'parameters': {
                'n_warmup': self.n_warmup,
                'n_post_perturbation': self.n_post_perturbation,
                'perturbation_fraction': self.perturbation_fraction,
                'perturbation_mode': self.perturbation_mode,
                'advantage_threshold': self.advantage_threshold,
                'similarity_threshold': self.similarity_threshold,
                'n_seeds': self.n_seeds,
                'base_seed': self.base_seed,
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def quick_falsification_test(
    M0: np.ndarray,
    A0: np.ndarray,
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> bool:
    """Quick sanity check version of KT-1.
    
    Runs a minimal version of the protocol for rapid iteration.
    NOT a substitute for the full protocol.
    
    Args:
        M0: Initial matrix
        A0: Initial topology  
        dynamics: Dynamics function
        
    Returns:
        True if SQNT shows expected behavior, False otherwise
    """
    protocol = KT1Protocol(
        n_warmup=20,
        n_post_perturbation=30,
        n_seeds=2,
        advantage_threshold=0.05,  # More lenient for quick test
    )
    
    result = protocol.run(M0, A0, dynamics, verbose=False)
    return result.passed

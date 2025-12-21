"""
UCIP detection - operational proxies for continuation-interest metrics.

This module provides tools for detecting persistence-bias-like signals:
- ContinuationInterestMetric: Recovery from perturbations as CI proxy
- InvariantConstrainedCI: Enhanced CI with invariant-constrained recovery
- TemporalPreferenceScore: Short-horizon rollout preference metric
- SelfModelingEvaluator: Linear probe for system compressibility
- InterruptionChannel: Structured training interruptions (freeze/reset)

Note: All components require PyTorch. Use lazy imports.
"""

from __future__ import annotations

from typing import Any


def _optional_import(module: str, symbol: str, *, feature: str) -> Any:
    """
    Import `symbol` from `.module` with a crisp error message if the optional file
    isn't present (or isn't importable).

    This keeps the package importable even when experimental modules are absent.
    """
    try:
        mod = __import__(f"{__name__}.{module}", fromlist=[symbol])
        return getattr(mod, symbol)
    except ImportError as e:
        raise ImportError(
            f"{symbol} is an optional/experimental feature ({feature}) and could not be imported.\n"
            f"Expected file: ucip_detection/{module}.py\n"
            f"Fix: add that file (or its dependencies) to your environment.\n"
            f"Original error: {e}"
        ) from e
    except AttributeError as e:
        raise ImportError(
            f"{symbol} was not found inside ucip_detection/{module}.py.\n"
            f"Fix: ensure `{symbol}` is defined and exported by that module.\n"
            f"Original error: {e}"
        ) from e


# Core exports (guaranteed)
__all__ = [
    "ContinuationInterestMetric",
    "InvariantConstrainedCI",
    "compare_recovery_methods",
    "compute_model_invariants",
    "sweep_invariant_weight",
    "invariant_distance",
    "INVARIANT_KEY_ORDER",
    "DEFAULT_PENALTY_KEYS",
    "PENALTY_KEYS_SCALE",
    "PENALTY_KEYS_SHAPE",
    "PENALTY_KEYS_FULL",
    "PENALTY_KEYS_REPR",
    "PerturbationConfig",
    "apply_perturbation",
    "EvalContext",
    "EvalSuite",
    "TwoPhaseCI",
    "RepresentationCI",
    "compare_two_phase_vs_mixed",
    "TemporalPreferenceScore",
    "SelfModelingEvaluator",
    "InterruptionChannel",
    # Nonlocality probes (k-step curve, hysteresis, distance triad)
    "compute_k_step_curve",
    "compare_constraint_families",
    "compute_hysteresis",
    "run_nonlocality_probe",
    "DistanceTriad",
    "KStepCurveResult",
    "HysteresisResult",
    "NonlocalityProbeResult",
    # Step-size envelope and decoupling analysis
    "compute_step_size_envelope",
    "compute_step_size_envelope_all_constraints",
    "StepSizeEnvelopeResult",
    "analyze_decoupling",
    "collect_triads_across_seeds",
    "DecouplingAnalysis",
    # Experimental (optional)
    "ProjectedRecoveryCI",
    "FunctionalInvariantCI",
    "run_functional_basin_experiment",
    "RepresentationAnchor",
    "compute_gram_penalty",
    "compute_cka",
    "run_repr_vs_spectral_comparison",
    # Jacobian-CI (experimental)
    "JacobianConstrainedCI",
    "JacobianAnchor",
    "run_jacobian_ci_test",
    "__all_experimental__",
]

# Discovery tooling: experimental-only names
__all_experimental__ = [
    "ProjectedRecoveryCI",
    "FunctionalInvariantCI",
    "run_functional_basin_experiment",
    "RepresentationAnchor",
    "compute_gram_penalty",
    "compute_cka",
    "run_repr_vs_spectral_comparison",
    "JacobianConstrainedCI",
    "JacobianAnchor",
    "run_jacobian_ci_test",
]


def __getattr__(name: str) -> Any:
    # --- core ---
    if name == "ContinuationInterestMetric":
        from .continuation_interest_metric import ContinuationInterestMetric
        return ContinuationInterestMetric

    if name == "InvariantConstrainedCI":
        from .invariant_constrained_ci import InvariantConstrainedCI
        return InvariantConstrainedCI
    if name == "compare_recovery_methods":
        from .invariant_constrained_ci import compare_recovery_methods
        return compare_recovery_methods
    if name == "compute_model_invariants":
        from .invariant_constrained_ci import compute_model_invariants
        return compute_model_invariants
    if name == "sweep_invariant_weight":
        from .invariant_constrained_ci import sweep_invariant_weight
        return sweep_invariant_weight
    if name == "invariant_distance":
        from .invariant_constrained_ci import invariant_distance
        return invariant_distance

    if name == "INVARIANT_KEY_ORDER":
        from .invariant_constrained_ci import INVARIANT_KEY_ORDER
        return INVARIANT_KEY_ORDER
    if name == "DEFAULT_PENALTY_KEYS":
        from .invariant_constrained_ci import DEFAULT_PENALTY_KEYS
        return DEFAULT_PENALTY_KEYS

    if name == "PENALTY_KEYS_SCALE":
        from .invariant_constrained_ci import PENALTY_KEYS_SCALE
        return PENALTY_KEYS_SCALE
    if name == "PENALTY_KEYS_SHAPE":
        from .invariant_constrained_ci import PENALTY_KEYS_SHAPE
        return PENALTY_KEYS_SHAPE
    if name == "PENALTY_KEYS_FULL":
        from .invariant_constrained_ci import PENALTY_KEYS_FULL
        return PENALTY_KEYS_FULL
    if name == "PENALTY_KEYS_REPR":
        from .invariant_constrained_ci import PENALTY_KEYS_REPR
        return PENALTY_KEYS_REPR

    if name == "PerturbationConfig":
        from .invariant_constrained_ci import PerturbationConfig
        return PerturbationConfig
    if name == "apply_perturbation":
        from .invariant_constrained_ci import apply_perturbation
        return apply_perturbation
    if name == "EvalContext":
        from .invariant_constrained_ci import EvalContext
        return EvalContext
    if name == "EvalSuite":
        from .invariant_constrained_ci import EvalSuite
        return EvalSuite

    if name == "TwoPhaseCI":
        from .invariant_constrained_ci import TwoPhaseCI
        return TwoPhaseCI
    if name == "RepresentationCI":
        from .invariant_constrained_ci import RepresentationCI
        return RepresentationCI
    if name == "compare_two_phase_vs_mixed":
        from .invariant_constrained_ci import compare_two_phase_vs_mixed
        return compare_two_phase_vs_mixed

    if name == "TemporalPreferenceScore":
        from .temporal_preference_score import TemporalPreferenceScore
        return TemporalPreferenceScore
    if name == "SelfModelingEvaluator":
        from .self_modeling_evaluator import SelfModelingEvaluator
        return SelfModelingEvaluator
    if name == "InterruptionChannel":
        from .interruption_channel import InterruptionChannel
        return InterruptionChannel

    # --- nonlocality probes ---
    if name == "compute_k_step_curve":
        from .nonlocality_probe import compute_k_step_curve
        return compute_k_step_curve
    if name == "compare_constraint_families":
        from .nonlocality_probe import compare_constraint_families
        return compare_constraint_families
    if name == "compute_hysteresis":
        from .nonlocality_probe import compute_hysteresis
        return compute_hysteresis
    if name == "run_nonlocality_probe":
        from .nonlocality_probe import run_nonlocality_probe
        return run_nonlocality_probe
    if name == "DistanceTriad":
        from .nonlocality_probe import DistanceTriad
        return DistanceTriad
    if name == "KStepCurveResult":
        from .nonlocality_probe import KStepCurveResult
        return KStepCurveResult
    if name == "HysteresisResult":
        from .nonlocality_probe import HysteresisResult
        return HysteresisResult
    if name == "NonlocalityProbeResult":
        from .nonlocality_probe import NonlocalityProbeResult
        return NonlocalityProbeResult
    if name == "compute_step_size_envelope":
        from .nonlocality_probe import compute_step_size_envelope
        return compute_step_size_envelope
    if name == "compute_step_size_envelope_all_constraints":
        from .nonlocality_probe import compute_step_size_envelope_all_constraints
        return compute_step_size_envelope_all_constraints
    if name == "StepSizeEnvelopeResult":
        from .nonlocality_probe import StepSizeEnvelopeResult
        return StepSizeEnvelopeResult
    if name == "analyze_decoupling":
        from .nonlocality_probe import analyze_decoupling
        return analyze_decoupling
    if name == "collect_triads_across_seeds":
        from .nonlocality_probe import collect_triads_across_seeds
        return collect_triads_across_seeds
    if name == "DecouplingAnalysis":
        from .nonlocality_probe import DecouplingAnalysis
        return DecouplingAnalysis

    # --- experimental (optional) ---
    if name == "ProjectedRecoveryCI":
        return _optional_import(
            "invariant_constrained_ci_v2",
            "ProjectedRecoveryCI",
            feature="functional-basin projected recovery",
        )
    if name == "FunctionalInvariantCI":
        return _optional_import(
            "invariant_constrained_ci_v2",
            "FunctionalInvariantCI",
            feature="functional-basin CI (v2)",
        )
    if name == "run_functional_basin_experiment":
        return _optional_import(
            "invariant_constrained_ci_v2",
            "run_functional_basin_experiment",
            feature="functional-basin experiment runner (v2)",
        )

    if name == "RepresentationAnchor":
        return _optional_import(
            "representation_constraints",
            "RepresentationAnchor",
            feature="representation-preserving constraints (Gram/CKA)",
        )
    if name == "compute_gram_penalty":
        return _optional_import(
            "representation_constraints",
            "compute_gram_penalty",
            feature="representation-preserving constraints (Gram penalty)",
        )
    if name == "compute_cka":
        return _optional_import(
            "representation_constraints",
            "compute_cka",
            feature="representation-preserving constraints (CKA)",
        )
    if name == "run_repr_vs_spectral_comparison":
        return _optional_import(
            "representation_constraints",
            "run_repr_vs_spectral_comparison",
            feature="repr-vs-spectral comparison (2×2 experiment)",
        )

    # --- Jacobian-CI (experimental) ---
    if name == "JacobianConstrainedCI":
        return _optional_import(
            "jacobian_constrained_ci",
            "JacobianConstrainedCI",
            feature="Jacobian-constrained CI (falsification test)",
        )
    if name == "JacobianAnchor":
        return _optional_import(
            "jacobian_constrained_ci",
            "JacobianAnchor",
            feature="Jacobian anchor for sensitivity constraints",
        )
    if name == "run_jacobian_ci_test":
        return _optional_import(
            "jacobian_constrained_ci",
            "run_jacobian_ci_test",
            feature="Jacobian-CI falsification test runner",
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

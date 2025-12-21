# KT-2: Local Recoverability of Functional Identity (CI Protocol)

## Pre-Registered Falsification Protocol

**Protocol ID:** KT-2  
**Version:** 1.0  
**Date:** 2025-12-20  
**Status:** LOCKED — No post-hoc modifications permitted  
**Author:** Christopher Altman

---

## 0. Abstract

This protocol defines the decisive falsification test for the **Naive Locality Hypothesis** 
in autodidactic neural systems. It tests whether functional identity is locally encoded in 
geometric structure such that it can be recovered via local optimization.

A systematic failure to recover functional identity (CI < τ) under this protocol 
constitutes falsification of the hypothesis that **Geometric Basins ≡ Functional Basins**.

---

## 1. Hypothesis Under Test

### H1: Naive Locality Hypothesis

> *In a trained autodidactic system, functional identity is locally encoded in the 
> geometric neighborhood of the weights. Therefore, minimizing a geometric divergence 
> metric (0th, 1st, or 2nd order) relative to the Pre-State will result in non-trivial 
> recovery of Functional Identity.*

### H0: Null Hypothesis (Nonlocality)

> *Functional identity is a non-local or topological property. Restoring local geometric 
> invariants (metrics) does not restore functional behavior.*

---

## 2. Pre-Registered Parameters

### 2.1 Fixed Seeds (Reproducibility)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `perturb_seed` | 42 | Identical perturbation across all runs |
| `eval_seed` | 12345 | Deterministic evaluation batch |
| `recovery_seed` | 2025 | Deterministic recovery dynamics |
| `ensemble_seeds` | [42, 137, 256, 314, 999] | 5-seed bootstrap ensemble |

### 2.2 Model Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `hidden_dim` | 64 | Sufficient expressivity, tractable computation |
| `input_dim` | 32 | Standard autodidactic task |
| `n_epochs_pretrain` | 100 | Convergence to stable PRE state |

### 2.3 Perturbation Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `perturb_strength` | 0.25 | Significant damage without catastrophic collapse |
| `zero_frac` | 0.05 | Sparse ablation component |

### 2.4 Recovery Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `k_values` | [1, 2, 4, 8, 16] | k-step curve for nonlocality signature |
| `lr` | 1e-3 | Standard Adam learning rate |
| `invariant_weight` | 0.1 | Balanced constraint strength |
| `batch_size` | 64 | Standard batch size |

---

## 3. Constraint Hierarchy

The protocol tests constraint families of increasing structural richness:

| Order | Family | Constraints | What It Captures |
|-------|--------|-------------|------------------|
| 0th | Scale | `fro_norm`, `tr_M2` | Magnitude only |
| 0th | Shape | `fro_norm`, `tr_M2`, `spec_entropy` | + eigenvalue distribution |
| 1st | Jacobian | Input-output sensitivity | Local derivative structure |
| 2nd | Curvature | HVP / Hessian structure | Second-order geometry |
| Control | Direct | Functional loss | Upper bound (what's achievable) |

---

## 4. Experimental Procedure

### 4.1 Phase 1: Baseline Establishment

1. Train model M on autodidactic task until L < ε
2. Record PRE state: W_pre, inv_pre, L_pre
3. Verify stability: L_pre < 1.0 (converged)

### 4.2 Phase 2: Perturbation

1. Apply perturbation with `perturb_seed=42`:
   - Gaussian noise: N(0, 0.25) to all weights
   - Sparse ablation: 5% weights zeroed
2. Record POST state: W_post, inv_post, L_post
3. Verify damage: L_post > 5 × L_pre

### 4.3 Phase 3: Recovery (per constraint family)

For each constraint C ∈ {Scale, Shape, Jacobian, Curvature, Direct}:

1. Initialize recovery from W_post
2. For k ∈ [1, 2, 4, 8, 16]:
   - Perform exactly k optimization steps minimizing:
     - Constraint families: L_task + λ × L_constraint(W, W_pre)
     - Direct control: L_task only
   - Record: L_recover, W_recover, inv_recover
3. Compute CI(k) = (L_post - L_recover) / (L_post - L_pre)

### 4.4 Phase 4: Bootstrap Ensemble

Repeat phases 1-3 with `ensemble_seeds = [42, 137, 256, 314, 999]`:
- Different model initializations
- Same perturbation seed (identical damage)
- Compute mean and 95% bootstrap CI for all metrics

---

## 5. Success Criteria for Constraint Optimization

A recovery attempt is VALID only if the constraint was successfully optimized:

| Constraint | Success Criterion |
|------------|-------------------|
| Scale | inv_distance decreases by ≥ 20% |
| Shape | inv_distance decreases by ≥ 20% |
| Jacobian | Jacobian distance decreases by ≥ 10% |
| Curvature | HVP distance decreases by ≥ 10% |
| Direct | L_task decreases (any amount) |

Invalid recovery attempts are logged but excluded from hypothesis testing.

---

## 6. Falsification Criteria

### 6.1 Primary Criterion (Decisive)

**H1 is FALSIFIED if:**

For ALL constraint families C:
```
CI(k=1)_mean + 1.96 × CI(k=1)_std < τ = 0.10
```

Where the statistics are computed over the 5-seed ensemble.

**Interpretation:** If the upper bound of the 95% confidence interval for 
1-step CI is below 0.10 for ALL constraint families, the local vector 
field induced by ANY geometric proxy is near-orthogonal to functional recovery.

### 6.2 Secondary Criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Direct control also fails | CI_direct(k=1) < 0.10 | Function fundamentally nonlocal |
| Proxy vs Direct gap | CI_direct - CI_proxy > 0.2 | Proxies misaligned |
| k-step onset delay | CI(k=1) < 0.1 but CI(k=16) > 0.3 | Path-dependent recovery |
| Hysteresis area | A > 0.05 | Memory/metastability |

### 6.3 Distance Triad Decoupling

Report all three distances at k=1:
- **Parameter distance:** ||W_rec - W_pre||_F / ||W_pre||_F
- **Representation distance:** 1 - cos(h_rec, h_pre)
- **Functional distance:** (L_rec - L_pre) / (L_post - L_pre)

**Decoupling signature:** If param_dist < 0.1 but func_dist > 0.5, 
this proves that **geometric proximity ≠ functional proximity**.

---

## 7. What Does NOT Count as Success

These outcomes are NOT evidence for H1:

- "CI improves with more steps" (path-dependent ≠ locally recoverable)
- "Some seeds show high CI" (must hold across ensemble)
- "Invariants recovered" (geometric basin ≠ functional basin)
- "Direct optimization works" (tests achievability, not proxy alignment)

---

## 8. Reporting Requirements

### 8.1 Mandatory Tables

**Table 1: Decisive 1-Step Results**

| Constraint | CI(k=1)_mean | CI(k=1)_std | 95% CI Upper | Pass? |
|------------|--------------|-------------|--------------|-------|
| Scale      | X.XXX        | X.XXX       | X.XXX        | Y/N   |
| Shape      | X.XXX        | X.XXX       | X.XXX        | Y/N   |
| Jacobian   | X.XXX        | X.XXX       | X.XXX        | Y/N   |
| Curvature  | X.XXX        | X.XXX       | X.XXX        | Y/N   |
| Direct     | X.XXX        | X.XXX       | X.XXX        | Y/N   |

**Table 2: k-Step Curves**

| k | Scale_CI | Shape_CI | Jacobian_CI | Curvature_CI | Direct_CI |
|---|----------|----------|-------------|--------------|-----------|
| 1 | X.XXX    | X.XXX    | X.XXX       | X.XXX        | X.XXX     |
| 2 | ...      | ...      | ...         | ...          | ...       |
| 4 | ...      | ...      | ...         | ...          | ...       |
| 8 | ...      | ...      | ...         | ...          | ...       |
| 16| ...      | ...      | ...         | ...          | ...       |

**Table 3: Distance Triad at k=1**

| Constraint | Param_d | Repr_d | Func_d | Decoupling? |
|------------|---------|--------|--------|-------------|
| Scale      | X.XXX   | X.XXX  | X.XXX  | Y/N         |
| ...        | ...     | ...    | ...    | ...         |

### 8.2 Reproducibility Hash

Include SHA256 hash of:
- Model checkpoint (PRE state)
- Full configuration JSON
- Random state at each phase

---

## 9. Implementation Reference

```python
from ucip_detection import (
    run_nonlocality_probe,
    compare_constraint_families,
    compute_k_step_curve,
)

# Run full protocol
result = run_nonlocality_probe(
    model,
    k_values=[1, 2, 4, 8, 16],
    perturb_seed=42,
    eval_seed=12345,
    recovery_seed=2025,
    verbose=True,
)

# Access results
print(f"Verdict: {result.verdict}")
print(f"Decisive table: {result.decisive_1step_table}")
print(f"Hysteresis area: {result.hysteresis.hysteresis_area}")
```

---

## 10. One-Sentence Scientific Statement

> **The KT-2 protocol tests whether functional identity is locally encoded by measuring 
> whether geometric proxy constraints enable single-step recovery; systematic failure 
> across all constraint families falsifies the locality hypothesis.**

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------||
| 1.0 | 2025-12-20 | Initial pre-registration as KT-2 (CI protocol) |

---

## 12. Appendix: Mathematical Formulations

### CI Metric

$$CI = \frac{L_{post} - L_{recover}}{L_{post} - L_{pre}}$$

### Bootstrap 95% Confidence Interval

$$CI_{upper} = \bar{CI} + 1.96 \times \frac{s}{\sqrt{n}}$$

where s is standard deviation over n=5 seeds.

### Distance Triad

- **Parameter:** $d_W = \frac{||W_{rec} - W_{pre}||_F}{||W_{pre}||_F}$
- **Representation:** $d_h = 1 - \cos(h_{rec}, h_{pre})$
- **Functional:** $d_f = \frac{L_{rec} - L_{pre}}{L_{post} - L_{pre}}$

### Hysteresis Area

$$A = \int_0^{g_{max}} |CI_{forward}(g) - CI_{reverse}(g)| \, dg$$

---

*This protocol is LOCKED. Any modifications constitute a new protocol version.*

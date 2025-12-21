# KT-1: Topology–Perturbation Memory Test

## Pre-Registered Falsification Protocol

**Protocol ID:** KT-1  
**Version:** 1.0  
**Date:** 2025-12-19  
**Status:** LOCKED — No post-hoc modifications permitted  
**Author:** Christopher Altman

---

## 1. Aim

Test whether **SQNT-inspired topology plasticity** creates genuine structural memory 
under perturbation, or whether the system's topology is merely epiphenomenal.

This is a falsifier for the claim:

> **Topology Plasticity Hypothesis (H1):** Adaptive topology creates structural 
> memory that aids recovery from perturbation beyond what frozen topology provides.

---

## 2. Background: SQNT Lineage

This protocol tests a classical instantiation of principles from:

- C. Altman, J. Pykacz & R. Zapatrin, "Superpositional Quantum Network Topologies," 
  Int. J. Theor. Phys. 43, 2029–2041 (2004).
- C. Altman & R. Zapatrin, "Backpropagation in Adaptive Quantum Networks," 
  Int. J. Theor. Phys. 49, 2991–2997 (2010).

The core SQNT insight: network topology should be *adaptive and trainable*, not fixed.
This implementation tests whether such plasticity confers measurable advantage.

---

## 3. Definitions

### 3.1 Topology Configurations

| Configuration | Description |
|---------------|-------------|
| **SQNT (plastic)** | Topology adapts via participation-based plasticity rules |
| **FROZEN** | Fixed topology from initialization, no adaptation |
| **RANDOM** | Topology rerandomized each step (no memory) |

### 3.2 Recovery Metrics

| Metric | Definition |
|--------|------------|
| **Matrix Fidelity** | Bounded cosine-style Frobenius overlap in [0,1] |
| **Spectral Fidelity** | Overlap of sorted eigenvalue distributions |
| **Topology Fidelity** | Graph edit distance or Laplacian spectral distance |

### 3.3 Reversibility Classes

| Class | Criterion |
|-------|-----------|
| FULLY_REVERSIBLE | All fidelities > 0.95 |
| PARTIALLY_REVERSIBLE | At least one fidelity > 0.7 |
| IRREVERSIBLE | All fidelities < 0.7, none negative |
| CHAOTIC | Fidelities degrade exponentially with perturbation |
| UNSTABLE | Numerical instability (NaN/Inf) |

---

## 4. Procedure (Locked)

### 4.1 Phase 1: Evolution

1. Initialize matrix M₀ from Hermitian ensemble (seed locked)
2. Evolve via Langevin dynamics for T steps
3. Record trajectory {M_t} and topology at each step

### 4.2 Phase 2: Time Reversal

1. At step T, reverse dynamics (flip dt sign)
2. Evolve backward for T steps
3. Record recovery trajectory {M'_t}

### 4.3 Phase 3: Fidelity Measurement

For each topology configuration:
1. Compute matrix_fidelity(M_0, M'_0)
2. Compute spectral_fidelity(M_0, M'_0)
3. Compute topology_fidelity(G_0, G'_0)

### 4.4 Phase 4: Perturbation Sensitivity

1. Add small perturbation ε to M₀
2. Repeat phases 1-3
3. Measure fidelity degradation vs ε

---

## 5. Falsification Criteria (Locked)

### 5.1 Primary Criterion

**H1 is FALSIFIED if:**

```
fidelity(SQNT) ≤ fidelity(FROZEN) + δ
```

where δ = 0.05 (tolerance for noise).

**Interpretation:** If plastic topology shows no advantage over frozen topology 
in recovery from perturbation, then topology plasticity does not create 
meaningful structural memory.

### 5.2 Secondary Criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| SQNT vs RANDOM | SQNT > RANDOM by 0.1 | Plasticity has any effect |
| Sensitivity scaling | Linear vs exponential | Chaotic vs regular dynamics |
| Topology fidelity | > Matrix fidelity | Topology is the memory carrier |

---

## 6. Pre-Registered Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| dim | 12 | Tractable computation |
| langevin_steps | 800 | Sufficient equilibration |
| dt | 1e-3 | Numerical stability |
| temperature | 1e-2 | Mild stochasticity |
| coupling_g | 0.1 | Moderate interaction |
| perturb_eps | [0.01, 0.05, 0.1, 0.25] | Sensitivity sweep |
| seeds | [42, 137, 256, 314, 999] | 5-seed ensemble |

---

## 7. Implementation Reference

```python
from analysis import TimeReversalProbe, ReversibilityClass
from sqnt import sqnt_update

# Run time reversal experiment
probe = TimeReversalProbe(
    forward_steps=400,
    reverse_steps=400,
    perturbation_magnitude=0.1,
)

result = probe.run_with_perturbation(
    M_initial=M0,
    dynamics=langevin_dynamics,
    topology_update=sqnt_update,
)

print(f"Matrix fidelity: {result.matrix_fidelity:.4f}")
print(f"Spectral fidelity: {result.spectral_fidelity:.4f}")
print(f"Classification: {result.reversibility_class}")
```

---

## 8. Artifact Requirements

Each run MUST output:
- `kt1_results.json` containing: protocol_id, version, git_hash, seeds, 
  all fidelity values, reversibility classification
- `kt1_trajectory.npz` containing: matrix trajectories, topology snapshots

---

## 9. Scientific Question

> Does adaptive topology plasticity (SQNT-style) create genuine structural 
> memory that aids recovery, or is topology merely an epiphenomenal 
> consequence of matrix dynamics?

---

*Protocol locked on 2025-12-19.*

# Primary Claim and Endpoint

**Status:** LOCKED as of v0.2.0
**Protocol:** KT-2 v1.0
**No post-hoc modifications permitted after tag**

---

## The Claim (One Paragraph)

Geometric/proxy proximity does not imply functional recovery in KT-2. Specifically: in a trained autodidactic matrix-to-RNN system, single-step constrained recovery that restores local geometric structure (0th-order spectral invariants including Frobenius norm, trace moments, and spectral entropy) does NOT restore functional identity. Across all tested constraint families (None, Scale, Shape, Direct), CI(k=1) remains below threshold (0.10), demonstrating that geometric basins are not functional basins in this system.

---

## Primary Endpoint

**Test:** Decisive 1-step CI test across all constraint families
**Command:** `python -m experiments.kt2_locality_falsifier --run-decisive`
**Threshold:** CI(k=1) < 0.10 for ALL constraint families → locality falsified
**Output artifact:** `results/kt2_decisive_1step.json`

---

## Fixed Seeds (Deterministic Reproduction)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `perturb_seed` | 42 | Identical perturbation across all runs |
| `eval_seed` | 12345 | Deterministic evaluation batch |
| `recovery_seed` | 2025 | Deterministic recovery dynamics |

---

## Fixed Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `dim` | 12 | Matrix dimension |
| `hidden` | 64 | RNN hidden size |
| `lr` | 1e-3 | Recovery learning rate |
| `perturb_strength` | 0.25 | Gaussian noise scale |
| `zero_frac` | 0.05 | Sparse ablation fraction |
| `invariant_weight` | 0.1 | Constraint penalty coefficient |

All values defined as constants in `experiments/kt2_locality_falsifier.py`: `PERTURB_SEED`, `EVAL_SEED`, `RECOVERY_SEED`, `CI_THRESHOLD`, `DEFAULT_DIM`, `DEFAULT_HIDDEN`, `DEFAULT_LR`, `DEFAULT_PERTURB_STRENGTH`, `DEFAULT_ZERO_FRAC`, `DEFAULT_INVARIANT_WEIGHT`.

---

## Constraint Families Tested

1. **None** — Baseline (no constraints)
2. **Scale** — Frobenius norm + trace(M²)
3. **Shape** — Scale + spectral entropy
4. **Direct** — Unconstrained task loss (control)

---

## Primary Test Definition

The primary test implements the KT-2 protocol with k=1 (single recovery step):

1. Train model to PRE state (converged autodidactic loop)
2. Apply controlled perturbation → POST state
3. Execute 1-step constrained recovery for each family
4. Compute CI = (L_post - L_recover) / (L_post - L_pre)
5. Compare CI against threshold τ = 0.10

**Pass condition (falsification):** ALL families yield CI(k=1) < 0.10

---

## Interpretation

**CI metric:**
- CI = 1.0 → Full recovery to PRE loss in one step
- CI = 0.0 → No recovery beyond POST
- CI < 0.0 → Recovery step made things worse

**Verdict:**
- If ALL constraint families yield CI(k=1) < 0.10: Geometric basins ≠ functional basins (locality falsified)
- If ANY family yields CI(k=1) ≥ 0.10: That proxy enables local functional recovery

---

## Distance Triad (Decoupling Evidence)

At k=1, three distances are measured:

1. **Parameter distance:** ||W_recover - W_pre||_F / ||W_pre||_F
2. **Representation distance:** 1 - cos(h_recover, h_pre)
3. **Functional distance:** (L_recover - L_pre) / (L_post - L_pre) = 1 - CI

**Decoupling signature:** If parameter/representation distances are small but functional distance remains large, this demonstrates that geometric proximity ≠ functional proximity.

---

## No Post-Hoc Changes Rule

As of tag v0.2.0, this claim and endpoint are LOCKED. Any modifications to:
- Seeds
- Hyperparameters
- Threshold
- Constraint family definitions
- CI computation

constitute a new protocol version and must be documented as such with full justification.

**Rationale:** Post-hoc parameter tuning invalidates falsification claims. The protocol must be criterion-locked before observing outcomes.

---

## Reproducibility

Every decisive run outputs `results/kt2_decisive_1step.json` with full provenance:
- Git commit hash
- Timestamp
- Platform info
- Dependency versions
- All seeds and parameters

This artifact is deterministic given the locked seeds and serves as a cryptographic binding to the protocol state.

---

**End of claim document. Last modified: 2025-12-22**

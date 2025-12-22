# Reproducing the Results

This repository contains a **designed falsifier**. The decisive results (falsification of the Naive Locality Hypothesis) are reproducible with a single command on a clean machine.

## Prerequisites

- **OS:** Linux or macOS (Windows should work but is untested)
- **Python:** 3.10+
- **Hardware:** Standard CPU (no GPU required, though supported)
- **Time:** ~2-5 minutes

## Quickstart (The Decisive Run)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/christopher-altman/autodidactic-qml.git
    cd autodidactic-qml
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e .
    # OR
    pip install -r requirements.txt
    ```

3.  **Run the decisive KT-2 experiment:**
    ```bash
    python -m experiments.kt2_locality_falsifier --run-decisive
    ```

## Expected Output

You should see output similar to the following (values will be exact):

```text
kt2_decisive_1step.json
...
VERDICT: FALSIFIED
H1 (Naive Locality) is FALSIFIED: All CI(k=1) < 0.10
```

This confirms that even when geometric constraints (Scale, Shape, Direct) are satisfied or optimized, functional identity (CI) is not locally recovered at k=1.

## Generated Artifacts

The run produces artifacts in `results/` (default):

-   `kt2_decisive_1step.json`: The raw data for the decisive falsification table. **This is a provenance-stamped deterministic reference artifact.** It includes full metadata (seeds, platform, versions) for auditability.
-   (Optional) `kt2_k_step_curves.json`: If running with `--k-step-curve`.

## Determinism

All decisive experiments use locked seeds:
-   `perturb_seed=42`: Controls the damage applied.
-   `eval_seed=12345`: Controls the test batch.
-   `recovery_seed=2025`: Controls the stochastic gradient descent path (if applicable).

No hyperparameter sweeping is performed during the decisive run. The parameters are fixed in `protocols/KT-2`.

## Robustness Grid

Test replicability across 30 runs (3 dimensions × 10 seeds):

```bash
python -m experiments.kt2_robustness_grid
```

Outputs:
- `results/kt2_robustness_grid.json` (full raw data + summary + provenance)
- `results/kt2_robustness_grid.csv` (tabular per-run results)

This demonstrates that the decisive result is not a single lucky seed.

## Negative Control

Prove the harness can restore function when optimizing for function directly:

```bash
python -m experiments.kt2_locality_falsifier --negative-control
```

Outputs: `results/kt2_negative_control.json`

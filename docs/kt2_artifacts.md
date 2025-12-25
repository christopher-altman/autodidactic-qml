# KT-2 Artifact Manifest

All KT-2 artifacts are JSON files written to the output directory (default: `results/`, configurable via `--output-dir`).

## Artifact List

| Artifact | Command Flag | Description |
|----------|--------------|-------------|
| `kt2_decisive_1step.json` | `--run-decisive` | Decisive 1-step CI table across all constraint families (primary falsification test) |
| `kt2_k_step_curves.json` | `--k-step-curve` | CI(k) curves for k ∈ {1,2,4,8,16} showing nonlocality signature |
| `kt2_hysteresis.json` | `--hysteresis` | Forward/reverse perturbation sweep with hysteresis area measurement |
| `kt2_step_envelope.json` | `--step-envelope` | Best 1-step CI over step-size (η) grid for each constraint family |
| `kt2_distance_triads.json` | (included in curves) | Parameter + representation + functional distance triads |
| `kt2_decoupling.json` | `--decoupling-analysis` | Distance-triad decoupling analysis across multiple seeds |
| `kt2_full_protocol.json` | `--full-protocol` | Complete protocol output with all tests and overall verdict |
| `kt2_negative_control.json` | `--negative-control` | Negative control (distillation vs proxy recovery, harness sanity check) |

## Output Directory Behavior

- **Default:** All artifacts write to `results/` (relative to working directory)
- **Custom:** Use `--output-dir path/to/dir` to specify a different location
- **Negative control exception:** `kt2_negative_control.json` currently writes to `results/` root regardless of `--output-dir`

## Artifact Schema (Common Fields)

All KT-2 artifacts include provenance metadata:

```json
{
  "protocol_id": "KT-2",
  "protocol_version": "1.0",
  "test": "<test_name>",
  "git_hash": "<commit_hash>",
  "timestamp": "<ISO8601_timestamp>",
  "seeds": {
    "perturb": 42,
    "eval": 12345,
    "recovery": 2025
  },
  ...
}
```

## Locked Seeds

All artifacts are generated with deterministic seeds:

- `PERTURB_SEED = 42` — Perturbation randomness
- `EVAL_SEED = 12345` — Evaluation batch generation
- `RECOVERY_SEED = 2025` — Recovery step randomness

These seeds are hardcoded in `experiments/kt2_locality_falsifier.py` and must not be modified without incrementing the protocol version.

## Reproducibility

Given the locked seeds and protocol parameters, all artifacts should be bit-for-bit reproducible across runs (modulo timestamp and git hash).

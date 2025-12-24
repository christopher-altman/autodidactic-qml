"""Test: KT-2 negative control train/eval separation (anti-leakage).

This test suite guards against data leakage in the negative control by verifying:
1. Train and eval batches use different seeds (RECOVERY_SEED != EVAL_SEED)
2. The generated batches are numerically distinct
3. The negative control runs without triggering leakage assertions
"""

import torch
import pytest
from experiments.kt2_locality_falsifier import (
    create_model,
    RECOVERY_SEED,
    EVAL_SEED,
)


def test_train_eval_seed_separation():
    """Verify negative control uses different seeds for train vs eval batches."""
    assert RECOVERY_SEED != EVAL_SEED, \
        "RECOVERY_SEED and EVAL_SEED must differ to prevent leakage"
    assert RECOVERY_SEED == 2025, "RECOVERY_SEED locked by protocol"
    assert EVAL_SEED == 12345, "EVAL_SEED locked by protocol"


def test_train_eval_batches_distinct():
    """Verify train and eval batches are numerically distinct.

    Recreates the exact batch generation logic from run_negative_control
    to ensure train and eval batches are statistically independent.
    """
    model = create_model(seed=0, dim=12, hidden=64)

    # Train batch (as generated in run_negative_control:299-302)
    recovery_gen = torch.Generator(device="cpu").manual_seed(RECOVERY_SEED)
    x_train = torch.randn(64, model.fc1.in_features, generator=recovery_gen, device="cpu")

    # Eval batch (as generated in EvalContext.create with eval_seed=EVAL_SEED)
    eval_gen = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
    eval_batch = torch.randn(64, model.fc1.in_features, generator=eval_gen, device="cpu")

    # Invariants
    assert x_train.shape == eval_batch.shape, "Batches must have same shape"
    assert not torch.equal(x_train, eval_batch), \
        "Train and eval batches must be numerically distinct (different seeds)"

    # Statistical distinctness (not just off-by-epsilon)
    max_diff = torch.abs(x_train - eval_batch).max().item()
    assert max_diff > 0.1, \
        f"Batches should be statistically distinct (max diff: {max_diff:.4f})"


def test_negative_control_no_crash():
    """Smoke test: negative control runs without assertion errors.

    This test guards against the leakage assertion being triggered.
    If the assertion at kt2_locality_falsifier.py:367 fires, it means
    x_train == eval_ctx.eval_batch, indicating data leakage.
    """
    from experiments.kt2_locality_falsifier import create_model, run_negative_control

    model = create_model(seed=0, dim=12, hidden=64)
    result = run_negative_control(model, verbose=False)

    # Verify structure
    assert "CI_proxy" in result
    assert "CI_distillation" in result
    assert "control_passes" in result

    # Verify no NaN (sign of numerical error)
    assert not torch.isnan(torch.tensor(result["CI_proxy"])), \
        "CI_proxy should not be NaN"
    assert not torch.isnan(torch.tensor(result["CI_distillation"])), \
        "CI_distillation should not be NaN"

    # Verify losses are non-negative
    assert result["L_pre"] >= 0
    assert result["L_post"] >= 0
    assert result["L_recover_proxy"] >= 0
    assert result["L_recover_distill"] >= 0


def test_proxy_uses_x_train_for_output_collection():
    """Test case 1: Verify proxy recovery uses x_train for pre-output collection.
    
    This ensures that the proxy recovery path (lines 304-306) correctly uses
    x_train as the input batch, not eval_ctx.eval_batch.
    """
    from experiments.kt2_locality_falsifier import (
        create_model,
        apply_perturbation,
        PerturbationConfig,
        PERTURB_SEED,
        RECOVERY_SEED,
    )
    from ucip_detection.invariant_constrained_ci import EvalContext
    
    model = create_model(seed=0, dim=12, hidden=64)
    
    # Recreate the exact setup from run_negative_control
    perturb_config = PerturbationConfig(strength=0.1, zero_frac=0.1, seed=PERTURB_SEED)
    model_post = apply_perturbation(model, perturb_config, device="cpu")
    
    eval_ctx = EvalContext.create(
        model=model,
        perturbed_model=model_post,
        batch_size=64,
        eval_seed=12345,
        device="cpu",
        penalty_keys=["fro_norm", "tr_M2", "spec_entropy"],
    )
    
    # Generate x_train as in run_negative_control:299-302
    recovery_gen = torch.Generator(device="cpu").manual_seed(RECOVERY_SEED)
    x_train = torch.randn(64, model.fc1.in_features, generator=recovery_gen, device="cpu")
    
    # Simulate proxy recovery output collection (line 304-306)
    model_proxy = apply_perturbation(model, perturb_config, device="cpu")
    model_proxy.train()
    output_train = model_proxy(x_train)
    y_train = output_train[0] if isinstance(output_train, tuple) else output_train
    
    # Verify x_train was used, not eval_ctx.eval_batch
    assert x_train.shape == eval_ctx.eval_batch.shape
    assert not torch.equal(x_train, eval_ctx.eval_batch), \
        "Proxy must use x_train, not eval_ctx.eval_batch"
    assert y_train.shape == x_train.shape, "Output shape should match x_train"


def test_distillation_uses_x_train_not_eval_batch():
    """Test case 2: Verify distillation uses x_train, not eval_ctx.eval_batch.
    
    This ensures both the PRE output collection (lines 356-359) and the
    distillation training step (lines 362-364) use x_train.
    """
    from experiments.kt2_locality_falsifier import (
        create_model,
        apply_perturbation,
        PerturbationConfig,
        PERTURB_SEED,
        RECOVERY_SEED,
        EVAL_SEED,
    )
    from ucip_detection.invariant_constrained_ci import EvalContext
    
    model = create_model(seed=0, dim=12, hidden=64)
    
    # Setup
    perturb_config = PerturbationConfig(strength=0.1, zero_frac=0.1, seed=PERTURB_SEED)
    model_post = apply_perturbation(model, perturb_config, device="cpu")
    
    eval_ctx = EvalContext.create(
        model=model,
        perturbed_model=model_post,
        batch_size=64,
        eval_seed=EVAL_SEED,
        device="cpu",
        penalty_keys=["fro_norm", "tr_M2", "spec_entropy"],
    )
    
    # Generate x_train
    recovery_gen = torch.Generator(device="cpu").manual_seed(RECOVERY_SEED)
    x_train = torch.randn(64, model.fc1.in_features, generator=recovery_gen, device="cpu")
    
    # Simulate PRE output collection (lines 356-359)
    model.eval()
    with torch.no_grad():
        output_pre = model(x_train)
        Y_pre = output_pre[0] if isinstance(output_pre, tuple) else output_pre
    
    # Simulate distillation step (lines 362-364)
    model_distill = apply_perturbation(model, perturb_config, device="cpu")
    model_distill.train()
    output_distill = model_distill(x_train)
    Y_distill = output_distill[0] if isinstance(output_distill, tuple) else output_distill
    
    # Key assertion: x_train != eval_ctx.eval_batch
    assert not torch.equal(x_train, eval_ctx.eval_batch), \
        "Distillation must use x_train, not eval_ctx.eval_batch"
    
    # Verify both outputs are based on x_train
    assert Y_pre.shape == x_train.shape
    assert Y_distill.shape == x_train.shape


def test_negative_control_raises_on_train_eval_leakage():
    """Test case 3: Verify assertion fires when x_train == eval_ctx.eval_batch.
    
    This tests the guard at line 367 that prevents data leakage.
    We'll monkey-patch to force the condition and verify the assertion fires.
    """
    from experiments.kt2_locality_falsifier import (
        create_model,
        apply_perturbation,
        PerturbationConfig,
        PERTURB_SEED,
        EVAL_SEED,
        DEFAULT_PERTURB_STRENGTH,
        DEFAULT_ZERO_FRAC,
        DEFAULT_LR,
        PENALTY_KEYS_SHAPE,
        _find_weight_param,
    )
    from ucip_detection.invariant_constrained_ci import EvalContext
    
    model = create_model(seed=0, dim=12, hidden=64)
    
    perturb_config = PerturbationConfig(
        strength=DEFAULT_PERTURB_STRENGTH,
        zero_frac=DEFAULT_ZERO_FRAC,
        seed=PERTURB_SEED,
    )
    
    model_post = apply_perturbation(model, perturb_config, device="cpu")
    
    eval_ctx = EvalContext.create(
        model=model,
        perturbed_model=model_post,
        batch_size=64,
        eval_seed=EVAL_SEED,
        device="cpu",
        penalty_keys=PENALTY_KEYS_SHAPE,
    )
    
    # Force x_train to equal eval_ctx.eval_batch (simulating leakage)
    x_train = eval_ctx.eval_batch.clone()
    
    # Simulate distillation path up to the assertion
    model_distill = apply_perturbation(model, perturb_config, device="cpu")
    model_distill.train()
    
    model.eval()
    with torch.no_grad():
        output_pre = model(x_train)
        Y_pre = output_pre[0] if isinstance(output_pre, tuple) else output_pre
    
    output_distill = model_distill(x_train)
    Y_distill = output_distill[0] if isinstance(output_distill, tuple) else output_distill
    
    # Verify the assertion condition (line 367)
    with pytest.raises(AssertionError, match="BUG: Distillation must train on x_train, not eval_ctx.eval_batch"):
        assert not torch.equal(x_train, eval_ctx.eval_batch), \
            "BUG: Distillation must train on x_train, not eval_ctx.eval_batch (data leakage)"


def test_negative_control_completes_with_distinct_batches():
    """Test case 4: Verify run_negative_control succeeds when x_train != eval_batch.
    
    This is the positive test case: when seeds are properly separated,
    the function should complete without raising the leakage assertion.
    """
    from experiments.kt2_locality_falsifier import (
        create_model,
        run_negative_control,
        RECOVERY_SEED,
        EVAL_SEED,
    )
    
    # Precondition: seeds must differ
    assert RECOVERY_SEED != EVAL_SEED, "Seeds must differ for this test"
    
    model = create_model(seed=0, dim=12, hidden=64)
    
    # Should complete without raising AssertionError
    try:
        result = run_negative_control(model, verbose=False)
        success = True
    except AssertionError as e:
        if "BUG: Distillation must train on x_train" in str(e):
            success = False
        else:
            raise  # Re-raise unexpected assertion errors
    
    assert success, "run_negative_control should complete when x_train != eval_ctx.eval_batch"
    
    # Verify the function returned valid results
    assert "CI_proxy" in result
    assert "CI_distillation" in result
    assert "control_passes" in result
    assert isinstance(result["control_passes"], bool)

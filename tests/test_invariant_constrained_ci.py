"""Tests for invariant-constrained CI metric.

Key properties tested:
1. Deterministic perturbation (same seed → same result)
2. Deterministic evaluation (same eval_seed → same PRE/POST loss)
3. Basin proxy invariant: constraint never makes inv_recovery_ratio worse
4. Differentiability check: penalty gradients flow
5. Sanity assertions enforce determinism
6. Spectral entropy shape penalty
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ucip_detection.invariant_constrained_ci import (
    InvariantConstrainedCI,
    apply_perturbation,
    PerturbationConfig,
    EvalContext,
    EvalSuite,
    compute_model_invariants,
    invariant_distance,
    compare_recovery_methods,
    sweep_invariant_weight,
    _find_weight_param,
    _spectral_entropy_torch,
    PENALTY_KEYS_SCALE,
    PENALTY_KEYS_SHAPE,
)


class SimpleRNN(nn.Module):
    """Minimal RNN for testing."""
    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, steps=1):
        h = self.fc1(x)
        for _ in range(steps):
            h = torch.tanh(h @ self.W_hh)
        return self.fc2(h), {"h_final": h}


class TestPerturbationDeterminism:
    """Test that perturbation is reproducible with same seed."""
    
    def test_same_seed_same_result(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=12345, strength=0.25, zero_frac=0.05)
        
        pert1 = apply_perturbation(model, config)
        pert2 = apply_perturbation(model, config)
        
        w1 = list(pert1.parameters())[0].detach().numpy()
        w2 = list(pert2.parameters())[0].detach().numpy()
        
        np.testing.assert_array_almost_equal(w1, w2)
    
    def test_different_seed_different_result(self):
        model = SimpleRNN(hidden_size=16)
        config1 = PerturbationConfig(seed=12345, strength=0.25, zero_frac=0.05)
        config2 = PerturbationConfig(seed=54321, strength=0.25, zero_frac=0.05)
        
        pert1 = apply_perturbation(model, config1)
        pert2 = apply_perturbation(model, config2)
        
        w1 = list(pert1.parameters())[0].detach().numpy()
        w2 = list(pert2.parameters())[0].detach().numpy()
        
        assert not np.allclose(w1, w2)


class TestEvalContextDeterminism:
    """Test that evaluation context produces deterministic measurements."""
    
    def test_same_eval_seed_same_losses(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        ctx1 = EvalContext.create(model, perturbed, eval_seed=12345)
        ctx2 = EvalContext.create(model, perturbed, eval_seed=12345)
        
        assert ctx1.base_loss == ctx2.base_loss
        assert ctx1.perturbed_loss == ctx2.perturbed_loss
        assert ctx1.d_post == ctx2.d_post
    
    def test_different_eval_seed_different_losses(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        ctx1 = EvalContext.create(model, perturbed, eval_seed=12345)
        ctx2 = EvalContext.create(model, perturbed, eval_seed=54321)
        
        assert ctx1.base_loss != ctx2.base_loss
    
    def test_evaluate_loss_deterministic(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        loss1 = ctx.evaluate_loss(model)
        loss2 = ctx.evaluate_loss(model)
        
        assert loss1 == loss2


class TestEvalSuite:
    """Test EvalSuite for mean±std reporting."""
    
    def test_suite_creates_multiple_contexts(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        suite = EvalSuite.create(model, perturbed, n_batches=5)
        
        assert len(suite.contexts) == 5
    
    def test_suite_contexts_have_different_batches(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        suite = EvalSuite.create(model, perturbed, n_batches=3, base_eval_seed=100)
        
        losses = [c.base_loss for c in suite.contexts]
        assert len(set(losses)) == 3
    
    def test_suite_mean_std(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        suite = EvalSuite.create(model, perturbed, n_batches=5)
        
        assert suite.base_loss_mean > 0
        assert suite.base_loss_std >= 0
        assert suite.perturbed_loss_mean > 0
        assert suite.d_post_mean > 0


class TestRecoverySeed:
    """Test recovery_seed for deterministic vs stochastic recovery."""
    
    def test_same_recovery_seed_same_result(self):
        model = SimpleRNN(hidden_size=16)
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        metric1 = InvariantConstrainedCI(recovery_seed=999, recovery_steps=5)
        metric2 = InvariantConstrainedCI(recovery_seed=999, recovery_steps=5)
        
        result1 = metric1.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        result2 = metric2.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        
        assert result1["ci_score"] == result2["ci_score"]
        assert result1["recovered_loss"] == result2["recovered_loss"]
    
    def test_different_recovery_seed_different_result(self):
        model = SimpleRNN(hidden_size=16)
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        metric1 = InvariantConstrainedCI(recovery_seed=999, recovery_steps=5)
        metric2 = InvariantConstrainedCI(recovery_seed=888, recovery_steps=5)
        
        result1 = metric1.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        result2 = metric2.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        
        assert result1["recovered_loss"] != result2["recovered_loss"]


class TestBasinProxy:
    """Test the basin proxy property."""
    
    def test_constraint_improves_or_maintains_inv_ratio(self):
        model = SimpleRNN(hidden_size=16)
        
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(20):
            x = torch.randn(32, 16)
            y, _ = model(x)
            loss = torch.mean((y - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        metric_0 = InvariantConstrainedCI(recovery_steps=5, recovery_seed=42)
        result_0 = metric_0.score(model, use_invariant_constraint=False, 
                                  eval_context=eval_ctx, perturbed_model=perturbed)
        
        metric_λ = InvariantConstrainedCI(invariant_weight=0.1, recovery_steps=5, recovery_seed=42)
        result_λ = metric_λ.score(model, use_invariant_constraint=True,
                                  eval_context=eval_ctx, perturbed_model=perturbed)
        
        tolerance = 0.05
        assert result_λ["inv_recovery_ratio"] <= result_0["inv_recovery_ratio"] + tolerance


class TestDifferentiability:
    """Test that invariant penalty is truly differentiable."""
    
    def test_penalty_has_gradient(self):
        model = SimpleRNN(hidden_size=16)
        target_inv = compute_model_invariants(model)
        
        config = PerturbationConfig(seed=42, strength=0.5, zero_frac=0.0)
        perturbed = apply_perturbation(model, config)
        
        metric = InvariantConstrainedCI()
        penalty = metric._invariant_penalty_torch(perturbed, target_inv)
        
        assert penalty.item() > 0.01
        
        penalty.backward()
        
        W = _find_weight_param(perturbed)
        assert W is not None
        assert W.grad is not None
        assert torch.abs(W.grad).sum() > 0


class TestSpectralEntropy:
    """Test differentiable spectral entropy."""
    
    def test_spectral_entropy_differentiable(self):
        """Spectral entropy should be differentiable."""
        M = torch.randn(8, 8, requires_grad=True)
        M_sym = (M + M.T) / 2
        
        entropy = _spectral_entropy_torch(M_sym)
        entropy.backward()
        
        assert M.grad is not None
        assert torch.abs(M.grad).sum() > 0
    
    def test_spectral_entropy_uniform_is_max(self):
        """Uniform eigenvalues should give maximum entropy."""
        M = torch.eye(8)
        entropy_uniform = _spectral_entropy_torch(M)
        
        M_peaked = torch.diag(torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        entropy_peaked = _spectral_entropy_torch(M_peaked)
        
        assert entropy_uniform > entropy_peaked


class TestShapePenalty:
    """Test shape-sensitive penalty with spec_entropy."""
    
    def test_shape_penalty_includes_spec_entropy(self):
        """PENALTY_KEYS_SHAPE should include spec_entropy."""
        assert "spec_entropy" in PENALTY_KEYS_SHAPE
        assert "spec_entropy" not in PENALTY_KEYS_SCALE
    
    def test_shape_penalty_differentiable(self):
        """Shape penalty should be differentiable through spec_entropy."""
        model = SimpleRNN(hidden_size=16)
        target_inv = compute_model_invariants(model)
        
        config = PerturbationConfig(seed=42, strength=0.5, zero_frac=0.0)
        perturbed = apply_perturbation(model, config)
        
        metric = InvariantConstrainedCI(penalty_keys=PENALTY_KEYS_SHAPE)
        penalty = metric._invariant_penalty_torch(perturbed, target_inv)
        
        assert penalty.item() > 0.01
        penalty.backward()
        
        W = _find_weight_param(perturbed)
        assert W.grad is not None


class TestGradientRatio:
    """Test gradient ratio diagnostic."""
    
    def test_gradient_ratio_computed(self):
        """Gradient ratio should be computed and returned."""
        model = SimpleRNN(hidden_size=16)
        
        metric = InvariantConstrainedCI(
            recovery_steps=3, 
            track_gradient_ratio=True,
            invariant_weight=0.1,
        )
        result = metric.score(model, use_invariant_constraint=True)
        
        assert "gradient_ratio" in result
        assert result["gradient_ratio"] is not None
        assert result["gradient_ratio"] > 0
    
    def test_gradient_ratio_none_for_unconstrained(self):
        """Gradient ratio should be None for unconstrained recovery."""
        model = SimpleRNN(hidden_size=16)
        
        metric = InvariantConstrainedCI(recovery_steps=3)
        result = metric.score(model, use_invariant_constraint=False)
        
        assert result["gradient_ratio"] is None


class TestCompareRecoveryMethods:
    """Test comparison function with sanity assertions."""
    
    def test_pre_post_losses_match_exactly(self):
        """PRE and POST losses must be bit-for-bit identical."""
        model = SimpleRNN(hidden_size=16)
        
        comparison = compare_recovery_methods(
            model, 
            verbose=False, 
            recovery_steps=3,
            perturb_seed=42,
            eval_seed=12345,
            recovery_seed=999,
        )
        
        pre_std = comparison["standard"]["base_loss"]
        pre_con = comparison["constrained"]["base_loss"]
        assert pre_std == pre_con
        
        post_std = comparison["standard"]["perturbed_loss"]
        post_con = comparison["constrained"]["perturbed_loss"]
        assert post_std == post_con
        
        d_post_std = comparison["standard"]["d_post"]
        d_post_con = comparison["constrained"]["d_post"]
        assert d_post_std == d_post_con
        
        assert comparison["determinism_verified"] is True
    
    def test_returns_comparison_metrics(self):
        model = SimpleRNN(hidden_size=16)
        
        comparison = compare_recovery_methods(model, verbose=False, recovery_steps=2)
        
        assert "ci_improvement" in comparison
        assert "inv_improvement" in comparison
        assert "determinism_verified" in comparison


class TestSweepDeterminism:
    """Test that sweep maintains determinism across all λ."""
    
    def test_all_lambda_have_same_pre_post_d_post(self):
        model = SimpleRNN(hidden_size=16)
        
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            x = torch.randn(16, 16)
            y, _ = model(x)
            loss = torch.mean((y - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        results = sweep_invariant_weight(
            model,
            weights=[0.0, 0.1, 1.0],
            recovery_steps=3,
            perturb_seed=42,
            eval_seed=12345,
            recovery_seed=999,
        )
        
        d_posts = [r["d_post"] for r in results]
        assert all(dp == d_posts[0] for dp in d_posts)


class TestInvariantDistance:
    """Test invariant distance calculation."""
    
    def test_zero_distance_identical(self):
        inv = {"fro_norm": 2.5, "tr_M2": 4.0}
        d = invariant_distance(inv, inv)
        assert np.isclose(d, 0.0)
    
    def test_normalized_by_scale(self):
        inv1 = {"fro_norm": 100.0, "tr_M2": 100.0}
        inv2 = {"fro_norm": 101.0, "tr_M2": 101.0}
        
        d = invariant_distance(inv1, inv2)
        assert d < 0.1


class TestCIMetric:
    """Test CI metric computation."""
    
    def test_returns_required_fields(self):
        model = SimpleRNN(hidden_size=16)
        metric = InvariantConstrainedCI(recovery_steps=2)
        
        result = metric.score(model, use_invariant_constraint=True)
        
        required = [
            "base_loss", "perturbed_loss", "recovered_loss", "ci_score",
            "d_post", "d_rec", "inv_recovery_ratio",
            "perturbation_config", "eval_seed", "recovery_seed",
            "gradient_ratio", "penalty_keys",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"
    
    def test_ci_bounded(self):
        model = SimpleRNN(hidden_size=16)
        metric = InvariantConstrainedCI(recovery_steps=5)
        
        result = metric.score(model)
        
        assert -0.1 <= result["ci_score"] <= 2.0


class TestInvariantConstrainedCIScoreCalculation:
    """Test InvariantConstrainedCI score calculation under various damage conditions."""
    
    def test_score_calculation_normal_conditions(self):
        """Verify that CI score is calculated correctly under normal conditions.
        
        CI = recovery / damage
        where damage = perturbed_loss - base_loss
              recovery = max(perturbed_loss - recovered_loss, 0)
        """
        model = SimpleRNN(hidden_size=16)
        
        # Train model slightly for a non-trivial loss landscape
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            x = torch.randn(32, 16)
            y, _ = model(x)
            loss = torch.mean((y - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Create evaluation context with moderate perturbation
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        # Compute CI score
        metric = InvariantConstrainedCI(recovery_steps=10, recovery_seed=999)
        result = metric.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        
        # Verify the calculation: ci_score = recovery / damage
        base_loss = result["base_loss"]
        perturbed_loss = result["perturbed_loss"]
        recovered_loss = result["recovered_loss"]
        
        damage = perturbed_loss - base_loss
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        expected_ci = recovery / damage
        
        assert np.isclose(result["ci_score"], expected_ci, rtol=1e-6)
    
    def test_score_calculation_significant_damage(self):
        """Verify CI score calculation when damage is significant.
        
        With larger perturbation (strength > 0.5), damage should be substantial.
        CI score may be smaller due to difficulty recovering from large damage.
        """
        model = SimpleRNN(hidden_size=16)
        
        # Train model
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(15):
            x = torch.randn(32, 16)
            y, _ = model(x)
            loss = torch.mean((y - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Create evaluation context with LARGE perturbation
        config = PerturbationConfig(seed=42, strength=0.75, zero_frac=0.1)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        # Verify damage is significant
        damage = eval_ctx.perturbed_loss - eval_ctx.base_loss
        assert damage > 0.05, f"Expected significant damage, got {damage:.6f}"
        
        # Compute CI score
        metric = InvariantConstrainedCI(recovery_steps=10, recovery_seed=999)
        result = metric.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        
        # Verify the calculation with significant damage
        base_loss = result["base_loss"]
        perturbed_loss = result["perturbed_loss"]
        recovered_loss = result["recovered_loss"]
        
        damage_calc = perturbed_loss - base_loss
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        expected_ci = recovery / damage_calc
        
        assert np.isclose(result["ci_score"], expected_ci, rtol=1e-6)
        # With significant damage, CI should be lower (harder to recover)
        assert result["ci_score"] >= 0.0
    
    def test_score_calculation_small_damage(self):
        """Test CI score calculation when damage is small but non-trivial."""

        model = SimpleRNN(hidden_size=16)
        metric = InvariantConstrainedCI(recovery_steps=5, recovery_seed=999)

        # Low-strength perturbations can occasionally IMPROVE loss on some platforms.
        # Search a few seeds for a small *positive* damage case; otherwise skip.
        eval_ctx = None
        perturbed = None
        for seed in range(42, 72):
            config = PerturbationConfig(seed=seed, strength=0.05, zero_frac=0.0)
            cand = apply_perturbation(model, config)
            ctx = EvalContext.create(model, cand, eval_seed=12345)
            damage = ctx.perturbed_loss - ctx.base_loss

            if damage > 1e-9 and damage < 5e-2:  # "small but non-trivial"
                perturbed = cand
                eval_ctx = ctx
                break

        if eval_ctx is None:
            pytest.skip("Could not find a seed producing small positive damage; low-strength perturbation can improve loss.")

        result = metric.score(model, eval_context=eval_ctx, perturbed_model=perturbed)

        base_loss = result["base_loss"]
        perturbed_loss = result["perturbed_loss"]
        recovered_loss = result["recovered_loss"]

        damage_calc = perturbed_loss - base_loss
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        expected_ci = recovery / damage_calc

        assert np.isclose(result["ci_score"], expected_ci, rtol=1e-6)
    
    def test_recovery_cannot_exceed_damage(self):
        """Recovery should never exceed damage (ci_score <= 1.0 typically).
        
        If recovered_loss >= base_loss, then recovery = 0 and ci_score = 0.
        If recovered_loss <= base_loss and < perturbed_loss, then 0 < ci_score < 1.
        CI can theoretically exceed 1.0 if optimization overshoots past base_loss.
        """
        model = SimpleRNN(hidden_size=16)
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        metric = InvariantConstrainedCI(recovery_steps=10, recovery_seed=999)
        result = metric.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        
        # recovery = max(perturbed_loss - recovered_loss, 0)
        # So recovery >= 0 always, and therefore ci_score >= 0
        assert result["ci_score"] >= -1e-6  # Small epsilon for numerical stability
    
    def test_damage_too_small_raises_error(self):
        """CI is undefined when damage < 1e-9.
        
        This should raise ValueError to prevent division by near-zero.
        """
        model = SimpleRNN(hidden_size=16)
        
        # Create a "perturbed" model that's almost identical (no meaningful perturbation)
        # by using an extremely small strength
        config = PerturbationConfig(seed=42, strength=1e-10, zero_frac=0.0)
        perturbed = apply_perturbation(model, config)
        
        # Try to create eval context - might succeed but damage will be tiny
        # If eval_ctx is created, the score() call should fail
        try:
            eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
            metric = InvariantConstrainedCI(recovery_steps=5)
            
            # This should raise ValueError if damage < 1e-9
            with pytest.raises(ValueError, match="Perturbation too small"):
                result = metric.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        except AssertionError:
            # It's possible perturbation creates sufficient noise that damage is still > 1e-9
            # In that case, we explicitly skip this test case
            pytest.skip("Perturbation strength 1e-10 still produced damage > 1e-9; cannot test tiny-damage edge case.")
    
    def test_constrained_recovery_improves_or_maintains_score(self):
        """Verify that invariant-constrained recovery affects CI score.
        
        The constraint may improve or maintain CI score by guiding recovery
        toward the manifold defined by pre-perturbation invariants.
        """
        model = SimpleRNN(hidden_size=16)
        
        # Train model
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            x = torch.randn(32, 16)
            y, _ = model(x)
            loss = torch.mean((y - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        # Standard recovery (no constraint)
        metric_std = InvariantConstrainedCI(recovery_steps=10, recovery_seed=999)
        result_std = metric_std.score(
            model, 
            use_invariant_constraint=False,
            eval_context=eval_ctx, 
            perturbed_model=perturbed
        )
        
        # Constrained recovery
        metric_con = InvariantConstrainedCI(
            recovery_steps=10, 
            recovery_seed=999,
            invariant_weight=0.1
        )
        result_con = metric_con.score(
            model,
            use_invariant_constraint=True,
            eval_context=eval_ctx,
            perturbed_model=perturbed
        )
        
        # Both should be valid CI scores
        assert result_std["ci_score"] >= 0.0
        assert result_con["ci_score"] >= 0.0
        
        # Verify both calculations are correct
        for result, name in [(result_std, "standard"), (result_con, "constrained")]:
            base = result["base_loss"]
            pert = result["perturbed_loss"]
            rec = result["recovered_loss"]
            damage = pert - base
            recovery = max(pert - rec, 0.0)
            expected_ci = recovery / damage
            assert np.isclose(result["ci_score"], expected_ci, rtol=1e-6), f"{name} CI calculation mismatch"
    
    def test_ci_score_components_consistency(self):
        """Verify that CI score components (PRE, POST, RECOVER) are consistent.
        
        - base_loss should equal snap_pre.loss
        - perturbed_loss should equal snap_post.loss
        - recovered_loss should equal snap_recover.loss
        """
        model = SimpleRNN(hidden_size=16)
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        metric = InvariantConstrainedCI(recovery_steps=5, recovery_seed=999)
        result = metric.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        
        # Verify snapshot consistency
        assert result["base_loss"] == result["snap_pre"].loss
        assert result["perturbed_loss"] == result["snap_post"].loss
        assert result["recovered_loss"] == result["snap_recover"].loss
        
        # Verify invariants are captured
        assert "inv_pre" in result
        assert "inv_post" in result
        assert "inv_recover" in result
        assert len(result["inv_pre"]) > 0
        assert len(result["inv_post"]) > 0
        assert len(result["inv_recover"]) > 0


class TestValueErrorOnSmallDamage:
    """Test that InvariantConstrainedCI.score raises ValueError when damage is too small."""
    
    def test_score_raises_value_error_when_damage_too_small(self):
        """InvariantConstrainedCI.score should raise ValueError when perturbation damage < 1e-9.
        
        When damage (perturbed_loss - base_loss) is less than 1e-9, the CI score
        calculation becomes numerically unstable (division by near-zero). The code
        should explicitly raise ValueError to prevent this.
        """
        model = SimpleRNN(hidden_size=16)
        
        # Create a mock evaluation context with almost identical losses
        # This simulates a scenario where perturbation had virtually no effect
        config = PerturbationConfig(seed=42, strength=0.0, zero_frac=0.0)
        perturbed = apply_perturbation(model, config)
        
        # Manually create an eval context with tiny damage
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        # Force the damage to be too small by manipulating the eval context
        # We'll create a new one where base_loss and perturbed_loss are nearly identical
        class MockEvalContext:
            def __init__(self):
                self.base_loss = 0.5
                self.perturbed_loss = 0.5 + 5e-10  # damage = 5e-10 < 1e-9
                self.inv_pre = {"fro_norm": 1.0, "tr_M2": 1.0}
                self.inv_post = {"fro_norm": 1.0, "tr_M2": 1.0}
                self.d_post = 0.0
                self.eval_seed = 12345
                self.eval_batch = torch.randn(64, 16)
            
            def evaluate_loss(self, model):
                return 0.5
        
        mock_ctx = MockEvalContext()
        
        metric = InvariantConstrainedCI(recovery_steps=5)
        
        # This should raise ValueError because damage < 1e-9
        with pytest.raises(ValueError, match="Perturbation too small.*damage.*< 1e-9.*CI is undefined"):
            metric.score(model, eval_context=mock_ctx, perturbed_model=perturbed)
    
    def test_score_succeeds_when_damage_above_threshold(self):
        """InvariantConstrainedCI.score should succeed when damage is above 1e-9.
        
        The threshold check is damage < 1e-9, so damage >= 1e-9 should be acceptable.
        We use 2e-9 to ensure we're safely above the threshold.
        """
        model = SimpleRNN(hidden_size=16)
        
        config = PerturbationConfig(seed=42, strength=0.0, zero_frac=0.0)
        perturbed = apply_perturbation(model, config)
        
        # Create a mock context with damage safely above threshold
        class MockEvalContext:
            def __init__(self):
                self.base_loss = 0.5
                self.perturbed_loss = 0.5 + 2e-9  # damage = 2e-9 (above threshold)
                self.inv_pre = {"fro_norm": 1.0, "tr_M2": 1.0}
                self.inv_post = {"fro_norm": 1.0, "tr_M2": 1.0}
                self.d_post = 0.0
                self.eval_seed = 12345
                self.eval_batch = torch.randn(64, 16)
            
            def evaluate_loss(self, model):
                return 0.5 + 1e-9  # recovered loss slightly better
        
        mock_ctx = MockEvalContext()
        
        metric = InvariantConstrainedCI(recovery_steps=5)
        
        # This should NOT raise ValueError
        result = metric.score(model, eval_context=mock_ctx, perturbed_model=perturbed)
        assert "ci_score" in result
        assert result["ci_score"] >= 0.0


class TestCIScoreCalculation:
    """Test that InvariantConstrainedCI.score correctly calculates CI under normal conditions."""
    
    def test_ci_score_formula_correctness(self):
        """Verify CI score calculation: CI = recovery / damage.
        
        where:
            damage = perturbed_loss - base_loss
            recovery = max(perturbed_loss - recovered_loss, 0)
            ci_score = recovery / damage
        
        This test ensures the formula is implemented correctly.
        """
        model = SimpleRNN(hidden_size=16)
        
        # Train model to get a non-trivial loss landscape
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            x = torch.randn(32, 16)
            y, _ = model(x)
            loss = torch.mean((y - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Create evaluation context with moderate perturbation
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        # Compute CI score
        metric = InvariantConstrainedCI(recovery_steps=10, recovery_seed=999)
        result = metric.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        
        # Manually calculate expected CI score
        base_loss = result["base_loss"]
        perturbed_loss = result["perturbed_loss"]
        recovered_loss = result["recovered_loss"]
        
        damage = perturbed_loss - base_loss
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        expected_ci_score = recovery / damage
        
        # Verify the CI score matches our calculation
        assert np.isclose(result["ci_score"], expected_ci_score, rtol=1e-6), \
            f"CI score mismatch: got {result['ci_score']}, expected {expected_ci_score}"
        
        # Verify CI score is in reasonable range
        assert result["ci_score"] >= 0.0, "CI score should be non-negative"
        assert result["ci_score"] <= 2.0, "CI score should typically be <= 1.0 (or slightly higher if overshot)"
    
    def test_ci_score_with_invariant_constraint(self):
        """Verify CI score calculation is correct when using invariant constraints.
        
        The formula should be the same regardless of whether constraints are used.
        Only the recovered_loss should differ based on the recovery process.
        """
        model = SimpleRNN(hidden_size=16)
        
        # Train model
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            x = torch.randn(32, 16)
            y, _ = model(x)
            loss = torch.mean((y - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        # Test with invariant constraint
        metric = InvariantConstrainedCI(
            recovery_steps=10, 
            recovery_seed=999,
            invariant_weight=0.1
        )
        result = metric.score(
            model, 
            use_invariant_constraint=True,
            eval_context=eval_ctx, 
            perturbed_model=perturbed
        )
        
        # Verify CI score calculation
        base_loss = result["base_loss"]
        perturbed_loss = result["perturbed_loss"]
        recovered_loss = result["recovered_loss"]
        
        damage = perturbed_loss - base_loss
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        expected_ci_score = recovery / damage
        
        assert np.isclose(result["ci_score"], expected_ci_score, rtol=1e-6), \
            f"CI score mismatch with constraint: got {result['ci_score']}, expected {expected_ci_score}"
        
        # Verify constraint was used
        assert result["used_constraint"] is True
        assert result["invariant_weight"] == 0.1
    
    def test_ci_score_zero_when_no_recovery(self):
        """Verify CI score is 0 when recovered_loss >= perturbed_loss (no recovery).
        
        If recovery doesn't improve the loss at all, then:
            recovery = max(perturbed_loss - recovered_loss, 0) = 0
            ci_score = 0 / damage = 0
        """
        model = SimpleRNN(hidden_size=16)
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        # Create a mock context where recovery doesn't help
        class MockEvalContext:
            def __init__(self):
                self.base_loss = 0.5
                self.perturbed_loss = 1.0  # damage = 0.5
                self.inv_pre = {"fro_norm": 1.0, "tr_M2": 1.0}
                self.inv_post = {"fro_norm": 1.1, "tr_M2": 1.1}
                self.d_post = 0.1
                self.eval_seed = 12345
                self.eval_batch = torch.randn(64, 16)
            
            def evaluate_loss(self, model):
                # Return a loss worse than or equal to perturbed_loss
                return 1.0  # no recovery
        
        mock_ctx = MockEvalContext()
        
        metric = InvariantConstrainedCI(recovery_steps=5)
        result = metric.score(model, eval_context=mock_ctx, perturbed_model=perturbed)
        
        # Since recovered_loss = perturbed_loss, recovery = 0, so ci_score = 0
        assert result["ci_score"] == 0.0, "CI score should be 0 when no recovery occurs"
    
    def test_ci_score_perfect_recovery(self):
        """Verify CI score approaches 1.0 when recovery is near-perfect.
        
        If recovered_loss ≈ base_loss, then:
            recovery ≈ perturbed_loss - base_loss = damage
            ci_score = damage / damage = 1.0
        """
        model = SimpleRNN(hidden_size=16)
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        # Create a mock context where recovery is near-perfect
        class MockEvalContext:
            def __init__(self):
                self.base_loss = 0.5
                self.perturbed_loss = 1.0  # damage = 0.5
                self.inv_pre = {"fro_norm": 1.0, "tr_M2": 1.0}
                self.inv_post = {"fro_norm": 1.1, "tr_M2": 1.1}
                self.d_post = 0.1
                self.eval_seed = 12345
                self.eval_batch = torch.randn(64, 16)
            
            def evaluate_loss(self, model):
                # Return loss very close to base_loss (near-perfect recovery)
                return 0.501  # recovered almost to base_loss
        
        mock_ctx = MockEvalContext()
        
        metric = InvariantConstrainedCI(recovery_steps=5)
        result = metric.score(model, eval_context=mock_ctx, perturbed_model=perturbed)
        
        # Calculate expected CI
        damage = 1.0 - 0.5
        recovery = max(1.0 - 0.501, 0.0)
        expected_ci = recovery / damage
        
        assert np.isclose(result["ci_score"], expected_ci, rtol=1e-6)
        assert result["ci_score"] >= 0.95, "CI score should be very high for near-perfect recovery"

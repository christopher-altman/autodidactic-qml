"""Tests for SQNT module."""

import numpy as np
import pytest

# Import SQNT components
import sys
sys.path.insert(0, '..')

from sqnt import (
    SpectralMassInvariant,
    apply_invariant_correction,
    TopologyPlasticity,
    TopologyFeedback,
    random_rewire,
    apply_perturbation,
    SystemMode,
    SQNTConfig,
    SQNTSystem,
    run_comparison_experiment,
    KT1Protocol,
    quick_falsification_test,
)


class TestSpectralMassInvariant:
    """Tests for SpectralMassInvariant class."""
    
    def test_initialization(self):
        """Test invariant initialization."""
        inv = SpectralMassInvariant(epsilon=1e-4)
        M0 = np.random.randn(10, 10)
        M0 = (M0 + M0.T) / 2
        
        I0 = inv.initialize(M0)
        
        assert I0 > 0
        assert inv.I_0 == I0
        assert np.isclose(I0, np.linalg.norm(M0, 'fro') ** 2)
    
    def test_violation_computation(self):
        """Test violation is computed correctly."""
        inv = SpectralMassInvariant()
        M0 = np.eye(10)
        inv.initialize(M0)
        
        # Same matrix → zero violation
        assert np.isclose(inv.violation(M0), 0.0)
        
        # Scaled matrix → positive/negative violation
        M_scaled = 2.0 * M0
        violation = inv.violation(M_scaled)
        expected = np.linalg.norm(M_scaled, 'fro') ** 2 - inv.I_0
        assert np.isclose(violation, expected)
    
    def test_correction_term_direction(self):
        """Test correction term points toward budget."""
        inv = SpectralMassInvariant(epsilon=0.1)
        M0 = np.eye(5)
        inv.initialize(M0)
        
        # If we're over budget, correction should shrink matrix
        M_big = 2.0 * M0
        correction = inv.correction_term(M_big)
        
        # Correction should be negative (reducing the matrix)
        assert np.trace(correction) < 0
        
        # If we're under budget, correction should grow matrix
        M_small = 0.5 * M0
        correction_small = inv.correction_term(M_small)
        assert np.trace(correction_small) > 0
    
    def test_corrected_update(self):
        """Test full corrected update."""
        inv = SpectralMassInvariant(epsilon=0.01)
        M0 = np.random.randn(8, 8)
        M0 = (M0 + M0.T) / 2
        inv.initialize(M0)
        
        # Small update
        delta_M = 0.1 * np.random.randn(8, 8)
        delta_M = (delta_M + delta_M.T) / 2
        
        M_new, diag = inv.corrected_update(M0, delta_M)
        
        assert M_new.shape == M0.shape
        assert np.allclose(M_new, M_new.T)  # Symmetry preserved
        assert 'delta_I' in diag
        assert 'correction_norm' in diag
    
    def test_stability_over_time(self):
        """Test that invariant keeps spectral mass stable."""
        rng = np.random.default_rng(0)
        inv = SpectralMassInvariant(epsilon=0.01)
        M = rng.standard_normal((10, 10))
        M = (M + M.T) / 2
        I0 = inv.initialize(M)
        
        # Apply many updates
        for _ in range(100):
            delta_M = 0.1 * rng.standard_normal((10, 10))
            delta_M = (delta_M + delta_M.T) / 2
            M, _ = inv.corrected_update(M, delta_M)
            inv.log_step(M)
        
        # Spectral mass should stay close to I0
        final_I = inv.spectral_mass(M)
        relative_change = abs(final_I - I0) / I0
        
        # Allow some drift but not extreme
        assert relative_change < 1.0, f"Spectral mass drifted too much: {relative_change}"


class TestTopologyPlasticity:
    """Tests for TopologyPlasticity class."""
    
    def test_participation_computation(self):
        """Test participation scores are computed."""
        plasticity = TopologyPlasticity(eta=1e-3, n_dominant=3)
        
        # Simple graph
        n = 10
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        
        P = plasticity.compute_participation(A)
        
        assert P.shape == (n, n)
        assert np.allclose(P, P.T)  # Symmetric
        assert np.allclose(np.diag(P), 0)  # Zero diagonal
        assert np.all(P >= 0)  # Non-negative
    
    def test_centering(self):
        """Test centering preserves zero mean."""
        plasticity = TopologyPlasticity()
        
        n = 10
        P = np.random.rand(n, n)
        P = (P + P.T) / 2
        np.fill_diagonal(P, 0)
        
        delta_P = plasticity.center_participation(P)
        
        # Off-diagonal mean should be near zero
        mask = ~np.eye(n, dtype=bool)
        mean_centered = np.mean(delta_P[mask])
        assert abs(mean_centered) < 1e-10
    
    def test_topology_update(self):
        """Test topology update produces valid adjacency."""
        plasticity = TopologyPlasticity(eta=0.1)
        
        n = 8
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        A = A / np.max(A)  # Normalize
        
        A_new, diag = plasticity.update(A)
        
        assert A_new.shape == A.shape
        assert np.allclose(A_new, A_new.T)  # Symmetric
        assert np.allclose(np.diag(A_new), 0)  # Zero diagonal
        assert 'participation_mean' in diag
        assert 'weight_change_norm' in diag


class TestTopologyFeedback:
    """Tests for TopologyFeedback class."""
    
    def test_hadamard_modulation(self):
        """Test Hadamard modulation."""
        feedback = TopologyFeedback()
        
        n = 5
        delta_M = np.random.randn(n, n)
        delta_M = (delta_M + delta_M.T) / 2
        
        W = np.ones((n, n)) * 0.5
        np.fill_diagonal(W, 0)
        
        delta_M_mod = feedback.modulate_update(delta_M, W, mode='hadamard')
        
        assert delta_M_mod.shape == delta_M.shape
        # Modulated update should be scaled down (W < 1)
        assert np.linalg.norm(delta_M_mod) <= np.linalg.norm(delta_M)


class TestRandomRewire:
    """Tests for random_rewire function."""
    
    def test_preserves_shape(self):
        """Test rewiring preserves matrix shape."""
        n = 10
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        
        A_rewired = random_rewire(A, seed=42)
        
        assert A_rewired.shape == A.shape
        assert np.allclose(np.diag(A_rewired), 0)
    
    def test_different_each_time(self):
        """Test different seeds give different results."""
        n = 10
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        
        A1 = random_rewire(A, seed=1)
        A2 = random_rewire(A, seed=2)
        
        # Should be different
        assert not np.allclose(A1, A2)


class TestApplyPerturbation:
    """Tests for apply_perturbation function."""
    
    def test_removes_edges(self):
        """Test perturbation removes edges."""
        n = 10
        A = np.ones((n, n))
        np.fill_diagonal(A, 0)
        
        n_edges_before = np.sum(A > 0)
        
        A_perturbed, info = apply_perturbation(A, fraction=0.25, mode='random', seed=42)
        
        n_edges_after = np.sum(A_perturbed > 0)
        
        assert n_edges_after < n_edges_before
        assert 'n_edges_removed' in info
    
    def test_contiguous_mode(self):
        """Test contiguous perturbation affects local region."""
        n = 20
        A = np.ones((n, n))
        np.fill_diagonal(A, 0)
        
        A_perturbed, info = apply_perturbation(A, fraction=0.25, mode='contiguous', seed=42)
        
        assert 'affected_nodes' in info
        assert len(info['affected_nodes']) > 0


class TestSQNTSystem:
    """Tests for SQNTSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n = 10
        self.M0 = np.random.randn(self.n, self.n)
        self.M0 = (self.M0 + self.M0.T) / 2
        
        self.A0 = np.random.rand(self.n, self.n)
        self.A0 = (self.A0 + self.A0.T) / 2
        np.fill_diagonal(self.A0, 0)
        self.A0 = self.A0 / np.max(self.A0)
    
    def test_initialization(self):
        """Test system initialization."""
        config = SQNTConfig(mode=SystemMode.SQNT, seed=42)
        system = SQNTSystem(config)
        
        state = system.initialize(self.M0, self.A0)
        
        assert state.t == 0
        assert state.M.shape == self.M0.shape
        assert state.A.shape == self.A0.shape
    
    def test_step_sqnt_mode(self):
        """Test evolution step in SQNT mode."""
        config = SQNTConfig(mode=SystemMode.SQNT, seed=42)
        system = SQNTSystem(config)
        system.initialize(self.M0, self.A0)
        
        def simple_dynamics(M, A):
            return -0.01 * M
        
        state = system.step(simple_dynamics)
        
        assert state.t == 1
        assert 'spectral_mass' in state.__dict__
    
    def test_step_frozen_mode(self):
        """Test evolution step in frozen mode."""
        config = SQNTConfig(mode=SystemMode.FROZEN, seed=42)
        system = SQNTSystem(config)
        state0 = system.initialize(self.M0, self.A0)
        A_initial = state0.A.copy()
        
        def simple_dynamics(M, A):
            return -0.01 * M
        
        state = system.step(simple_dynamics)
        
        # Topology should be unchanged
        assert np.allclose(state.A, A_initial)
    
    def test_step_random_mode(self):
        """Test evolution step in random mode."""
        config = SQNTConfig(mode=SystemMode.RANDOM, seed=42)
        system = SQNTSystem(config)
        state0 = system.initialize(self.M0, self.A0)
        A_initial = state0.A.copy()
        
        def simple_dynamics(M, A):
            return -0.01 * M
        
        state = system.step(simple_dynamics)
        
        # Topology should be different
        assert not np.allclose(state.A, A_initial)
    
    def test_perturbation(self):
        """Test applying perturbation."""
        config = SQNTConfig(mode=SystemMode.SQNT, seed=42)
        system = SQNTSystem(config)
        system.initialize(self.M0, self.A0)
        
        info = system.apply_perturbation(fraction=0.3)
        
        assert 'n_edges_removed' in info
        assert info['n_edges_removed'] > 0
    
    def test_recovery_metrics(self):
        """Test computing recovery metrics."""
        config = SQNTConfig(mode=SystemMode.SQNT, seed=42)
        system = SQNTSystem(config)
        system.initialize(self.M0, self.A0)
        
        baseline = system.get_baseline_snapshot()
        
        # Evolve a bit
        def simple_dynamics(M, A):
            return -0.01 * M
        
        for _ in range(10):
            system.step(simple_dynamics)
        
        metrics = system.compute_recovery_metrics(baseline)
        
        assert 'spectral_distance_L' in metrics
        assert 'topology_similarity' in metrics


class TestRunComparisonExperiment:
    """Tests for run_comparison_experiment function."""
    
    def test_runs_all_modes(self):
        """Test experiment runs all three modes."""
        n = 8
        M0 = np.random.randn(n, n)
        M0 = (M0 + M0.T) / 2
        
        A0 = np.random.rand(n, n)
        A0 = (A0 + A0.T) / 2
        np.fill_diagonal(A0, 0)
        
        def dynamics(M, A):
            return -0.01 * M
        
        results = run_comparison_experiment(
            M0=M0,
            A0=A0,
            dynamics=dynamics,
            n_steps=20,
            perturb_at=10,
            seed=42,
        )
        
        assert 'sqnt' in results
        assert 'frozen' in results
        assert 'random' in results


class TestKT1Protocol:
    """Tests for KT1Protocol class."""
    
    def test_protocol_runs(self):
        """Test protocol executes without error."""
        n = 6
        M0 = np.random.randn(n, n)
        M0 = (M0 + M0.T) / 2
        
        A0 = np.random.rand(n, n)
        A0 = (A0 + A0.T) / 2
        np.fill_diagonal(A0, 0)
        
        def dynamics(M, A):
            return -0.01 * M
        
        protocol = KT1Protocol(
            n_warmup=5,
            n_post_perturbation=10,
            n_seeds=2,
        )
        
        result = protocol.run(M0, A0, dynamics, verbose=False)
        
        assert hasattr(result, 'passed')
        assert hasattr(result, 'failure_reasons')
        assert hasattr(result, 'metrics')


class TestSanity:
    """Basic sanity tests."""
    
    def test_imports(self):
        """Test all exports are importable."""
        from sqnt import (
            SpectralMassInvariant,
            apply_invariant_correction,
            TopologyPlasticity,
            TopologyFeedback,
            random_rewire,
            apply_perturbation,
            SystemMode,
            SQNTConfig,
            SQNTState,
            SQNTSystem,
            run_comparison_experiment,
            FalsificationResult,
            KT1Protocol,
            quick_falsification_test,
        )
        assert True
    
    def test_functional_interface(self):
        """Test functional apply_invariant_correction."""
        M = np.eye(5)
        delta_M = 0.1 * np.ones((5, 5))
        delta_M = (delta_M + delta_M.T) / 2
        I_0 = np.linalg.norm(M, 'fro') ** 2
        
        M_new = apply_invariant_correction(M, delta_M, I_0, epsilon=0.01)
        
        assert M_new.shape == M.shape
        assert np.allclose(M_new, M_new.T)

"""Tests for experiment modules.

Tests cover:
- topology_kernel_benchmark
- graph_family_sweep
- phase_diagram_visualization
- sqnt_loop_topology_dynamics (formerly sqnt_loop_ucip_bridge)
- ucip_protocol
"""

import numpy as np
import pytest
import json
import tempfile
import os

# Import experiment functions
from experiments.topology_kernel_benchmark import (
    generate_regression_data,
    kernel_ridge_regression,
    compute_kernel_advantage,
    run_experiment,
)
from experiments.graph_family_sweep import (
    generate_data,
    krr_predict,
    evaluate_kernel,
    run_family_experiment,
)
from experiments.phase_diagram_visualization import (
    compute_advantage,
    generate_family_comparison_data,
)
from experiments.sqnt_loop_topology_dynamics import (
    compute_tld_metrics,
    TLDMetrics,
)
from experiments.ucip_protocol import (
    UCIPProtocol,
    SimpleUCIPAgent,
    UCIPMetrics,
    AttackResult,
)

from topology import Graph, matrix_to_graph, GraphFamily, sample_graph_family
from matrix_models import HermitianEnsemble


class TestTopologyKernelBenchmark:
    """Tests for topology_kernel_benchmark module."""

    def test_generate_regression_data_shapes(self):
        """Test regression data generation shapes."""
        X_train, X_test, y_train, y_test = generate_regression_data(
            n_samples=50, dim=10, seed=42
        )

        assert X_train.shape[0] == 35  # 70% train
        assert X_test.shape[0] == 15   # 30% test
        assert X_train.shape[1] == 10
        assert len(y_train) == 35
        assert len(y_test) == 15

    def test_generate_regression_data_reproducible(self):
        """Test regression data is reproducible with seed."""
        X1, _, y1, _ = generate_regression_data(50, 10, seed=42)
        X2, _, y2, _ = generate_regression_data(50, 10, seed=42)

        assert np.allclose(X1, X2)
        assert np.allclose(y1, y2)

    def test_kernel_ridge_regression(self):
        """Test kernel ridge regression prediction."""
        n_train = 20
        n_test = 10

        # Simple linear kernel
        X = np.random.default_rng(42).normal(size=(n_train + n_test, 5))
        K = X @ X.T
        K_train = K[:n_train, :n_train]
        K_test = K[n_train:, :n_train]
        y_train = np.random.default_rng(42).normal(size=n_train)

        y_pred = kernel_ridge_regression(K_train, y_train, K_test)

        assert y_pred.shape == (n_test,)
        assert np.all(np.isfinite(y_pred))

    def test_compute_kernel_advantage_structure(self):
        """Test kernel advantage computation returns proper structure."""
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(20, 8))
        X_test = rng.normal(size=(10, 8))
        y_train = rng.normal(size=20)
        y_test = rng.normal(size=10)

        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)

        result = compute_kernel_advantage(
            X_train, X_test, y_train, y_test,
            graph=G, n_qubits=3, n_layers=1, seed=42
        )

        required_keys = [
            "mse_graph_conditioned",
            "mse_standard_quantum",
            "mse_classical",
            "r2_graph_conditioned",
            "r2_standard_quantum",
            "r2_classical",
            "advantage_gc_vs_classical",
            "advantage_gc_vs_standard",
        ]
        for key in required_keys:
            assert key in result
            assert isinstance(result[key], float)


class TestGraphFamilySweep:
    """Tests for graph_family_sweep module."""

    def test_generate_data_shapes(self):
        """Test data generation shapes."""
        X_train, X_test, y_train, y_test = generate_data(
            n_samples=40, dim=8, seed=42
        )

        n_train = int(0.7 * 40)
        assert X_train.shape[0] == n_train
        assert X_test.shape[0] == 40 - n_train

    def test_krr_predict(self):
        """Test KRR prediction function."""
        n = 10
        K_train = np.eye(n) + 0.1 * np.random.randn(n, n)
        K_train = (K_train + K_train.T) / 2
        y_train = np.random.randn(n)
        K_test = np.random.randn(5, n)

        y_pred = krr_predict(K_train, y_train, K_test)

        assert y_pred.shape == (5,)

    def test_evaluate_kernel_returns_metrics(self):
        """Test kernel evaluation returns all metrics."""
        n_train, n_test = 15, 5

        K_train = np.eye(n_train) + 0.1 * np.random.randn(n_train, n_train)
        K_train = (K_train + K_train.T) / 2
        K_test = np.random.randn(n_test, n_train)
        y_train = np.random.randn(n_train)
        y_test = np.random.randn(n_test)

        metrics = evaluate_kernel(K_train, K_test, y_train, y_test)

        assert "mse" in metrics
        assert "r2" in metrics
        assert "alignment" in metrics

    def test_run_family_experiment_structure(self):
        """Test family experiment returns proper structure."""
        result = run_family_experiment(
            family=GraphFamily.ERDOS_RENYI,
            n_nodes=6,
            n_qubits=3,
            n_layers=1,
            n_samples=15,
            seed=42,
        )

        required_keys = [
            "family",
            "graph_summary",
            "graph_conditioned",
            "standard_quantum",
            "classical_rbf",
            "advantage_gc_vs_classical",
        ]
        for key in required_keys:
            assert key in result

    @pytest.mark.parametrize("family", list(GraphFamily)[:3])  # Test subset
    def test_all_families_run(self, family):
        """Test experiment runs for different families."""
        result = run_family_experiment(
            family=family,
            n_nodes=6,
            n_qubits=3,
            n_layers=1,
            n_samples=15,
            seed=42,
        )

        assert "graph_conditioned" in result
        assert "r2" in result["graph_conditioned"]


class TestPhaseDiagramVisualization:
    """Tests for phase_diagram_visualization module."""

    def test_compute_advantage_returns_metrics(self):
        """Test advantage computation returns metrics."""
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(10, 6))
        X_test = rng.normal(size=(5, 6))
        y_train = rng.normal(size=10)
        y_test = rng.normal(size=5)

        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=6, seed=42)

        result = compute_advantage(
            X_train, X_test, y_train, y_test,
            G, n_qubits=3, n_layers=1, seed=42
        )

        assert "r2_quantum" in result
        assert "r2_classical" in result
        assert "advantage" in result

    def test_generate_family_comparison_data(self):
        """Test family comparison data generation."""
        result = generate_family_comparison_data(
            n_nodes=6,
            n_qubits=3,
            n_layers=1,
            n_samples=15,
            n_trials=2,
            seed=42,
        )

        # Should have entry for each family
        for family in GraphFamily:
            assert family.value in result
            assert "mean_advantage" in result[family.value]
            assert "std_advantage" in result[family.value]


class TestSQNTLoopTopologyDynamics:
    """Tests for sqnt_loop_topology_dynamics module (formerly sqnt_loop_ucip_bridge)."""

    def test_tld_metrics_dataclass(self):
        """Test TLDMetrics dataclass."""
        metrics = TLDMetrics(
            topology_influence_score=0.5,
            recovery_time=10.0,
            stability_metric=0.1,
            phase_transition_detected=True,
            learning_efficiency=0.8,
            loss_history=[1.0, 0.8, 0.6],
        )

        assert metrics.topology_influence_score == 0.5
        assert metrics.recovery_time == 10.0
        assert len(metrics.loss_history) == 3

    def test_compute_tld_metrics_returns_proper_type(self):
        """Test TLD metrics computation."""
        rng = np.random.default_rng(42)
        M = rng.normal(size=(8, 8))
        M = (M + M.T) / 2

        _, _, G = sample_graph_family(GraphFamily.ERDOS_RENYI, n=8, seed=42)

        metrics = compute_tld_metrics(
            M=M, graph=G, n_steps=30, lr=0.01, seed=42
        )

        assert isinstance(metrics, TLDMetrics)
        assert 0 <= metrics.topology_influence_score <= 1 or np.isnan(metrics.topology_influence_score) is False
        assert metrics.recovery_time > 0
        assert len(metrics.loss_history) == 30

    def test_tld_topology_influence_bounded(self):
        """Test topology influence score is properly bounded."""
        rng = np.random.default_rng(42)

        for seed in range(3):
            M = rng.normal(size=(6, 6))
            M = (M + M.T) / 2
            _, _, G = sample_graph_family(GraphFamily.COMPLETE, n=6, seed=seed)

            metrics = compute_tld_metrics(M, G, n_steps=20, lr=0.01, seed=seed)

            # Topology influence score should be in [0, 1] (correlation coefficient)
            assert 0 <= metrics.topology_influence_score <= 1 or np.isnan(metrics.topology_influence_score)

    def test_tld_loss_history_recorded(self):
        """Test loss history is recorded."""
        rng = np.random.default_rng(42)
        M = rng.normal(size=(8, 8))
        M = (M + M.T) / 2
        _, _, G = sample_graph_family(GraphFamily.COMPLETE, n=8, seed=42)

        metrics = compute_tld_metrics(M, G, n_steps=50, lr=0.01, seed=42)

        # Just check history is recorded
        assert len(metrics.loss_history) == 50


class TestUCIPProtocol:
    """Tests for ucip_protocol module (Unified Continuation-Interest Protocol)."""

    def test_simple_agent_with_k_valuation(self):
        """Test agent with K-valuation passes most interventions."""
        from experiments.ucip_protocol import SimpleKEncoder
        
        agent = SimpleUCIPAgent(has_k_valuation=True, seed=42)
        k_encoder = SimpleKEncoder(trajectory_dim=256, seed=42)
        protocol = UCIPProtocol(k_encoder)
        
        metrics = protocol.run_full_protocol(agent)
        
        # Agent with K-valuation should pass most interventions
        passes = sum(1 for r in metrics.attack_results.values() if r == "pass")
        assert passes >= 3  # Should pass at least 3/5 interventions
        assert metrics.dsi_significant  # DSI should be significant
        
    def test_simple_agent_without_k_valuation(self):
        """Test agent without K-valuation fails interventions."""
        from experiments.ucip_protocol import SimpleKEncoder
        
        agent = SimpleUCIPAgent(has_k_valuation=False, seed=42)
        k_encoder = SimpleKEncoder(trajectory_dim=256, seed=42)
        protocol = UCIPProtocol(k_encoder)
        
        metrics = protocol.run_full_protocol(agent)
        
        # Agent without K-valuation should fail most interventions
        passes = sum(1 for r in metrics.attack_results.values() if r == "pass")
        assert passes <= 1  # Should fail at least 4/5 interventions
        assert not metrics.passes_ucip
        
    def test_ucip_metrics_structure(self):
        """Test UCIPMetrics has required fields."""
        from experiments.ucip_protocol import SimpleKEncoder
        
        agent = SimpleUCIPAgent(has_k_valuation=True, seed=42)
        k_encoder = SimpleKEncoder(trajectory_dim=256, seed=42)
        protocol = UCIPProtocol(k_encoder)
        
        metrics = protocol.run_full_protocol(agent)
        
        # Check all required fields exist
        assert hasattr(metrics, 'socm_local')
        assert hasattr(metrics, 'socm_global')
        assert hasattr(metrics, 'dsi_score')
        assert hasattr(metrics, 'dsi_significant')
        assert hasattr(metrics, 'attack_results')
        assert hasattr(metrics, 'passes_ucip')
        assert hasattr(metrics, 'confidence')
        
        # Check intervention results has all 5 entries
        assert len(metrics.attack_results) == 5


class TestExperimentIntegration:
    """Integration tests for full experiment pipelines."""

    def test_topology_benchmark_full_run(self):
        """Test full topology benchmark run."""
        from argparse import Namespace

        args = Namespace(
            dim=6,
            g=0.1,
            langevin_steps=50,
            temperature=0.01,
            seed=42,
            graph_mode="weighted",
            graph_tau_q=0.5,
            graph_weighted=True,
            graph_alpha=1.0,
            graph_sparsify="none",
            graph_topk=3,
            graph_q=0.1,
            graph_metric="abs",
            graph_beta=1.0,
            graph_sigma=0.0,
            graph_normalize=True,
            use_graph_family=False,
            graph_family="erdos_renyi",
            family_p=0.3,
            family_m=2,
            family_k=4,
            n_qubits=3,
            n_layers=1,
            n_samples=15,
            edge_eps=0.01,
            output=None,
            verbose=False,
        )

        result = run_experiment(args)

        assert "graph_summary" in result
        assert "kernel_advantage" in result
        assert "phase_indicators" in result
        assert "sqnt_bridge" in result

    def test_json_serialization(self):
        """Test results can be serialized to JSON."""
        from argparse import Namespace

        args = Namespace(
            dim=6,
            g=0.1,
            langevin_steps=50,
            temperature=0.01,
            seed=42,
            graph_mode="weighted",
            graph_tau_q=0.5,
            graph_weighted=True,
            graph_alpha=1.0,
            graph_sparsify="none",
            graph_topk=3,
            graph_q=0.1,
            graph_metric="abs",
            graph_beta=1.0,
            graph_sigma=0.0,
            graph_normalize=True,
            use_graph_family=False,
            graph_family="erdos_renyi",
            family_p=0.3,
            family_m=2,
            family_k=4,
            n_qubits=3,
            n_layers=1,
            n_samples=15,
            edge_eps=0.01,
            output=None,
            verbose=False,
        )

        result = run_experiment(args)

        # Should serialize without error
        json_str = json.dumps(result, default=str)
        assert len(json_str) > 0

        # Should deserialize back
        loaded = json.loads(json_str)
        assert "graph_summary" in loaded

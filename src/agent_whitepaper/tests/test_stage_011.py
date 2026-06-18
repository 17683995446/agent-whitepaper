"""Test Stage 11: Hierarchical Generative Model."""

import sys
sys.path.insert(0, "/workspace/agent-whitepaper/src")

import numpy as np
import pytest

from agent_whitepaper.core.stage_011 import (
    LayerType,
    HierarchicalGenerativeModel,
    HierarchicalModelBuilder,
)


class TestLayerType:
    def test_enum_values(self):
        assert LayerType.OBSERVATION.value == "observation"
        assert LayerType.FEATURE.value == "feature"
        assert LayerType.SEMANTIC.value == "semantic"
        assert LayerType.PRIOR.value == "prior"


class TestHierarchicalGenerativeModel:
    """Test the HierarchicalGenerativeModel data class."""

    def _create_test_model(self, num_layers=2, layer_dims=(4, 2), obs_dim=8):
        """Create a test hierarchical generative model."""
        def weight_init_fn(in_dim, out_dim):
            return np.random.randn(in_dim, out_dim) * 0.1

        def precision_init_fn(n):
            return np.eye(n) * 2.0

        builder = HierarchicalModelBuilder()
        return builder.build_model(
            num_layers=num_layers,
            layer_dims=layer_dims,
            obs_dim=obs_dim,
            weight_init_fn=weight_init_fn,
            precision_init_fn=precision_init_fn,
        )

    def test_build_model_basic(self):
        model = self._create_test_model()
        assert model.num_layers == 2
        assert model.layer_dims == (4, 2)
        assert model.obs_dim == 8
        assert len(model.weights) == 2
        assert len(model.precisions) == 2

    def test_post_init_validation(self):
        builder = HierarchicalModelBuilder()

        with pytest.raises(ValueError, match="num_layers"):
            builder.build_model(
                num_layers=0,
                layer_dims=(4,),
                obs_dim=8,
                weight_init_fn=lambda i, o: np.zeros((i, o)),
                precision_init_fn=lambda n: np.eye(n),
            )

    def test_covariances_property(self):
        model = self._create_test_model()
        covs = model.covariances
        assert len(covs) == 2
        for cov, prec in zip(covs, model.precisions):
            np.testing.assert_allclose(cov @ prec, np.eye(cov.shape[0]), atol=1e-10)

    def test_obs_covariance_property(self):
        model = self._create_test_model()
        obs_cov = model.obs_covariance
        np.testing.assert_allclose(
            obs_cov @ model.obs_precision, np.eye(obs_cov.shape[0]), atol=1e-10
        )

    def test_top_down_predict(self):
        model = self._create_test_model(num_layers=2, layer_dims=(4, 2), obs_dim=8)
        parent_repr = np.array([1.0, 2.0])
        pred = model.top_down_predict(parent_repr, layer_index=2)
        assert pred.shape == (4,)

    def test_compute_abstraction_ordering(self):
        model = self._create_test_model(num_layers=3, layer_dims=(8, 4, 2), obs_dim=16)
        ordering = model.compute_abstraction_ordering()
        assert ordering == [1, 2, 3]

    def test_conditional_log_likelihood(self):
        model = self._create_test_model(num_layers=2, layer_dims=(4, 2), obs_dim=8)
        # layer_index=2: 生成 r^1 from r^2
        # child = r^1 (dim 4), parent = r^2 (dim 2)
        child = np.array([1.0, 2.0, 3.0, 4.0])
        parent = np.array([1.0, 2.0])
        ll = model.conditional_log_likelihood(child, parent, layer_index=2)
        assert isinstance(ll, float)

    def test_joint_log_likelihood(self):
        model = self._create_test_model(num_layers=2, layer_dims=(4, 2), obs_dim=8)
        observation = np.random.randn(8)
        representations = (np.random.randn(4), np.random.randn(2))
        ll = model.joint_log_likelihood(observation, representations)
        assert isinstance(ll, float)


class TestHierarchicalModelBuilder:
    def test_build_model_returns_correct_type(self):
        from agent_whitepaper.core.stage_011 import HierarchicalModelBuilder
        builder = HierarchicalModelBuilder()
        model = builder.build_model(
            num_layers=1,
            layer_dims=(4,),
            obs_dim=8,
            weight_init_fn=lambda i, o: np.zeros((i, o)),
            precision_init_fn=lambda n: np.eye(n),
        )
        assert isinstance(model, HierarchicalGenerativeModel)

    def test_compute_free_energy_from_model(self):
        builder = HierarchicalModelBuilder()
        model = builder.build_model(
            num_layers=2,
            layer_dims=(4, 2),
            obs_dim=8,
            weight_init_fn=lambda i, o: np.random.randn(i, o) * 0.1,
            precision_init_fn=lambda n: np.eye(n) * 2.0,
        )
        observation = np.random.randn(8)
        representations = (np.random.randn(4), np.random.randn(2))
        fe = builder.compute_free_energy_from_model(model, observation, representations)
        assert isinstance(fe, float)
        assert fe > 0  # 自由能应该为正

    def test_optimal_depth(self):
        builder = HierarchicalModelBuilder()
        model = builder.build_model(
            num_layers=3,
            layer_dims=(4, 3, 2),
            obs_dim=8,
            weight_init_fn=lambda i, o: np.random.randn(i, o) * 0.1,
            precision_init_fn=lambda n: np.eye(n),
        )
        depth = builder.optimal_depth(model, free_energy_budget=10.0, max_depth=5)
        assert depth == 3  # min(3, 5)

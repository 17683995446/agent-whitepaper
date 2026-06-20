"""Test the core predictive coding pipeline (Stages 11-15)."""

import sys
sys.path.insert(0, "/workspace/agent-whitepaper/src")

import numpy as np
import pytest

from agent_whitepaper.core.stage_011 import HierarchicalGenerativeModel, HierarchicalModelBuilder
from agent_whitepaper.core.stage_012 import BottomUpInferenceEngine
from agent_whitepaper.core.stage_013 import TopDownInferenceEngine
from agent_whitepaper.core.stage_014 import PerceptionUpdateEngine
from agent_whitepaper.core.stage_015 import WeightUpdateEngine


@pytest.fixture
def model():
    """构建测试模型。"""
    builder = HierarchicalModelBuilder()
    return builder.build_model(
        num_layers=2,
        layer_dims=(4, 2),
        obs_dim=8,
        weight_init_fn=lambda in_dim, out_dim: np.random.randn(in_dim, out_dim) * 0.1,
        precision_init_fn=lambda n: np.eye(n) * 2.0,
    )


@pytest.fixture
def observation():
    np.random.seed(42)
    return np.random.randn(8)


@pytest.fixture
def representations():
    np.random.seed(42)
    return (np.random.randn(4), np.random.randn(2))


def test_stage_11_build_model(model):
    """Test Stage 11: Build a hierarchical generative model."""
    assert model.num_layers == 2
    assert model.layer_dims == (4, 2)
    assert model.obs_dim == 8
    assert len(model.weights) == 2
    assert len(model.precisions) == 2


def test_stage_12_bottom_up(model, observation, representations):
    """Test Stage 12: Bottom-up error propagation."""
    engine = BottomUpInferenceEngine()
    propagation = engine.propagate(model, observation, representations)

    assert len(propagation.error_signals) == model.num_layers + 1
    assert len(propagation.propagated_signals) == model.num_layers
    assert len(propagation.precision_weighted_errors) == model.num_layers

    stability = engine.check_stability(propagation)
    assert "is_stable" in stability


def test_stage_13_top_down(model, observation, representations):
    """Test Stage 13: Top-down prediction propagation."""
    bu_engine = BottomUpInferenceEngine()
    bottom_up = bu_engine.propagate(model, observation, representations)

    engine = TopDownInferenceEngine()
    top_repr = representations[-1]
    top_down = engine.predict(model, bottom_up, top_repr)

    assert len(top_down.predictions) == model.num_layers - 1
    assert top_down.obs_prediction.shape == (model.obs_dim,)


def test_stage_14_perception_update(model, observation, representations):
    """Test Stage 14: Perception update via gradient descent."""
    bu_engine = BottomUpInferenceEngine()
    bottom_up = bu_engine.propagate(model, observation, representations)
    td_engine = TopDownInferenceEngine()
    top_down = td_engine.predict(model, bottom_up, representations[-1])

    engine = PerceptionUpdateEngine()
    update = engine.update(
        top_down,
        representations,
        time_constant=1.0,
        step_size=0.1,
        max_steps=50,
        convergence_threshold=1e-3,
    )

    assert len(update.representations_after) == len(update.representations_before)
    assert len(update.gradients) > 0


def test_stage_15_weight_update(model, observation, representations):
    """Test Stage 15: Weight update via Hebbian learning."""
    bu_engine = BottomUpInferenceEngine()
    bottom_up = bu_engine.propagate(model, observation, representations)
    td_engine = TopDownInferenceEngine()
    top_down = td_engine.predict(model, bottom_up, representations[-1])
    pu_engine = PerceptionUpdateEngine()
    perception_update = pu_engine.update(
        top_down, representations, max_steps=10, step_size=0.01
    )

    engine = WeightUpdateEngine()
    weights = model.weights
    fisher_diagonals = tuple(np.ones_like(w) for w in weights)
    anchor_weights = tuple(w.copy() for w in weights)

    weight_update = engine.update(
        perception_update,
        weights,
        fisher_diagonals,
        anchor_weights,
        time_constant=100.0,
        learning_rate=0.01,
        ewc_strength=0.1,
    )

    assert len(weight_update.weights_after) == len(weights)
    assert len(weight_update.weight_gradients) == len(weights)


def test_full_pipeline(model, observation, representations):
    """测试完整管道闭环。"""
    # Stage 12
    bu_engine = BottomUpInferenceEngine()
    bottom_up = bu_engine.propagate(model, observation, representations)

    # Stage 13
    td_engine = TopDownInferenceEngine()
    top_down = td_engine.predict(model, bottom_up, representations[-1])

    # Stage 14
    pu_engine = PerceptionUpdateEngine()
    perception = pu_engine.update(top_down, representations, max_steps=10, step_size=0.01)

    # Stage 15
    wu_engine = WeightUpdateEngine()
    weight = wu_engine.update(
        perception,
        model.weights,
        tuple(np.ones_like(w) for w in model.weights),
        tuple(w.copy() for w in model.weights),
        time_constant=100.0,
        learning_rate=0.01,
        ewc_strength=0.1,
    )

    # 验证整个管道的数据一致性
    assert len(weight.weights_after) == len(model.weights)

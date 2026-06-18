"""Test the core predictive coding pipeline (Stages 11-15)."""

import sys
sys.path.insert(0, "/workspace/agent-whitepaper/src")

import numpy as np

from agent_whitepaper.core.stage_011 import HierarchicalGenerativeModel, HierarchicalModelBuilder
from agent_whitepaper.core.stage_012 import BottomUpInferenceEngine
from agent_whitepaper.core.stage_013 import TopDownInferenceEngine
from agent_whitepaper.core.stage_014 import PerceptionUpdateEngine
from agent_whitepaper.core.stage_015 import WeightUpdateEngine


def test_stage_11_build_model():
    """Test Stage 11: Build a hierarchical generative model."""
    print("\n=== Stage 11: Hierarchical Generative Model ===")

    builder = HierarchicalModelBuilder()
    model = builder.build_model(
        num_layers=2,
        layer_dims=(4, 2),
        obs_dim=8,
        weight_init_fn=lambda n_out, n_in: np.random.randn(n_out, n_in) * 0.1,
        precision_init_fn=lambda n: np.eye(n) * 2.0,
    )

    print(f"  num_layers: {model.num_layers}")
    print(f"  layer_dims: {model.layer_dims}")
    print(f"  obs_dim: {model.obs_dim}")
    print(f"  weights shapes: {[w.shape for w in model.weights]}")
    print(f"  precisions shapes: {[p.shape for p in model.precisions]}")

    assert model.num_layers == 2
    assert model.layer_dims == (4, 2)
    assert model.obs_dim == 8
    assert len(model.weights) == 2
    assert len(model.precisions) == 2
    print("  ✓ Stage 11 passed")
    return model


def test_stage_12_bottom_up(model):
    """Test Stage 12: Bottom-up error propagation."""
    print("\n=== Stage 12: Bottom-Up Error Propagation ===")

    engine = BottomUpInferenceEngine()
    observation = np.random.randn(8)
    representations = (np.random.randn(4), np.random.randn(2))

    propagation = engine.propagate(model, observation, representations)

    print(f"  error_signals: {len(propagation.error_signals)}")
    print(f"  propagated_signals: {len(propagation.propagated_signals)}")
    print(f"  jacobians: {len(propagation.jacobians)}")
    print(f"  total_complexity: {propagation.total_complexity}")

    assert len(propagation.error_signals) == model.num_layers + 1  # e^0, e^1, ..., e^L
    assert len(propagation.propagated_signals) == model.num_layers
    assert len(propagation.precision_weighted_errors) == model.num_layers

    stability = engine.check_stability(propagation)
    print(f"  stability: {stability['is_stable']}")
    print("  ✓ Stage 12 passed")
    return propagation


def test_stage_13_top_down(model, bottom_up):
    """Test Stage 13: Top-down prediction propagation."""
    print("\n=== Stage 13: Top-Down Prediction Propagation ===")

    engine = TopDownInferenceEngine()
    top_repr = np.random.randn(2)

    top_down = engine.predict(model, bottom_up, top_repr)

    print(f"  predictions: {len(top_down.predictions)}")
    print(f"  obs_prediction shape: {top_down.obs_prediction.shape}")
    print(f"  updated_representations: {len(top_down.updated_representations)}")

    assert len(top_down.predictions) == model.num_layers - 1
    assert top_down.obs_prediction.shape == (model.obs_dim,)
    print("  ✓ Stage 13 passed")
    return top_down


def test_stage_14_perception_update(model, top_down):
    """Test Stage 14: Perception update via gradient descent."""
    print("\n=== Stage 14: Perception Update ===")

    engine = PerceptionUpdateEngine()
    # 使用随机初始表征（与模型层维度匹配）
    initial_reprs = tuple(np.random.randn(dim) for dim in model.layer_dims)

    update = engine.update(
        top_down,
        initial_reprs,
        time_constant=1.0,
        step_size=0.1,
        max_steps=50,
        convergence_threshold=1e-3,
    )

    print(f"  is_converged: {update.is_converged}")
    print(f"  num_steps: {update.num_steps}")
    print(f"  gradients: {len(update.gradients)}")

    assert len(update.representations_after) == len(update.representations_before)
    assert len(update.gradients) > 0
    print("  ✓ Stage 14 passed")
    return update


def test_stage_15_weight_update(perception_update, model):
    """Test Stage 15: Weight update via Hebbian learning."""
    print("\n=== Stage 15: Weight Update ===")

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

    print(f"  weights_before: {len(weight_update.weights_before)}")
    print(f"  weights_after: {len(weight_update.weights_after)}")
    print(f"  weight_gradients: {len(weight_update.weight_gradients)}")
    print(f"  learning_rates: {weight_update.learning_rates_per_layer}")

    assert len(weight_update.weights_after) == len(weights)
    assert len(weight_update.weight_gradients) == len(weights)
    print("  ✓ Stage 15 passed")
    return weight_update


def run_full_pipeline():
    """Run the complete Stages 11-15 pipeline."""
    print("=" * 60)
    print("Running Core Predictive Coding Pipeline (Stages 11-15)")
    print("=" * 60)

    # Stage 11: Build model
    model = test_stage_11_build_model()

    # Stage 12: Bottom-up propagation
    bottom_up = test_stage_12_bottom_up(model)

    # Stage 13: Top-down propagation
    top_down = test_stage_13_top_down(model, bottom_up)

    # Stage 14: Perception update
    perception = test_stage_14_perception_update(model, top_down)

    # Stage 15: Weight update
    weight = test_stage_15_weight_update(perception, model)

    print("\n" + "=" * 60)
    print("✓ All stages (11-15) passed successfully!")
    print("=" * 60)

    return {
        "model": model,
        "bottom_up": bottom_up,
        "top_down": top_down,
        "perception": perception,
        "weight": weight,
    }


if __name__ == "__main__":
    results = run_full_pipeline()

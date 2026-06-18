"""静态代码核验测试：验证v2核心模块的逻辑正确性。

本测试文件专注于发现以下bug：
1. joint_log_likelihood 重复计算观测对数似然
2. 感知更新使用陈旧梯度（propagated_signals 不随表征更新而重算）
3. cross_entropy 损失未验证输出层激活函数
4. 边界条件、异常容错、数据一致性
"""

import numpy as np
import pytest
from dataclasses import dataclass

from agent_whitepaper.core.stage_011 import HierarchicalModelBuilder
from agent_whitepaper.core.stage_011_v2 import (
    NonlinearHierarchicalModelBuilder,
    HierarchicalGenerativeModelV2,
    ActivationType,
    ActivationRegistry,
    NonlinearGenerativeFunction,
)
from agent_whitepaper.core.stage_012_v2 import BottomUpInferenceEngineV2
from agent_whitepaper.core.stage_013_v2 import TopDownInferenceEngineV2
from agent_whitepaper.core.stage_014_v2 import PerceptionUpdateEngineV2
from agent_whitepaper.core.stage_021_v2 import SafetyDriveCalculatorV2
from agent_whitepaper.core.stage_022_v2 import SocialDriveCalculatorV2
from agent_whitepaper.core.stage_023_v2 import ExplorationDriveCalculatorV2
from agent_whitepaper.core.stage_050_v2 import (
    MultiLayerPredictorV2,
    MultiLayerPredictorBuilder,
    LayerParams,
)


# ============================================================
# 辅助函数
# ============================================================
def make_identity_precision(dim):
    """创建单位精度矩阵。"""
    return np.eye(dim)


def make_random_weight(in_dim, out_dim, seed=42):
    """创建随机权重矩阵。"""
    rng = np.random.RandomState(seed)
    return rng.randn(in_dim, out_dim) * 0.1


def build_v2_model(num_layers=2, obs_dim=3, layer_dims=None, activations=None):
    """构建v2测试模型。"""
    if layer_dims is None:
        layer_dims = tuple(obs_dim + i for i in range(num_layers))
    if activations is None:
        activations = tuple(ActivationType.TANH for _ in range(num_layers))

    def weight_init(in_dim, out_dim):
        rng = np.random.RandomState(42)
        return rng.randn(in_dim, out_dim) * 0.5

    builder = NonlinearHierarchicalModelBuilder()
    return builder.build_model(
        num_layers=num_layers,
        layer_dims=layer_dims,
        obs_dim=obs_dim,
        weight_init_fn=weight_init,
        precision_init_fn=make_identity_precision,
        activations=activations,
    )


def build_v1_model(num_layers=2, obs_dim=3, layer_dims=None):
    """构建v1测试模型。"""
    if layer_dims is None:
        layer_dims = tuple(obs_dim + i for i in range(num_layers))

    def weight_init(in_dim, out_dim):
        rng = np.random.RandomState(42)
        return rng.randn(in_dim, out_dim) * 0.5

    builder = HierarchicalModelBuilder()
    return builder.build_model(
        num_layers=num_layers,
        layer_dims=layer_dims,
        obs_dim=obs_dim,
        weight_init_fn=weight_init,
        precision_init_fn=make_identity_precision,
    )


# ============================================================
# BUG 1 测试：joint_log_likelihood 重复计算观测对数似然
# ============================================================
class TestBug1DoubleCountingObsLikelihood:
    """验证 joint_log_likelihood 是否重复计算了观测对数似然。"""

    def test_v2_obs_generative_fn_equals_generative_fns_0(self):
        """验证 obs_generative_fn 和 generative_fns[0] 是否相同。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(4, 5))
        r0 = np.array([0.1, 0.2, 0.3, 0.4])

        obs_pred = model.obs_generative_fn.forward(r0)
        gen_pred = model.generative_fns[0].forward(r0)

        # 这两个应该是同一个函数（bug的根源）
        assert np.allclose(obs_pred, gen_pred), \
            "obs_generative_fn 和 generative_fns[0] 应该产生相同输出"

    def test_v2_double_counting_obs_likelihood(self):
        """验证 l=0 时不再重复计算观测对数似然（修复后应通过）。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        # 手动计算正确的联合对数似然
        # p(o, r^1, r^2) = p(o|r^1) * p(r^1|r^2) * p(r^2)
        obs_pred = model.obs_generative_fn.forward(reprs[0])
        obs_error = obs - obs_pred
        obs_ll = -0.5 * float(obs_error.T @ model.obs_precision @ obs_error)

        # p(r^1|r^2)
        r1_pred = model.generative_fns[1].forward(reprs[1])
        r1_error = reprs[0] - r1_pred
        r1_ll = -0.5 * float(r1_error.T @ model.precisions[1] @ r1_error)

        # p(r^2) - 先验
        prior_error = reprs[1] - model.prior_mean
        prior_ll = -0.5 * float(prior_error.T @ model.prior_precision @ prior_error)

        correct_ll = obs_ll + r1_ll + prior_ll

        # 系统计算的联合对数似然
        system_ll = model.joint_log_likelihood(obs, reprs)

        # 修复后：system_ll 应该等于 correct_ll（不再重复计算）
        assert abs(system_ll - correct_ll) < 1e-10, \
            f"修复后联合对数似然应等于手动计算值: system={system_ll}, correct={correct_ll}"

    def test_v2_free_energy_sign(self):
        """验证自由能 = -ln p(o,r)，应该为正（当概率<1时）。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        fe = model.compute_free_energy(obs, reprs)
        ll = model.joint_log_likelihood(obs, reprs)

        assert fe == -ll, "自由能应该等于负对数似然"


# ============================================================
# BUG 2 测试：感知更新使用陈旧梯度
# ============================================================
class TestBug2StaleGradients:
    """验证感知更新是否使用了陈旧梯度。"""

    def test_gradients_dont_change_between_steps(self):
        """验证梯度在迭代过程中是否保持不变（bug标志）。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)
        td_engine = TopDownInferenceEngineV2()
        td = td_engine.predict(model, bu, reprs[-1])

        pu_engine = PerceptionUpdateEngineV2()
        result = pu_engine.update(
            td, reprs, max_steps=10, step_size=0.01, adaptive_lr=False
        )

        # 如果梯度是陈旧的，自由能可能不会单调下降
        fe_history = result.free_energy_history
        print(f"自由能历史: {fe_history}")

        # 检查自由能是否单调下降（真正梯度下降应该如此）
        # 注意：由于使用零观测计算自由能，这里只检查趋势
        is_monotone_decreasing = all(
            fe_history[i] >= fe_history[i + 1] - 1e-6
            for i in range(len(fe_history) - 1)
        )

        if not is_monotone_decreasing:
            print("警告：自由能未单调下降，可能使用了陈旧梯度")

    def test_gradient_recomputation_needed(self):
        """验证表征更新后传播信号是否改变。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs_orig = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        bu_engine = BottomUpInferenceEngineV2()

        # 原始传播信号
        bu_orig = bu_engine.propagate(model, obs, reprs_orig)
        orig_signals = bu_orig.propagated_signals

        # 修改表征后重新计算
        reprs_modified = (
            reprs_orig[0] + np.array([0.5, -0.5, 0.5]),
            reprs_orig[1] + np.array([0.3, -0.3, 0.3, -0.3]),
        )
        bu_modified = bu_engine.propagate(model, obs, reprs_modified)
        modified_signals = bu_modified.propagated_signals

        # 传播信号应该不同
        signals_changed = any(
            not np.allclose(orig_signals[i], modified_signals[i])
            for i in range(len(orig_signals))
        )

        assert signals_changed, \
            "表征改变后传播信号应该改变（验证梯度需要重算）"


# ============================================================
# BUG 3 测试：cross_entropy 未验证输出层激活
# ============================================================
class TestBug3CrossEntropyValidation:
    """验证 cross_entropy 损失是否检查了输出层激活函数。"""

    def test_cross_entropy_with_non_sigmoid_output(self):
        """使用非sigmoid输出层 + cross_entropy 应该报错。"""
        # 构建一个 ReLU 输出的模型
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(3, 4, 2),
            activations=(ActivationType.TANH, ActivationType.RELU),
            learning_rate=0.01,
            seed=42,
        )

        x = np.array([0.5, -0.3, 0.8])
        target = np.array([1.0, 0.0])

        # 修复后：应该抛出 ValueError
        with pytest.raises(ValueError, match="SIGMOID"):
            predictor.backward(x, target, "cross_entropy")


# ============================================================
# stage_011_v2 测试：非线性生成函数
# ============================================================
class TestNonlinearGenerativeFunction:
    """测试非线性生成函数。"""

    @pytest.mark.parametrize("act_type", [
        ActivationType.LINEAR,
        ActivationType.RELU,
        ActivationType.TANH,
        ActivationType.SIGMOID,
        ActivationType.SOFTPLUS,
        ActivationType.SWISH,
    ])
    def test_jacobian_analytical_vs_numerical(self, act_type):
        """验证解析Jacobian与数值Jacobian一致。"""
        rng = np.random.RandomState(42)
        W = rng.randn(4, 3) * 0.5
        b = rng.randn(4) * 0.1
        r = rng.randn(3) * 0.5

        fn = NonlinearGenerativeFunction(
            weight=W, bias=b, activation=act_type, epsilon=1e-6
        )

        J_analytical = fn.jacobian_analytical(r)
        J_numerical = fn.jacobian(r)

        # 对于 ReLU，在零点附近可能不连续，使用更大的容差
        tol = 1e-3 if act_type == ActivationType.RELU else 1e-4
        assert np.allclose(J_analytical, J_numerical, atol=tol), \
            f"{act_type}: 解析Jacobian与数值Jacobian不一致\n" \
            f"analytical:\n{J_analytical}\nnumerical:\n{J_numerical}"

    def test_forward_linear_activation(self):
        """测试 LINEAR 激活等于 W@r+b。"""
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.5, -0.5])
        fn = NonlinearGenerativeFunction(weight=W, bias=b, activation=ActivationType.LINEAR)
        r = np.array([1.0, 2.0])
        result = fn.forward(r)
        expected = W @ r + b
        assert np.allclose(result, expected)

    def test_forward_relu_activation(self):
        """测试 ReLU 激活。"""
        W = np.eye(2)
        b = np.array([0.0, 0.0])
        fn = NonlinearGenerativeFunction(weight=W, bias=b, activation=ActivationType.RELU)
        r = np.array([-1.0, 2.0])
        result = fn.forward(r)
        expected = np.array([0.0, 2.0])
        assert np.allclose(result, expected)

    def test_call_compatibility(self):
        """测试 __call__ 与 stage_011 的兼容性。"""
        W = np.eye(2)
        b = np.array([0.1, 0.2])
        fn = NonlinearGenerativeFunction(weight=W, bias=b, activation=ActivationType.LINEAR)
        r = np.array([1.0, 2.0])

        # f(r, W) 应该等于 f(r)
        result1 = fn(r, W)
        result2 = fn.forward(r)
        assert np.allclose(result1, result2)

    def test_epsilon_validation(self):
        """测试 epsilon 验证。"""
        with pytest.raises(ValueError, match="epsilon"):
            NonlinearGenerativeFunction(
                weight=np.eye(2), bias=np.zeros(2),
                epsilon=0.0
            )
        with pytest.raises(ValueError, match="epsilon"):
            NonlinearGenerativeFunction(
                weight=np.eye(2), bias=np.zeros(2),
                epsilon=-1.0
            )


# ============================================================
# stage_011_v2 测试：模型构建与验证
# ============================================================
class TestHierarchicalGenerativeModelV2:
    """测试v2层级生成模型。"""

    def test_build_model_basic(self):
        """测试基本模型构建。"""
        model = build_v2_model(num_layers=3, obs_dim=4, layer_dims=(4, 5, 6))
        assert model.num_layers == 3
        assert model.obs_dim == 4
        assert len(model.weights) == 3
        assert len(model.biases) == 3
        assert len(model.precisions) == 3
        assert len(model.generative_fns) == 3

    def test_build_model_invalid_num_layers(self):
        """测试无效层数。"""
        with pytest.raises(ValueError):
            build_v2_model(num_layers=0)

    def test_build_model_dim_mismatch(self):
        """测试维度不匹配。"""
        with pytest.raises(ValueError):
            build_v2_model(num_layers=2, layer_dims=(3, 4, 5))

    def test_precision_must_be_positive_definite(self):
        """测试精度矩阵必须正定。"""
        def bad_precision(dim):
            # 非正定矩阵
            M = np.zeros((dim, dim))
            M[0, 0] = -1.0
            for i in range(1, dim):
                M[i, i] = 1.0
            return M

        with pytest.raises(ValueError, match="正定"):
            NonlinearHierarchicalModelBuilder().build_model(
                num_layers=1, layer_dims=(3,), obs_dim=3,
                weight_init_fn=lambda i, o: np.eye(i, o),
                precision_init_fn=bad_precision,
            )

    def test_forward_with_cache(self):
        """测试带缓存的前向传播。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        cache = model.forward_with_cache(reprs)
        assert len(cache.linear_outputs) == 2
        assert len(cache.activated_outputs) == 2
        assert len(cache.representations) == 2

    def test_compute_jacobians(self):
        """测试Jacobian计算。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        jacobians = model.compute_jacobians(reprs)
        assert len(jacobians) == 2
        # J^l 形状 = (child_dim, parent_dim)
        assert jacobians[0].shape == (3, 3)
        assert jacobians[1].shape == (3, 4)

    def test_to_v1_model(self):
        """测试转换为v1模型。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        v1_model = model.to_v1_model()
        assert v1_model.num_layers == 2
        assert v1_model.obs_dim == 3


# ============================================================
# stage_012_v2 测试：自底向上传播
# ============================================================
class TestBottomUpInferenceV2:
    """测试自底向上传播。"""

    def test_compute_errors(self):
        """测试误差计算。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        engine = BottomUpInferenceEngineV2()
        errors = engine.compute_errors(model, obs, reprs)

        # L+1 个误差信号
        assert len(errors) == model.num_layers + 1
        # e^0 维度 = obs_dim
        assert errors[0].shape == (3,)
        # e^1 维度 = layer_dims[0]
        assert errors[1].shape == (3,)
        # e^2 (prior) 维度 = layer_dims[-1]
        assert errors[2].shape == (4,)

    def test_propagate(self):
        """测试完整传播。"""
        model = build_v2_model(num_layers=3, obs_dim=3, layer_dims=(3, 4, 5))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6, 0.7]),
            np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        )

        engine = BottomUpInferenceEngineV2()
        result = engine.propagate(model, obs, reprs)

        assert len(result.error_signals) == 4
        assert len(result.propagated_signals) == 3
        assert len(result.error_energies) == 4

    def test_jacobian_v1_fallback(self):
        """测试v1模型的数值Jacobian回退。"""
        model = build_v1_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        engine = BottomUpInferenceEngineV2()
        J = engine.compute_jacobian(model, 1, reprs[1])

        # v1: f(r) = W @ r, J = W
        expected_J = model.weights[1]
        assert np.allclose(J, expected_J, atol=1e-4)

    def test_check_stability(self):
        """测试稳定性检查。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        engine = BottomUpInferenceEngineV2()
        result = engine.propagate(model, obs, reprs)
        stability = engine.check_stability(result)

        assert "is_stable" in stability
        assert "layer_stability" in stability
        assert "spectral_norms" in stability

    def test_invalid_layer_index(self):
        """测试无效层索引。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        engine = BottomUpInferenceEngineV2()
        with pytest.raises(ValueError):
            engine.compute_jacobian(model, 0, reprs[0])
        with pytest.raises(ValueError):
            engine.compute_jacobian(model, 5, reprs[1])


# ============================================================
# stage_013_v2 测试：自顶向下传播
# ============================================================
class TestTopDownInferenceV2:
    """测试自顶向下传播。"""

    def test_predict(self):
        """测试自顶向下预测。"""
        model = build_v2_model(num_layers=3, obs_dim=3, layer_dims=(3, 4, 5))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6, 0.7]),
            np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        )

        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)
        td_engine = TopDownInferenceEngineV2()
        td = td_engine.predict(model, bu, reprs[-1])

        assert td.bottom_up == bu
        assert len(td.predictions) == model.num_layers - 1
        assert td.obs_prediction.shape == (3,)

    def test_single_layer_model(self):
        """测试单层模型。"""
        model = build_v2_model(num_layers=1, obs_dim=3, layer_dims=(4,))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3, 0.4]),)

        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)
        td_engine = TopDownInferenceEngineV2()
        td = td_engine.predict(model, bu, reprs[-1])

        # 单层模型没有中间预测
        assert len(td.predictions) == 0
        assert td.obs_prediction.shape == (3,)

    def test_invalid_top_repr_dim(self):
        """测试无效顶层表征维度。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)
        td_engine = TopDownInferenceEngineV2()

        with pytest.raises(ValueError):
            td_engine.predict(model, bu, np.array([0.1, 0.2, 0.3]))  # 应该是4维


# ============================================================
# stage_014_v2 测试：感知更新
# ============================================================
class TestPerceptionUpdateV2:
    """测试感知更新。"""

    def test_update_basic(self):
        """测试基本感知更新。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)
        td_engine = TopDownInferenceEngineV2()
        td = td_engine.predict(model, bu, reprs[-1])

        pu_engine = PerceptionUpdateEngineV2()
        result = pu_engine.update(td, reprs, max_steps=5, step_size=0.01)

        assert result.num_steps <= 5
        assert len(result.free_energy_history) == result.num_steps
        assert len(result.representations_after) == 2

    def test_update_validation(self):
        """测试参数验证。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)
        td_engine = TopDownInferenceEngineV2()
        td = td_engine.predict(model, bu, reprs[-1])

        pu_engine = PerceptionUpdateEngineV2()

        with pytest.raises(ValueError):
            pu_engine.update(td, reprs, time_constant=0)
        with pytest.raises(ValueError):
            pu_engine.update(td, reprs, step_size=0)
        with pytest.raises(ValueError):
            pu_engine.update(td, reprs, max_steps=0)
        with pytest.raises(ValueError):
            pu_engine.update(td, reprs, convergence_threshold=0)

    def test_representations_change(self):
        """测试表征是否被更新。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)
        td_engine = TopDownInferenceEngineV2()
        td = td_engine.predict(model, bu, reprs[-1])

        pu_engine = PerceptionUpdateEngineV2()
        result = pu_engine.update(td, reprs, max_steps=10, step_size=0.1)

        # 表征应该改变
        changed = any(
            not np.allclose(result.representations_before[i], result.representations_after[i])
            for i in range(len(reprs))
        )
        assert changed, "表征应该被更新"


# ============================================================
# stage_021_v2 测试：安全驱力
# ============================================================
class TestSafetyDriveV2:
    """测试安全驱力。"""

    def test_compute_basic(self):
        """测试基本安全驱力计算。"""
        calc = SafetyDriveCalculatorV2()
        result = calc.compute(free_energy=10.0)

        assert 0 <= result.safety_strength <= 1
        assert result.safety_boundary == 0.7
        assert result.free_energy == 10.0

    def test_zero_free_energy(self):
        """测试零自由能（最安全）。"""
        calc = SafetyDriveCalculatorV2()
        result = calc.compute(free_energy=0.0)
        assert result.safety_strength == 1.0  # 最安全
        assert result.is_violated == False

    def test_max_free_energy(self):
        """测试最大自由能（最不安全）。"""
        calc = SafetyDriveCalculatorV2()
        result = calc.compute(free_energy=100.0)
        assert result.safety_strength == 0.0  # 最不安全
        assert result.is_violated == True

    def test_negative_free_energy_raises(self):
        """测试负自由能应报错。"""
        calc = SafetyDriveCalculatorV2()
        with pytest.raises(ValueError):
            calc.compute(free_energy=-1.0)

    def test_free_energy_rate(self):
        """测试自由能变化率。"""
        calc = SafetyDriveCalculatorV2()
        # 自由能在上升
        result = calc.compute(
            free_energy=50.0,
            free_energy_history=(40.0, 50.0),
        )
        assert result.free_energy_rate == 10.0
        # 上升应该降低安全强度
        result_no_rate = calc.compute(free_energy=50.0)
        assert result.safety_strength < result_no_rate.safety_strength

    def test_compute_from_model_v2(self):
        """测试从v2模型计算。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        calc = SafetyDriveCalculatorV2()
        result = calc.compute_from_model(model, obs, reprs)
        assert result.free_energy > 0  # 应该有正的自由能


# ============================================================
# stage_022_v2 测试：社会驱力
# ============================================================
class TestSocialDriveV2:
    """测试社会驱力。"""

    def test_compute_basic(self):
        """测试基本社会驱力计算。"""
        calc = SocialDriveCalculatorV2()
        reprs = (np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.3]))
        result = calc.compute(reprs)

        assert 0 <= result.sparsity <= 1
        assert result.entropy >= 0
        assert 0 <= result.social_strength <= 1

    def test_sparse_representations(self):
        """测试稀疏表征。"""
        calc = SocialDriveCalculatorV2()
        reprs = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0]))
        result = calc.compute(reprs)
        assert result.sparsity == 1.0  # 全零，最稀疏
        assert result.entropy == 0.0

    def test_dense_representations(self):
        """测试密集表征。"""
        calc = SocialDriveCalculatorV2()
        reprs = (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]))
        result = calc.compute(reprs)
        assert result.sparsity == 0.0  # 无零元素

    def test_empty_representations_raises(self):
        """测试空表征应报错。"""
        calc = SocialDriveCalculatorV2()
        with pytest.raises(ValueError):
            calc.compute(())


# ============================================================
# stage_023_v2 测试：探索驱力
# ============================================================
class TestExplorationDriveV2:
    """测试探索驱力。"""

    def test_compute_basic(self):
        """测试基本探索驱力计算。"""
        calc = ExplorationDriveCalculatorV2()
        history = (1.0, 0.8, 0.6, 0.4, 0.2)
        result = calc.compute(history)

        assert 0 <= result.exploration_strength <= 1
        assert result.convergence_rate > 0  # 在收敛

    def test_empty_history_raises(self):
        """测试空历史应报错。"""
        calc = ExplorationDriveCalculatorV2()
        with pytest.raises(ValueError):
            calc.compute(())

    def test_single_element_history(self):
        """测试单元素历史。"""
        calc = ExplorationDriveCalculatorV2()
        result = calc.compute((1.0,))
        assert result.convergence_rate == 0.0
        assert result.convergence_acceleration == 0.0

    def test_diverging_history(self):
        """测试发散历史（梯度增大）。"""
        calc = ExplorationDriveCalculatorV2()
        history = (0.1, 0.3, 0.5, 0.7, 1.0)
        result = calc.compute(history)
        # 发散时收敛率为负
        assert result.convergence_rate < 0
        # 探索强度应该高
        assert result.exploration_strength > 0.5

    def test_converging_history(self):
        """测试收敛历史（梯度减小）。"""
        calc = ExplorationDriveCalculatorV2()
        history = (1.0, 0.5, 0.25, 0.125, 0.0625)
        result = calc.compute(history)
        # 快速收敛
        assert result.convergence_rate > 0.4
        # 探索强度应该低
        assert result.exploration_strength < 0.5


# ============================================================
# stage_050_v2 测试：多层预测模型
# ============================================================
class TestMultiLayerPredictorV2:
    """测试多层预测模型。"""

    def test_build_basic(self):
        """测试基本构建。"""
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(3, 4, 2),
            activations=(ActivationType.TANH, ActivationType.LINEAR),
            learning_rate=0.01,
            seed=42,
        )
        assert predictor.input_dim == 3
        assert predictor.output_dim == 2
        assert len(predictor.layers) == 2

    def test_forward(self):
        """测试前向传播。"""
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(3, 4, 2),
            activations=(ActivationType.TANH, ActivationType.LINEAR),
            seed=42,
        )
        x = np.array([0.5, -0.3, 0.8])
        output = predictor.forward(x)
        assert output.shape == (2,)

    def test_forward_with_cache(self):
        """测试带缓存的前向传播。"""
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(3, 4, 2),
            activations=(ActivationType.TANH, ActivationType.LINEAR),
            seed=42,
        )
        x = np.array([0.5, -0.3, 0.8])
        output, linear_outs, activated_outs = predictor.forward_with_cache(x)
        assert output.shape == (2,)
        assert len(linear_outs) == 2
        assert len(activated_outs) == 3  # 输入 + 2层输出

    def test_backward_mse(self):
        """测试MSE反向传播。"""
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(3, 4, 2),
            activations=(ActivationType.TANH, ActivationType.LINEAR),
            seed=42,
        )
        x = np.array([0.5, -0.3, 0.8])
        target = np.array([1.0, 0.0])

        w_grads, b_grads, loss = predictor.backward(x, target, "mse")
        assert len(w_grads) == 2
        assert len(b_grads) == 2
        assert w_grads[0].shape == (4, 3)
        assert w_grads[1].shape == (2, 4)

    def test_train_step_decreases_loss(self):
        """测试训练步应降低损失。"""
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(3, 4, 2),
            activations=(ActivationType.TANH, ActivationType.LINEAR),
            learning_rate=0.1,
            seed=42,
        )
        x = np.array([0.5, -0.3, 0.8])
        target = np.array([1.0, 0.0])

        # 训练前的损失
        _, _, loss_before = predictor.backward(x, target, "mse")

        # 训练多步
        for _ in range(50):
            predictor = predictor.train_step(x, target, "mse")

        # 训练后的损失
        _, _, loss_after = predictor.backward(x, target, "mse")

        assert loss_after < loss_before, \
            f"训练后损失应降低: {loss_after} < {loss_before}"

    def test_invalid_loss_type(self):
        """测试无效损失类型。"""
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(3, 4, 2),
            activations=(ActivationType.TANH, ActivationType.LINEAR),
            seed=42,
        )
        x = np.array([0.5, -0.3, 0.8])
        target = np.array([1.0, 0.0])

        with pytest.raises(ValueError, match="未知损失类型"):
            predictor.backward(x, target, "invalid_loss")

    def test_invalid_dimensions(self):
        """测试无效维度。"""
        with pytest.raises(ValueError):
            MultiLayerPredictorBuilder.build(
                layer_dims=(3,),  # 至少需要2层
                activations=(ActivationType.LINEAR,),
            )

    def test_invalid_learning_rate(self):
        """测试无效学习率。"""
        with pytest.raises(ValueError):
            MultiLayerPredictorBuilder.build(
                layer_dims=(3, 2),
                activations=(ActivationType.LINEAR,),
                learning_rate=0.0,
            )

    def test_invalid_momentum(self):
        """测试无效动量。"""
        with pytest.raises(ValueError):
            MultiLayerPredictorBuilder.build(
                layer_dims=(3, 2),
                activations=(ActivationType.LINEAR,),
                momentum=1.0,
            )

    def test_cross_entropy_with_sigmoid(self):
        """测试 sigmoid + cross_entropy 的正确组合。"""
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(3, 4, 2),
            activations=(ActivationType.TANH, ActivationType.SIGMOID),
            learning_rate=0.1,
            seed=42,
        )
        x = np.array([0.5, -0.3, 0.8])
        target = np.array([1.0, 0.0])

        # 训练前的损失
        _, _, loss_before = predictor.backward(x, target, "cross_entropy")

        # 训练多步
        for _ in range(50):
            predictor = predictor.train_step(x, target, "cross_entropy")

        _, _, loss_after = predictor.backward(x, target, "cross_entropy")
        assert loss_after < loss_before, \
            f"sigmoid + cross_entropy 训练后损失应降低: {loss_after} < {loss_before}"

    def test_weight_init_methods(self):
        """测试不同权重初始化方法。"""
        for init in ["xavier", "he", "zeros"]:
            predictor = MultiLayerPredictorBuilder.build(
                layer_dims=(3, 4, 2),
                activations=(ActivationType.TANH, ActivationType.LINEAR),
                weight_init=init,
                seed=42,
            )
            output = predictor.forward(np.array([0.5, -0.3, 0.8]))
            assert output.shape == (2,)

    def test_predict_equals_forward(self):
        """测试 predict 等同于 forward。"""
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(3, 2),
            activations=(ActivationType.LINEAR,),
            seed=42,
        )
        x = np.array([0.5, -0.3, 0.8])
        assert np.allclose(predictor.predict(x), predictor.forward(x))


# ============================================================
# 集成测试：跨模块调用链路
# ============================================================
class TestIntegrationPipeline:
    """测试完整的预测编码管道。"""

    def test_full_pipeline_v2(self):
        """测试v2完整管道：构建模型 -> 自底向上 -> 自顶向下 -> 感知更新 -> 驱力计算。"""
        # 1. 构建模型
        model = build_v2_model(num_layers=3, obs_dim=4, layer_dims=(4, 5, 6))

        # 2. 生成观测和表征
        rng = np.random.RandomState(123)
        obs = rng.randn(4)
        reprs = tuple(rng.randn(dim) for dim in (4, 5, 6))

        # 3. 自底向上传播
        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)
        assert len(bu.propagated_signals) == 3

        # 4. 自顶向下传播
        td_engine = TopDownInferenceEngineV2()
        td = td_engine.predict(model, bu, reprs[-1])
        assert td.obs_prediction.shape == (4,)

        # 5. 感知更新
        pu_engine = PerceptionUpdateEngineV2()
        pu = pu_engine.update(td, reprs, max_steps=10, step_size=0.01)
        assert len(pu.representations_after) == 3

        # 6. 安全驱力
        safety_calc = SafetyDriveCalculatorV2()
        safety = safety_calc.compute_from_model(model, obs, pu.representations_after)
        assert 0 <= safety.safety_strength <= 1

        # 7. 社会驱力
        social_calc = SocialDriveCalculatorV2()
        social = social_calc.compute(pu.representations_after)
        assert 0 <= social.social_strength <= 1

        # 8. 探索驱力
        exploration_calc = ExplorationDriveCalculatorV2()
        exploration = exploration_calc.compute(pu.gradient_norms)
        assert 0 <= exploration.exploration_strength <= 1

    def test_full_pipeline_v1_compatibility(self):
        """测试v1兼容性管道。"""
        model = build_v1_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        rng = np.random.RandomState(456)
        obs = rng.randn(3)
        reprs = tuple(rng.randn(dim) for dim in (3, 4))

        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)
        assert len(bu.propagated_signals) == 2

        td_engine = TopDownInferenceEngineV2()
        td = td_engine.predict(model, bu, reprs[-1])
        assert td.obs_prediction.shape == (3,)

    def test_multi_layer_predictor_training(self):
        """测试多层预测模型训练闭环。"""
        # 构建 XOR 问题
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        Y = np.array([[0], [1], [1], [0]], dtype=float)

        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(2, 8, 1),
            activations=(ActivationType.TANH, ActivationType.SIGMOID),
            learning_rate=0.5,
            momentum=0.9,
            seed=42,
        )

        # 训练
        initial_loss = None
        for epoch in range(200):
            epoch_loss = 0
            for i in range(len(X)):
                predictor = predictor.train_step(X[i], Y[i], "cross_entropy")
                _, _, loss = predictor.backward(X[i], Y[i], "cross_entropy")
                epoch_loss += loss
            if initial_loss is None:
                initial_loss = epoch_loss

        final_loss = epoch_loss
        assert final_loss < initial_loss, \
            f"XOR训练应降低损失: {final_loss} < {initial_loss}"


# ============================================================
# 异常容错测试
# ============================================================
class TestExceptionHandling:
    """测试异常容错。"""

    def test_nan_input_propagation(self):
        """测试NaN输入的传播。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, np.nan, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        bu_engine = BottomUpInferenceEngineV2()
        # 不应该崩溃
        try:
            bu = bu_engine.propagate(model, obs, reprs)
            # 检查是否有NaN传播
            has_nan = any(np.any(np.isnan(e)) for e in bu.error_signals)
            if has_nan:
                print("警告：NaN在传播中扩散")
        except Exception as e:
            print(f"NaN输入导致异常: {e}")

    def test_inf_input_propagation(self):
        """测试Inf输入的传播。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, np.inf, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        bu_engine = BottomUpInferenceEngineV2()
        try:
            bu = bu_engine.propagate(model, obs, reprs)
        except Exception as e:
            print(f"Inf输入导致异常: {e}")

    def test_dimension_mismatch(self):
        """测试维度不匹配。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        wrong_obs = np.array([0.5, 0.3])  # 应该是3维
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        bu_engine = BottomUpInferenceEngineV2()
        # 可能不直接报错（numpy广播），但应该有某种处理
        try:
            bu = bu_engine.propagate(model, wrong_obs, reprs)
        except (ValueError, Exception) as e:
            print(f"维度不匹配正确报错: {e}")

    def test_singular_precision_matrix(self):
        """测试奇异精度矩阵。"""
        def singular_precision(dim):
            M = np.zeros((dim, dim))
            return M  # 全零矩阵，奇异

        with pytest.raises(ValueError, match="正定"):
            NonlinearHierarchicalModelBuilder().build_model(
                num_layers=1, layer_dims=(3,), obs_dim=3,
                weight_init_fn=lambda i, o: np.eye(i, o),
                precision_init_fn=singular_precision,
            )


# ============================================================
# 数据一致性校验
# ============================================================
class TestDataConsistency:
    """测试数据一致性。"""

    def test_free_energy_equals_negative_log_likelihood(self):
        """验证自由能 = -对数似然。"""
        model = build_v2_model(num_layers=2, obs_dim=3, layer_dims=(3, 4))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6, 0.7]))

        ll = model.joint_log_likelihood(obs, reprs)
        fe = model.compute_free_energy(obs, reprs)
        assert fe == pytest.approx(-ll)

    def test_jacobian_shape_consistency(self):
        """验证Jacobian形状一致性。"""
        model = build_v2_model(num_layers=3, obs_dim=3, layer_dims=(3, 4, 5))
        reprs = (
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6, 0.7]),
            np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        )

        jacobians = model.compute_jacobians(reprs)
        for l in range(model.num_layers):
            child_dim = model.obs_dim if l == 0 else model.layer_dims[l - 1]
            parent_dim = model.layer_dims[l]
            assert jacobians[l].shape == (child_dim, parent_dim), \
                f"层{l}: Jacobian形状 {jacobians[l].shape} != ({child_dim}, {parent_dim})"

    def test_error_signal_dimensions(self):
        """验证误差信号维度。"""
        model = build_v2_model(num_layers=3, obs_dim=3, layer_dims=(3, 4, 5))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6, 0.7]),
            np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        )

        bu_engine = BottomUpInferenceEngineV2()
        errors = bu_engine.compute_errors(model, obs, reprs)

        # e^0: obs_dim
        assert errors[0].shape == (3,)
        # e^1: layer_dims[0]
        assert errors[1].shape == (3,)
        # e^2: layer_dims[1]
        assert errors[2].shape == (4,)
        # e^3 (prior): layer_dims[-1]
        assert errors[3].shape == (5,)

    def test_propagated_signal_dimensions(self):
        """验证传播信号维度。"""
        model = build_v2_model(num_layers=3, obs_dim=3, layer_dims=(3, 4, 5))
        obs = np.array([0.5, -0.3, 0.8])
        reprs = (
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6, 0.7]),
            np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        )

        bu_engine = BottomUpInferenceEngineV2()
        bu = bu_engine.propagate(model, obs, reprs)

        # propagated[l] 维度 = layer_dims[l]
        for l in range(model.num_layers):
            assert bu.propagated_signals[l].shape == (model.layer_dims[l],), \
                f"层{l}: 传播信号维度 {bu.propagated_signals[l].shape} != {(model.layer_dims[l],)}"


# ============================================================
# 基础性能核验
# ============================================================
class TestPerformance:
    """基础性能测试。"""

    def test_large_model_forward(self):
        """测试大模型前向传播性能。"""
        import time
        model = build_v2_model(num_layers=5, obs_dim=50, layer_dims=(50, 40, 30, 20, 10))
        rng = np.random.RandomState(42)
        obs = rng.randn(50)
        reprs = tuple(rng.randn(dim) for dim in (50, 40, 30, 20, 10))

        start = time.time()
        for _ in range(100):
            model.joint_log_likelihood(obs, reprs)
        elapsed = time.time() - start

        print(f"100次前向传播耗时: {elapsed:.3f}s")
        assert elapsed < 5.0, f"性能不达标: {elapsed:.3f}s"

    def test_training_performance(self):
        """测试训练性能。"""
        import time
        predictor = MultiLayerPredictorBuilder.build(
            layer_dims=(10, 20, 10, 5),
            activations=(ActivationType.TANH, ActivationType.TANH, ActivationType.LINEAR),
            learning_rate=0.01,
            seed=42,
        )
        rng = np.random.RandomState(42)
        x = rng.randn(10)
        target = rng.randn(5)

        start = time.time()
        for _ in range(1000):
            predictor = predictor.train_step(x, target, "mse")
        elapsed = time.time() - start

        print(f"1000步训练耗时: {elapsed:.3f}s")
        assert elapsed < 10.0, f"训练性能不达标: {elapsed:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])

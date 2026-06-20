"""阶段95：系统启动与初始化。

本阶段定义系统初始化顺序、配置加载、模型初始化、驱力设定、
安全护栏初始化和首次感知与预测，封装为SystemInitializer数据类组合IntegratedSystem。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

from .stage_094 import IntegratedSystem, SystemIntegrator, EnvironmentStateVector, DriveEngineState, DecisionEngineState, ErrorEvaluatorState
from .stage_093 import MainScheduler, MainSchedulerExecutor
from .stage_092 import SystemDynamicsState, GlobalDynamics, GlobalDynamicsCalculator
from .stage_091 import GlobalObjectiveCalculator
from .stage_090 import SafetyGuard, SafetyGuardState
from .stage_050 import CrossLayerProtocol
from .stage_070 import EvolutionValidationResult
from .stage_079 import MemoryManagerState
from .stage_080 import MemoryValidationResult
from .stage_081 import CPUMonitor
from .stage_082 import MemoryMonitor
from .stage_083 import StorageCleaner
from .stage_084 import FileProtectionEngine
from .stage_085 import ExploreFreezer
from .stage_086 import AutoRollback
from .stage_087 import RatioEnforcer
from .stage_088 import IdleReplayTrigger
from .stage_089 import StableExploreTrigger

# 用于构建默认对象的依赖
from .stage_060 import ErrorValidationResult
from .stage_061 import MicroAdjustTrigger
from .stage_062 import MicroAdjuster
from .stage_063 import MicroAdjusterState
from .stage_064 import StructuralReformTrigger
from .stage_065 import StructuralReformer
from .stage_066 import ReformSafetyConstraint
from .stage_067 import StructuralReformerState
from .stage_068 import ExploreExploitEnforcer
from .stage_069 import StableExploreBooster
from .stage_071 import ShortTermMemory
from .stage_072 import LongTermMemory
from .stage_073 import SampleValueEvaluator
from .stage_074 import PriorityReplay
from .stage_075 import GenerativeReplay
from .stage_076 import OfflineReplay
from .stage_077 import MemoryConsolidation
from .stage_078 import ModelSnapshot


# ============================================================
# 阶段95 枚举：初始化状态
# ============================================================
class InitStatus(Enum):
    """初始化状态。"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================
# 阶段95 数据类：初始化步骤结果
# ============================================================
@dataclass(frozen=True)
class InitStepResult:
    """单个初始化步骤的结果。

    Attributes:
        step_name: 步骤名称
        status: 初始化状态
        duration_seconds: 耗时（秒）
        error_message: 错误信息（失败时）
    """
    step_name: str
    status: InitStatus
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


# ============================================================
# 阶段95 数据类：系统初始化器
# ============================================================
@dataclass(frozen=True)
class SystemInitializer:
    """系统初始化器的完整状态描述。

    组合阶段94的IntegratedSystem，在集成系统基础上增加初始化状态。

    Attributes:
        integrated_system: 阶段94的集成系统（组合字段）
        config: 加载的配置字典
        init_results: 各步骤初始化结果
        total_init_time: 总初始化时间
        is_ready: 系统是否就绪
        bootstrap_completed: bootstrap是否完成
        initial_free_energy: 初始自由能
        initial_drive_gaps: 初始驱力缺口

    Invariants:
        - total_init_time <= 30.0
    """
    integrated_system: Optional[IntegratedSystem]
    config: Dict = field(default_factory=dict)
    init_results: Tuple[InitStepResult, ...] = ()
    total_init_time: float = 0.0
    is_ready: bool = False
    bootstrap_completed: bool = False
    initial_free_energy: float = 0.0
    initial_drive_gaps: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.70, 0.40]))

    def __post_init__(self):
        if self.total_init_time < 0:
            raise ValueError(f"total_init_time必须 >= 0，收到 {self.total_init_time}")
        if self.total_init_time > 30.0:
            raise ValueError(f"total_init_time必须 <= 30.0，收到 {self.total_init_time}")


# ============================================================
# 阶段95 核心接口：系统初始化器
# ============================================================
class SystemInitializerExecutor:
    """阶段95的核心接口：系统启动与初始化。

    实现配置加载、模块初始化、安全验证和bootstrap。

    核心约束：
        - 初始化时间 <= 30秒
        - 关键模块失败则终止启动
        - 非关键模块失败使用默认值
    """

    def __init__(self, config_path: str = "agent_config.yaml"):
        """初始化系统初始化器。

        Args:
            config_path: 配置文件路径

        Examples:
            >>> initializer = SystemInitializerExecutor("config.yaml")
            >>> initializer.config_path
            'config.yaml'
        """
        self.config_path = config_path
        self._config: Optional[Dict] = None
        self._init_time = 0.0
        # 构建默认booster链
        evr = ErrorValidationResult()
        mat = MicroAdjustTrigger(validation_result=evr)
        ma = MicroAdjuster(trigger=mat)
        mas = MicroAdjusterState(adjuster=ma)
        srt = StructuralReformTrigger(micro_adjuster_state=mas)
        sr = StructuralReformer(trigger=srt)
        rsc = ReformSafetyConstraint(reformer=sr)
        srs = StructuralReformerState(safety=rsc)
        eee = ExploreExploitEnforcer(reformer_state=srs)
        self._default_booster = StableExploreBooster(enforcer=eee)

    def load_config(self, config_path: str) -> Dict:
        """加载配置文件（A95.2）。

        Args:
            config_path: 配置文件路径

        Returns:
            验证后的配置字典

        Examples:
            >>> initializer = SystemInitializerExecutor()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> isinstance(config, dict)
            True
        """
        import os
        if not os.path.exists(config_path):
            return self._default_config()
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                return self._validate_config(config)
        except Exception:
            return self._default_config()

    def _default_config(self) -> Dict:
        """返回默认配置。"""
        return {
            "perception": {"n_layers": 3, "layer_dims": [64, 128, 256]},
            "drive": {"preset": "balanced", "accuracy_weight": 0.40, "coverage_weight": 0.35, "efficiency_weight": 0.25},
            "predictor": {"hidden_dim": 64},
            "memory": {"buffer_size": 10000},
            "safety": {"cpu_max": 0.20, "mem_max": 0.20, "disk_max": 0.80},
            "scheduler": {"surface_interval": 1, "middle_interval": 600, "ultimate_interval": 36000, "target_loop_time": 0.05},
            "objective": {"lambda_complexity": 0.01, "lambda_safety": 1.0}
        }

    def _validate_config(self, config: Dict) -> Dict:
        """验证并修正配置参数。"""
        # 验证层级数量
        perception = config.get("perception", {})
        n_layers = perception.get("n_layers", 3)
        n_layers = max(1, min(10, n_layers))
        perception["n_layers"] = n_layers
        config["perception"] = perception

        # 验证驱力权重和
        drive = config.get("drive", {})
        w_acc = drive.get("accuracy_weight", 0.40)
        w_cov = drive.get("coverage_weight", 0.35)
        w_eff = drive.get("efficiency_weight", 0.25)
        total = w_acc + w_cov + w_eff
        if total > 0 and abs(total - 1.0) > 1e-6:
            drive["accuracy_weight"] = w_acc / total
            drive["coverage_weight"] = w_cov / total
            drive["efficiency_weight"] = w_eff / total
        config["drive"] = drive

        # 验证CPU上限
        safety = config.get("safety", {})
        cpu_max = safety.get("cpu_max", 0.20)
        safety["cpu_max"] = max(0.01, min(0.50, cpu_max))
        mem_max = safety.get("mem_max", 0.20)
        safety["mem_max"] = max(0.01, min(0.50, mem_max))
        config["safety"] = safety

        # 验证安全系数
        objective = config.get("objective", {})
        lambda_c = objective.get("lambda_complexity", 0.01)
        lambda_s = objective.get("lambda_safety", 1.0)
        if lambda_s <= lambda_c:
            objective["lambda_safety"] = lambda_c + 1.0
        config["objective"] = objective

        return config

    def init_perception(self, config: Dict) -> EnvironmentStateVector:
        """初始化感知模块（A95.1-step2）。

        Args:
            config: 配置字典

        Returns:
            感知模块实例

        Examples:
            >>> initializer = SystemInitializerExecutor()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> perception = initializer.init_perception(config)
            >>> perception is not None
            True
        """
        layers = config.get("perception", {}).get("n_layers", 3)
        return EnvironmentStateVector(
            observation=np.zeros(layers),
            context=np.zeros(layers),
            timestamp=0.0,
            layer=0
        )

    def init_drive(self, config: Dict, preset: str = "balanced") -> DriveEngineState:
        """初始化驱力引擎（A95.4）。

        Args:
            config: 配置字典
            preset: 驱力预设名称

        Returns:
            驱力引擎实例

        Raises:
            ValueError: 当preset不支持时

        Examples:
            >>> initializer = SystemInitializerExecutor()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> drive = initializer.init_drive(config, "balanced")
            >>> drive is not None
            True
        """
        valid_presets = {"balanced", "aggressive", "conservative", "explorer"}
        if preset not in valid_presets:
            raise ValueError(f"不支持的preset: {preset}")
        weights = {
            "balanced": (0.40, 0.35, 0.25),
            "aggressive": (0.50, 0.40, 0.10),
            "conservative": (0.30, 0.30, 0.40),
            "explorer": (0.25, 0.55, 0.20)
        }
        w = weights[preset]
        return DriveEngineState(
            free_energy=1.0,
            expected_free_energy=2.0,
            accuracy_drive=w[0],
            coverage_drive=w[1],
            efficiency_drive=w[2],
            curiosity_drive=0.1,
            reward_drive=0.0,
            drive_gaps=(0.45, 0.70, 0.40),
            timestamp=0.0
        )

    def init_predictor(self, config: Dict) -> CrossLayerProtocol:
        """初始化预测模型（A95.3）。

        Args:
            config: 配置字典

        Returns:
            层级预测模型实例

        Examples:
            >>> initializer = SystemInitializerExecutor()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> predictor = initializer.init_predictor(config)
            >>> predictor is not None
            True
        """
        hidden_dim = config.get("predictor", {}).get("hidden_dim", 64)
        return CrossLayerProtocol(
            time_scale_ratio=(1, 10, 100),
            attention_weights=np.ones(3) / 3.0
        )

    def init_memory(self, config: Dict) -> MemoryManagerState:
        """初始化记忆管理器。

        Args:
            config: 配置字典

        Returns:
            记忆管理器实例

        Examples:
            >>> initializer = SystemInitializerExecutor()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> memory = initializer.init_memory(config)
            >>> memory.current_step == 0
            True
        """
        return MemoryManagerState(
            model_snapshot=ModelSnapshot(
                consolidation=MemoryConsolidation(
                    offline_replay=OfflineReplay(
                        generative_replay=GenerativeReplay(
                            replay=PriorityReplay(
                                evaluator=SampleValueEvaluator(
                                    ltm=LongTermMemory(
                                        stm=ShortTermMemory(
                                            validation_result=EvolutionValidationResult(
                                                booster=self._default_booster,
                                                micro_improvement_rate=0.0,
                                                micro_test_passed=True,
                                                reform_improvement=0.0,
                                                reform_test_passed=False,
                                                rollback_test_passed=True,
                                                rollback_correctness=1.0,
                                                ratio_in_range_pct=0.7,
                                                ratio_test_passed=True,
                                                boost_benefit=0.0,
                                                boost_test_passed=True,
                                                no_regression=True,
                                                regression_test_passed=True,
                                                all_tests_passed=True
                                            )
                                        ),
                                        db_path=":memory:"
                                    ),
                                    error_mean=0.0, error_var=1.0,
                                    alpha=0.4, beta=0.4, gamma=0.2,
                                    forget_factor=0.99, top_k_ratio=0.3
                                ),
                                alpha=0.6, beta=0.4, beta_start=0.4,
                                beta_frames=100000, batch_size=32
                            ),
                            latent_dim=32,
                            quality_threshold=0.1,
                            avg_reconstruction_error=0.0
                        ),
                        batch_size=128,
                        min_improvement=0.001,
                        max_rounds=10,
                        debounce_steps=5,
                        idle_threshold=10.0
                    ),
                    consolidate_threshold=1000.0,
                    forget_tau_base=86400.0,
                    spacing_factor=2.0,
                    retention_threshold=0.85
                ),
                snapshot_interval=10000,
                max_snapshots=10,
                max_chain_length=5
            ),
            total_samples_stored=0,
            total_samples_consolidated=0,
            total_replays=0,
            total_snapshots=0,
            total_rollbacks=0,
            current_step=0,
            is_idle=False,
            last_consolidate_step=0,
            last_snapshot_step=0
        )

    def init_safety_guard(self, config: Dict) -> SafetyGuardState:
        """初始化安全护栏（A95.5）。

        Args:
            config: 配置字典

        Returns:
            安全护栏实例

        Examples:
            >>> initializer = SystemInitializerExecutor()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> sg = initializer.init_safety_guard(config)
            >>> sg is not None
            True
        """
        # 构建完整安全护栏状态链
        evr = ErrorValidationResult()
        mat = MicroAdjustTrigger(validation_result=evr)
        ma = MicroAdjuster(trigger=mat)
        mas = MicroAdjusterState(adjuster=ma)
        srt = StructuralReformTrigger(micro_adjuster_state=mas)
        sr = StructuralReformer(trigger=srt)
        rsc = ReformSafetyConstraint(reformer=sr)
        srs = StructuralReformerState(safety=rsc)
        eee = ExploreExploitEnforcer(reformer_state=srs)
        seb = StableExploreBooster(enforcer=eee)
        evr70 = EvolutionValidationResult(booster=seb)
        stm = ShortTermMemory(validation_result=evr70)
        ltm = LongTermMemory(stm=stm)
        sve = SampleValueEvaluator(ltm=ltm)
        pr = PriorityReplay(evaluator=sve)
        gr = GenerativeReplay(replay=pr)
        or_ = OfflineReplay(generative_replay=gr)
        mc = MemoryConsolidation(offline_replay=or_)
        ms = ModelSnapshot(consolidation=mc)
        mms = MemoryManagerState(model_snapshot=ms)
        mvr = MemoryValidationResult(manager_state=mms)
        cm = CPUMonitor(validation_result=mvr)
        mm = MemoryMonitor(cpu_monitor=cm)
        sc = StorageCleaner(memory_monitor=mm)
        fpe = FileProtectionEngine(storage_cleaner=sc)
        ef = ExploreFreezer(file_protection=fpe)
        ar = AutoRollback(explore_freezer=ef)
        re = RatioEnforcer(auto_rollback=ar)
        irt = IdleReplayTrigger(ratio_enforcer=re)
        set_ = StableExploreTrigger(idle_replay_trigger=irt)
        return SafetyGuardState(stable_trigger=set_)

    def bootstrap(self, integrator: SystemIntegrator) -> Dict:
        """执行首次感知与预测（A95.6）。

        Args:
            integrator: 初始化后的系统集成器

        Returns:
            bootstrap结果字典

        Examples:
            >>> initializer = SystemInitializerExecutor()
            >>> config = initializer.load_config("agent_config.yaml")
            >>> sg = initializer.init_safety_guard(config)
            >>> from stage_092 import GlobalDynamicsCalculator
            >>> calc = GlobalDynamicsCalculator()
            >>> from stage_091 import GlobalObjectiveCalculator
            >>> goc = GlobalObjectiveCalculator()
            >>> obj = goc.compute(1.0, 2.0, 0.5, sg, 1.0)
            >>> ds = SystemDynamicsState(
            ...     mu=np.array([1.0]), action=np.array([0.5]),
            ...     weights=np.array([0.1]), theta=np.array([0.01]),
            ...     precision=np.array([1.0]),
            ...     mu_dot=np.zeros(1), action_dot=np.zeros(1),
            ...     weights_dot=np.zeros(1), theta_dot=np.zeros(1),
            ...     precision_dot=np.zeros(1))
            >>> gd = GlobalDynamics(global_objective=obj, dynamics_state=ds)
            >>> scheduler = MainScheduler(global_dynamics=gd)
            >>> integrator = SystemIntegrator(scheduler, sg)
            >>> result = initializer.bootstrap(integrator)
            >>> 'free_energy' in result
            True
        """
        obs = np.array([0.5, 0.3, 0.2])
        result = integrator.step(obs)
        return {
            "free_energy": result["objective"],
            "action": result["action"],
            "step_count": result["step_count"],
            "bootstrap_valid": True
        }

    def initialize(self, config_path: str = "agent_config.yaml") -> SystemInitializer:
        """执行完整初始化流程（A95.1）。

        load_config -> init_perception -> init_drive ->
        init_predictor -> init_memory -> init_safety ->
        start_main_loop

        Args:
            config_path: 配置文件路径

        Returns:
            系统初始化器状态

        Examples:
            >>> initializer = SystemInitializerExecutor()
            >>> result = initializer.initialize()
            >>> isinstance(result, SystemInitializer)
            True
        """
        import time
        start = time.time()
        results = []

        # Step 1: load_config
        try:
            self._config = self.load_config(config_path)
            results.append(InitStepResult("load_config", InitStatus.COMPLETED, 0.0))
        except Exception as e:
            results.append(InitStepResult("load_config", InitStatus.FAILED, 0.0, str(e)))
            self._config = self._default_config()
            results.append(InitStepResult("load_config_fallback", InitStatus.SKIPPED, 0.0))

        # Step 2: init_perception
        try:
            perception = self.init_perception(self._config)
            results.append(InitStepResult("init_perception", InitStatus.COMPLETED, 0.0))
        except Exception as e:
            results.append(InitStepResult("init_perception", InitStatus.FAILED, 0.0, str(e)))
            raise RuntimeError(f"感知初始化失败: {e}")

        # Step 3: init_drive
        try:
            drive = self.init_drive(self._config)
            results.append(InitStepResult("init_drive", InitStatus.COMPLETED, 0.0))
        except Exception as e:
            results.append(InitStepResult("init_drive", InitStatus.SKIPPED, 0.0, str(e)))
            drive = DriveEngineState()

        # Step 4: init_predictor
        try:
            predictor = self.init_predictor(self._config)
            results.append(InitStepResult("init_predictor", InitStatus.COMPLETED, 0.0))
        except Exception as e:
            results.append(InitStepResult("init_predictor", InitStatus.FAILED, 0.0, str(e)))
            raise RuntimeError(f"预测模型初始化失败: {e}")

        # Step 5: init_memory
        try:
            memory = self.init_memory(self._config)
            results.append(InitStepResult("init_memory", InitStatus.COMPLETED, 0.0))
        except Exception as e:
            results.append(InitStepResult("init_memory", InitStatus.SKIPPED, 0.0, str(e)))
            memory = MemoryManagerState(
                model_snapshot=ModelSnapshot(
                    consolidation=MemoryConsolidation(
                        offline_replay=OfflineReplay(
                            generative_replay=GenerativeReplay(
                                replay=PriorityReplay(
                                    evaluator=SampleValueEvaluator(
                                        ltm=LongTermMemory(stm=ShortTermMemory(validation_result=EvolutionValidationResult(booster=self._default_booster))),
                                        error_mean=0.0, error_var=1.0,
                                        alpha=0.4, beta=0.4, gamma=0.2,
                                        forget_factor=0.99, top_k_ratio=0.3
                                    ),
                                    alpha=0.6, beta=0.4, beta_start=0.4,
                                    beta_frames=100000, batch_size=32
                                ),
                                latent_dim=32, quality_threshold=0.1,
                                avg_reconstruction_error=0.0
                            ),
                            batch_size=128, min_improvement=0.001,
                            max_rounds=10, debounce_steps=5, idle_threshold=10.0
                        ),
                        consolidate_threshold=1000.0,
                        forget_tau_base=86400.0,
                        spacing_factor=2.0,
                        retention_threshold=0.85
                    ),
                    snapshot_interval=10000, max_snapshots=10, max_chain_length=5
                ),
                total_samples_stored=0, total_samples_consolidated=0,
                total_replays=0, total_snapshots=0, total_rollbacks=0,
                current_step=0, is_idle=False,
                last_consolidate_step=0, last_snapshot_step=0
            )

        # Step 6: init_safety
        try:
            safety = self.init_safety_guard(self._config)
            results.append(InitStepResult("init_safety", InitStatus.COMPLETED, 0.0))
        except Exception as e:
            results.append(InitStepResult("init_safety", InitStatus.FAILED, 0.0, str(e)))
            raise RuntimeError(f"安全护栏初始化失败: {e}")

        self._init_time = time.time() - start

        # 构建IntegratedSystem
        goc = GlobalObjectiveCalculator()
        obj = goc.compute(1.0, 2.0, 0.5, safety, 1.0)
        ds = SystemDynamicsState(
            mu=np.zeros(3), action=np.zeros(1),
            weights=np.zeros(3), theta=np.zeros(3),
            precision=np.ones(3),
            mu_dot=np.zeros(3), action_dot=np.zeros(1),
            weights_dot=np.zeros(3), theta_dot=np.zeros(3),
            precision_dot=np.zeros(3)
        )
        gd = GlobalDynamics(global_objective=obj, dynamics_state=ds)
        scheduler = MainScheduler(global_dynamics=gd)
        sg = SafetyGuard()
        integrator = SystemIntegrator(scheduler, sg, self._config)

        # Bootstrap
        bootstrap_result = self.bootstrap(integrator)

        # 构建IntegratedSystem
        integrated = integrator.get_state()

        return SystemInitializer(
            integrated_system=integrated,
            config=self._config,
            init_results=tuple(results),
            total_init_time=self._init_time,
            is_ready=True,
            bootstrap_completed=bootstrap_result.get("bootstrap_valid", False),
            initial_free_energy=bootstrap_result.get("free_energy", 0.0),
            initial_drive_gaps=np.array([0.45, 0.70, 0.40])
        )

    def verify(self, initializer: SystemInitializer) -> bool:
        """验证初始化后的系统。

        Args:
            initializer: 待验证的初始化器状态

        Returns:
            当系统已就绪且关键组件存在时返回True

        Examples:
            >>> initializer = SystemInitializerExecutor()
            >>> result = initializer.initialize()
            >>> initializer.verify(result)
            True
        """
        if not initializer.is_ready:
            return False
        if initializer.integrated_system is None:
            return False
        return True

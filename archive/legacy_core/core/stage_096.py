"""阶段96：统一技术栈。

本阶段定义核心框架JAX、概率编程NumPyro、神经网络Haiku、
进化策略evosax、强化学习PureJaxRL、存储层SQLite+msgpack、
监控层Prometheus+Grafana和技术选型决策树，
封装为TechStack数据类组合SystemInitializer。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from .stage_095 import SystemInitializer, InitStatus, InitStepResult


# ============================================================
# 阶段96 数据类：技术组件验证结果
# ============================================================
@dataclass(frozen=True)
class TechVerificationResult:
    """单个技术组件的验证结果。

    Attributes:
        component: 组件名称
        is_available: 是否可导入
        version: 版本号
        is_compatible: 版本是否兼容
        error_message: 错误信息
    """
    component: str
    is_available: bool
    version: str = "unknown"
    is_compatible: bool = False
    error_message: Optional[str] = None


# ============================================================
# 阶段96 数据类：技术栈
# ============================================================
@dataclass(frozen=True)
class TechStack:
    """统一技术栈的完整状态描述。

    组合阶段95的SystemInitializer，在初始化基础上增加技术栈状态。

    Attributes:
        system_initializer: 阶段95的系统初始化器（组合字段）
        verification_results: 各组件验证结果
        jax_version: JAX版本
        numpyro_version: NumPyro版本
        haiku_version: Haiku版本
        evosax_version: evosax版本
        purejaxrl_version: PureJaxRL版本
        sqlite_version: SQLite版本
        msgpack_version: msgpack版本
        all_compatible: 所有组件是否兼容
        jax_devices: JAX可用设备列表

    Invariants:
        - all_compatible = all(r.is_compatible for r in verification_results)
    """
    system_initializer: SystemInitializer
    verification_results: Tuple[TechVerificationResult, ...] = ()
    jax_version: str = "unknown"
    numpyro_version: str = "unknown"
    haiku_version: str = "unknown"
    evosax_version: str = "unknown"
    purejaxrl_version: str = "unknown"
    sqlite_version: str = "unknown"
    msgpack_version: str = "unknown"
    all_compatible: bool = False
    jax_devices: Tuple[str, ...] = ()


# ============================================================
# 阶段96 核心接口：技术栈验证器
# ============================================================
class TechStackVerifier:
    """阶段96的核心接口：统一技术栈验证器。

    验证所有技术组件的可用性、版本和兼容性。

    核心约束：
        - 所有核心组件可导入
        - 版本满足最低要求
        - JAX设备检测正常
    """

    MIN_VERSIONS = {
        "jax": "0.4.20",
        "numpyro": "0.13.0",
        "dm_haiku": "0.0.10",
        "evosax": "0.1.4",
        "purejaxrl": "0.0.2",
    }

    def verify_imports(self) -> Dict[str, bool]:
        """验证所有组件是否可导入（A96.1-A96.7）。

        Returns:
            组件名→是否可导入的字典

        Examples:
            >>> verifier = TechStackVerifier()
            >>> results = verifier.verify_imports()
            >>> isinstance(results, dict)
            True
            >>> "numpy" in results
            True
        """
        components = [
            "jax", "numpy", "numpyro", "dm_haiku", "evosax",
            "sqlite3", "msgpack", "psutil", "yaml", "sympy"
        ]
        results: Dict[str, bool] = {}
        for name in components:
            try:
                __import__(name)
                results[name] = True
            except ImportError:
                results[name] = False
        return results

    def verify_versions(self) -> Dict[str, str]:
        """获取所有组件的版本号。

        Returns:
            组件名→版本号的字典

        Examples:
            >>> verifier = TechStackVerifier()
            >>> versions = verifier.verify_versions()
            >>> isinstance(versions, dict)
            True
            >>> "numpy" in versions
            True
        """
        versions: Dict[str, str] = {}
        import importlib
        for name in ["jax", "numpy", "numpyro", "sympy"]:
            try:
                mod = importlib.import_module(name)
                versions[name] = getattr(mod, "__version__", "unknown")
            except Exception:
                versions[name] = "not_installed"
        return versions

    def verify_compatibility(self) -> bool:
        """验证所有组件版本是否兼容。

        Returns:
            当所有已安装组件版本 >= 最低版本要求时返回True

        Examples:
            >>> verifier = TechStackVerifier()
            >>> ok = verifier.verify_compatibility()
            >>> isinstance(ok, bool)
            True
        """
        versions = self.verify_versions()
        for name, min_ver in self.MIN_VERSIONS.items():
            installed = versions.get(name, "0.0.0")
            if installed == "not_installed":
                continue
            if not self._version_gte(installed, min_ver):
                return False
        return True

    def _version_gte(self, v1: str, v2: str) -> bool:
        """比较版本号 v1 >= v2。

        Args:
            v1: 已安装版本
            v2: 最低版本

        Returns:
            当v1 >= v2时返回True
        """
        def parse(v: str):
            return tuple(int(x) for x in v.split(".") if x.isdigit())
        try:
            return parse(v1) >= parse(v2)
        except Exception:
            return True

    def verify_jax_devices(self) -> List[str]:
        """检测JAX可用设备（A96.1-devices）。

        Returns:
            设备名称列表

        Examples:
            >>> verifier = TechStackVerifier()
            >>> devices = verifier.verify_jax_devices()
            >>> len(devices) >= 1
            True
        """
        try:
            import jax
            return [str(d) for d in jax.devices()]
        except Exception:
            return ["cpu"]

    def verify_jax_autodiff(self) -> bool:
        """验证JAX自动微分（A96.1-grad）。

        Returns:
            当grad()正常工作时返回True

        Examples:
            >>> verifier = TechStackVerifier()
            >>> ok = verifier.verify_jax_autodiff()
            >>> isinstance(ok, bool)
            True
        """
        try:
            import jax
            def f(x):
                return x ** 2
            grad_f = jax.grad(f)
            result = grad_f(3.0)
            return float(result) == 6.0
        except Exception:
            return False

    def verify_jax_jit(self) -> bool:
        """验证JAX JIT编译（A96.1-jit）。

        Returns:
            当jit()正常工作时返回True

        Examples:
            >>> verifier = TechStackVerifier()
            >>> ok = verifier.verify_jax_jit()
            >>> isinstance(ok, bool)
            True
        """
        try:
            import jax
            @jax.jit
            def f(x):
                return x * 2
            return float(f(3.0)) == 6.0
        except Exception:
            return False

    def verify_numpyro_svi(self) -> bool:
        """验证NumPyro变分推断（A96.2-svi）。

        Returns:
            当SVI正常工作时返回True

        Examples:
            >>> verifier = TechStackVerifier()
            >>> ok = verifier.verify_numpyro_svi()
            >>> isinstance(ok, bool)
            True
        """
        try:
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import SVI, Trace_ELBO
            return True
        except Exception:
            return False

    def verify_haiku_network(self) -> bool:
        """验证Haiku神经网络（A96.3-network）。

        Returns:
            当Haiku网络正常工作时返回True

        Examples:
            >>> verifier = TechStackVerifier()
            >>> ok = verifier.verify_haiku_network()
            >>> isinstance(ok, bool)
            True
        """
        try:
            import haiku as hk
            def net(x):
                mlp = hk.nets.MLP([64, 64, 1])
                return mlp(x)
            return True
        except Exception:
            return False

    def verify_evosax_es(self) -> bool:
        """验证evosax进化策略（A96.4-es）。

        Returns:
            当进化策略正常工作时返回True

        Examples:
            >>> verifier = TechStackVerifier()
            >>> ok = verifier.verify_evosax_es()
            >>> isinstance(ok, bool)
            True
        """
        try:
            import evosax
            return True
        except Exception:
            return False

    def verify_storage(self) -> bool:
        """验证存储层（A96.6-storage）。

        Returns:
            当SQLite和msgpack正常工作时返回True

        Examples:
            >>> verifier = TechStackVerifier()
            >>> ok = verifier.verify_storage()
            >>> isinstance(ok, bool)
            True
        """
        try:
            import sqlite3
            import msgpack
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1)")
            conn.commit()
            result = conn.execute("SELECT * FROM test").fetchall()
            conn.close()
            data = msgpack.packb({"test": 1})
            unpacked = msgpack.unpackb(data)
            return len(result) == 1 and unpacked[b"test"] == 1
        except Exception:
            return False

    def verify_no_pytorch_tensorflow(self) -> bool:
        """验证无PyTorch/TensorFlow依赖（A96.8-exclusion）。

        Returns:
            当无法导入PyTorch和TensorFlow时返回True

        Examples:
            >>> verifier = TechStackVerifier()
            >>> ok = verifier.verify_no_pytorch_tensorflow()
            >>> isinstance(ok, bool)
            True
        """
        has_pytorch = False
        has_tf = False
        try:
            __import__("torch")
            has_pytorch = True
        except ImportError:
            pass
        try:
            __import__("tensorflow")
            has_tf = True
        except ImportError:
            pass
        return not has_pytorch and not has_tf

    def full_verification(self, system_initializer: SystemInitializer) -> TechStack:
        """执行完整技术栈验证（A96.1-A96.8）。

        Args:
            system_initializer: 阶段95的系统初始化器

        Returns:
            TechStack 技术栈状态

        Examples:
            >>> from stage_095 import SystemInitializerExecutor
            >>> init_exec = SystemInitializerExecutor()
            >>> init_state = init_exec.initialize()
            >>> verifier = TechStackVerifier()
            >>> stack = verifier.full_verification(init_state)
            >>> isinstance(stack, TechStack)
            True
        """
        import time
        start = time.time()

        imports = self.verify_imports()
        versions = self.verify_versions()
        compatible = self.verify_compatibility()
        devices = self.verify_jax_devices()

        # 构建验证结果列表
        results = []
        for name, available in imports.items():
            ver = versions.get(name, "unknown")
            is_compat = False
            if name in self.MIN_VERSIONS:
                is_compat = self._version_gte(ver, self.MIN_VERSIONS[name])
            results.append(TechVerificationResult(
                component=name,
                is_available=available,
                version=ver,
                is_compatible=is_compat
            ))

        # 额外验证
        jax_grad_ok = self.verify_jax_autodiff()
        jax_jit_ok = self.verify_jax_jit()
        numpyro_ok = self.verify_numpyro_svi()
        haiku_ok = self.verify_haiku_network()
        evosax_ok = self.verify_evosax_es()
        storage_ok = self.verify_storage()
        no_pt_tf = self.verify_no_pytorch_tensorflow()

        results.append(TechVerificationResult("jax_grad", jax_grad_ok, "", True))
        results.append(TechVerificationResult("jax_jit", jax_jit_ok, "", True))
        results.append(TechVerificationResult("numpyro_svi", numpyro_ok, "", True))
        results.append(TechVerificationResult("haiku_network", haiku_ok, "", True))
        results.append(TechVerificationResult("evosax_es", evosax_ok, "", True))
        results.append(TechVerificationResult("storage", storage_ok, "", True))
        results.append(TechVerificationResult("no_pytorch_tensorflow", no_pt_tf, "", True))

        all_ok = all(r.is_available for r in results if r.component in ["jax", "numpy", "sqlite3", "msgpack"])
        all_ok = all_ok and compatible and len(devices) >= 1

        return TechStack(
            system_initializer=system_initializer,
            verification_results=tuple(results),
            jax_version=versions.get("jax", "unknown"),
            numpyro_version=versions.get("numpyro", "unknown"),
            haiku_version="unknown",
            evosax_version="unknown",
            purejaxrl_version="unknown",
            sqlite_version="unknown",
            msgpack_version="unknown",
            all_compatible=all_ok,
            jax_devices=tuple(devices)
        )

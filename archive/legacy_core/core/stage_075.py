"""阶段75：生成式回放——VAE编码压缩、隐式存储与重构质量保证。

本阶段实现生成式回放机制。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

from .stage_074 import PriorityReplay


@dataclass(frozen=True)
class GenerativeReplay:
    """生成式回放的完整描述。

    组合阶段74的PriorityReplay，在优先级回放的基础上增加
    VAE生成能力。

    Attributes:
        replay: 阶段74的优先级回放（组合字段）
        latent_dim: 隐编码维度d_z
        encoder_params: 编码器参数
        decoder_params: 解码器参数
        quality_threshold: 重构质量阈值
        compression_ratio: 压缩比
        avg_reconstruction_error: 平均重构误差
        vae_retrain_count: VAE重训练次数

    Invariants:
        - latent_dim > 0
        - quality_threshold > 0
        - avg_reconstruction_error >= 0
    """
    replay: PriorityReplay
    latent_dim: int = 32
    encoder_params: Optional[Dict] = None
    decoder_params: Optional[Dict] = None
    quality_threshold: float = 0.05
    compression_ratio: float = 16.0
    avg_reconstruction_error: float = 0.0
    vae_retrain_count: int = 0

    def __post_init__(self):
        if self.latent_dim <= 0:
            raise ValueError(
                f"隐编码维度必须 > 0，收到 {self.latent_dim}"
            )
        if self.quality_threshold <= 0:
            raise ValueError(
                f"质量阈值必须 > 0，收到 {self.quality_threshold}"
            )
        if self.avg_reconstruction_error < 0:
            raise ValueError(
                f"平均重构误差必须 >= 0，收到 {self.avg_reconstruction_error}"
            )

    @property
    def quality_ok(self) -> bool:
        """重构质量是否达标。"""
        return self.avg_reconstruction_error < self.quality_threshold


class GenerativeReplayInterface:
    """阶段75的核心接口：生成式回放。

    实现VAE编码、解码、压缩存储和生成质量评估。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 重构误差低于质量阈值
        - 压缩比 > 1
        - 隐编码维度远小于原始维度
    """

    def encode(
        self,
        x: np.ndarray,
        encoder_params: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """VAE编码（A75.1）。

        Args:
            x: 输入样本，shape (batch, d_x)
            encoder_params: 编码器参数，包含 "W_mu", "b_mu",
                "W_logvar", "b_logvar"

        Returns:
            (z, mu, log_var)
            - z: 隐编码，shape (batch, d_z)
            - mu: 均值，shape (batch, d_z)
            - log_var: 对数方差，shape (batch, d_z)

        Raises:
            ValueError: 当encoder_params缺少必要键时
        """
        required = {"W_mu", "b_mu", "W_logvar", "b_logvar"}
        missing = required - set(encoder_params.keys())
        if missing:
            raise ValueError(f"encoder_params缺少键: {missing}")
        W_mu = encoder_params["W_mu"]
        b_mu = encoder_params["b_mu"]
        W_logvar = encoder_params["W_logvar"]
        b_logvar = encoder_params["b_logvar"]
        mu = x @ W_mu + b_mu
        log_var = x @ W_logvar + b_logvar
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        return z, mu, log_var

    def decode(
        self,
        z: np.ndarray,
        decoder_params: Dict
    ) -> np.ndarray:
        """VAE解码（A75.2）。

        Args:
            z: 隐编码，shape (batch, d_z)
            decoder_params: 解码器参数，包含 "W_out", "b_out"

        Returns:
            重构样本，shape (batch, d_x)

        Raises:
            ValueError: 当decoder_params缺少必要键时
        """
        required = {"W_out", "b_out"}
        missing = required - set(decoder_params.keys())
        if missing:
            raise ValueError(f"decoder_params缺少键: {missing}")
        W_out = decoder_params["W_out"]
        b_out = decoder_params["b_out"]
        x_hat = z @ W_out + b_out
        return x_hat

    def compress_store(
        self,
        x: np.ndarray,
        metadata: Dict,
        encoder_params: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """压缩存储（A75.3）。

        Args:
            x: 原始样本，shape (batch, d_x)
            metadata: 元数据字典
            encoder_params: 编码器参数

        Returns:
            (latent_z, metadata) 压缩后的存储记录
        """
        z, mu, log_var = self.encode(x, encoder_params)
        meta_out = dict(metadata)
        meta_out["latent_z"] = z
        meta_out["mu"] = mu
        meta_out["log_var"] = log_var
        return z, meta_out

    def generate(
        self,
        z: np.ndarray,
        decoder_params: Dict
    ) -> np.ndarray:
        """重新生成（A75.4）。

        Args:
            z: 隐编码，shape (batch, d_z)
            decoder_params: 解码器参数

        Returns:
            重构样本，shape (batch, d_x)
        """
        return self.decode(z, decoder_params)

    def evaluate_quality(
        self,
        x_original: np.ndarray,
        x_reconstructed: np.ndarray,
        quality_threshold: float = 0.05
    ) -> Tuple[float, bool]:
        """评估生成质量（A75.5）。

        Args:
            x_original: 原始样本，shape (batch, d_x)
            x_reconstructed: 重构样本，shape (batch, d_x)
            quality_threshold: 质量阈值

        Returns:
            (reconstruction_error, quality_ok)
            - reconstruction_error: 均方误差
            - quality_ok: 误差是否低于阈值

        Raises:
            ValueError: 当输入形状不匹配时
        """
        if x_original.shape != x_reconstructed.shape:
            raise ValueError(
                f"形状不匹配: {x_original.shape} vs {x_reconstructed.shape}"
            )
        mse = float(np.mean((x_original - x_reconstructed) ** 2))
        quality_ok = mse < quality_threshold
        return mse, quality_ok

    def interpolate(
        self,
        z1: np.ndarray,
        z2: np.ndarray,
        lambda_val: float
    ) -> np.ndarray:
        """隐空间插值生成（A75.4-interp）。

        Args:
            z1: 隐编码1，shape (d_z,)
            z2: 隐编码2，shape (d_z,)
            lambda_val: 插值系数 ∈ [0, 1]

        Returns:
            插值隐编码，shape (d_z,)

        Raises:
            ValueError: 当lambda_val不在[0,1]范围内时
            ValueError: 当z1和z2形状不匹配时
        """
        if not (0 <= lambda_val <= 1):
            raise ValueError(
                f"lambda_val必须 ∈ [0,1]，收到 {lambda_val}"
            )
        if z1.shape != z2.shape:
            raise ValueError(
                f"z1和z2形状不匹配: {z1.shape} vs {z2.shape}"
            )
        return (1 - lambda_val) * z1 + lambda_val * z2

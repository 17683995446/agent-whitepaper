"""阶段I：真实环境适配层包。

公共API：
    EnvironmentProtocol: 环境协议抽象基类
    StateEncoder: 状态编码器抽象基类
    GridWorldStateEncoder: 网格世界状态编码器
    BitmapStateEncoder: 位图状态编码器
    EnvironmentAdapter: 环境适配器
    FunctionCallEnvironment: 函数调用环境
    run_episode_universal: 通用单回合运行
    replay_learning_universal: 通用回放学习
    create_universal_agent: 创建通用Agent
    MultiEnvironmentTrainer: 多环境训练器
"""

from .environment_interface import (
    EnvironmentProtocol,
    StateEncoder,
    GridWorldStateEncoder,
    BitmapStateEncoder,
    EnvironmentAdapter,
)
from .function_env import FunctionCallEnvironment
from .adapters import (
    run_episode_universal,
    replay_learning_universal,
    create_universal_agent,
    MultiEnvironmentTrainer,
)

__all__ = [
    "EnvironmentProtocol",
    "StateEncoder",
    "GridWorldStateEncoder",
    "BitmapStateEncoder",
    "EnvironmentAdapter",
    "FunctionCallEnvironment",
    "run_episode_universal",
    "replay_learning_universal",
    "create_universal_agent",
    "MultiEnvironmentTrainer",
]

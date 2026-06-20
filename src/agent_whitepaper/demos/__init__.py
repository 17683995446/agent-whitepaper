"""演示包。

公共API：
    main: 端到端演示主入口
    demo_capability_1_self_driven_long_task: 能力1-自驱长任务演示
    demo_capability_2_skill_enhancement: 能力2-技能增强演示
    demo_capability_3_multi_agent_collaboration: 能力3-多Agent协作演示
    demo_final_verification: 最终验证演示
"""

from .e2e_demo import (
    main,
    demo_capability_1_self_driven_long_task,
    demo_capability_2_skill_enhancement,
    demo_capability_3_multi_agent_collaboration,
    demo_final_verification,
)

__all__ = [
    "main",
    "demo_capability_1_self_driven_long_task",
    "demo_capability_2_skill_enhancement",
    "demo_capability_3_multi_agent_collaboration",
    "demo_final_verification",
]

"""示例01：单Agent训练 - 最简入门。

本示例演示：
1. 创建网格世界环境
2. 创建自驱Agent
3. 训练多个情节
4. 查看学习效果

这是使用 Agent Whitepaper 的最小完整示例。
运行：PYTHONPATH=src python examples/01_single_agent_training.py
"""

from agent_whitepaper import GridWorld, SelfDrivenAgent


def main() -> None:
    print("=" * 60)
    print("  示例01：单Agent训练")
    print("=" * 60)

    # 1. 创建环境：5x5网格，Agent需从(0,0)走到(4,4)
    env = GridWorld(size=5, seed=42)
    print(f"\n环境: {env.size}x{env.size} 网格")
    print(f"动作空间: {env.action_space.n} 个动作")
    print(f"观测维度: {env.observation_dim}")

    # 2. 创建Agent：25个状态（5x5），4个动作
    agent = SelfDrivenAgent(
        action_space=env.action_space,
        n_states=25,
        learning_rate=0.1,
        gamma=0.99,
        seed=42,
    )
    print("\nAgent: n_states=25, lr=0.1, gamma=0.99")

    # 3. 训练100个情节
    print("\n训练中（100个情节）...")
    n_episodes = 100
    for i in range(n_episodes):
        episode = agent.run_episode(env, max_steps=100)
        if (i + 1) % 20 == 0:
            print(
                f"  情节 {i + 1:3d}/{n_episodes}: "
                f"奖励={episode.total_reward:+.2f} "
                f"成功={'是' if episode.success else '否'} "
                f"成功率={agent.stats.success_rate:.1%}"
            )

    # 4. 最终统计
    print("\n" + "=" * 60)
    print("  训练结果")
    print("=" * 60)
    print(f"  总情节数: {agent.stats.total_episodes}")
    print(f"  总步数: {agent.stats.total_steps}")
    print(f"  成功率: {agent.stats.success_rate:.1%}")
    print(f"  平均奖励: {agent.stats.avg_reward:.2f}")
    print(f"  Q表大小: {agent.q_table.n_states}x{agent.q_table.n_actions}")

    # 5. 最优策略展示
    print("\n  最优动作（每个状态的最优动作）:")
    actions = ["↑", "↓", "←", "→"]
    for row in range(5):
        line = "  "
        for col in range(5):
            state = row * 5 + col
            best = agent.q_table.best_action(state)
            line += actions[best] + " "
        print(line)


if __name__ == "__main__":
    main()

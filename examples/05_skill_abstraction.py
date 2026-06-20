"""示例05：技能抽象与元学习。

本示例演示：
1. 训练Agent产生情节
2. 从成功情节抽象技能（n-gram模式）
3. 查看抽象出的技能
4. 元学习优化学习参数

运行：PYTHONPATH=src python examples/05_skill_abstraction.py
"""

from agent_whitepaper import GridWorld, SelfDrivenAgent
from agent_whitepaper.stage_b.meta_learner import MetaLearner
from agent_whitepaper.stage_b.skill_abstractor import SkillAbstractor, SkillLibrary


def main() -> None:
    print("=" * 60)
    print("  示例05：技能抽象与元学习")
    print("=" * 60)

    # 1. 创建环境与Agent
    env = GridWorld(size=5, seed=42)
    agent = SelfDrivenAgent(
        action_space=env.action_space,
        n_states=25,
        learning_rate=0.2,
        seed=42,
    )

    # 2. 训练产生情节
    print("\n阶段1: 训练Agent产生情节（30个情节）...")
    episodes = []
    for i in range(30):
        episode = agent.run_episode(env, max_steps=50)
        episodes.append(episode)
    successful = [ep for ep in episodes if ep.success]
    print(f"  总情节: {len(episodes)}, 成功: {len(successful)}")
    print(f"  成功率: {agent.stats.success_rate:.1%}")

    # 3. 技能抽象
    print("\n阶段2: 从成功情节抽象技能...")
    skill_library = SkillLibrary()
    abstractor = SkillAbstractor(
        min_n=2, max_n=4, min_frequency=2, min_success_rate=0.5
    )
    new_skills = abstractor.abstract_from_episodes(episodes, skill_library)
    print(f"  抽象出新技能: {len(new_skills)} 个")
    for skill in new_skills[:5]:
        print(f"    {skill.name}: 动作序列={skill.action_sequence}")

    # 4. 元学习
    print("\n阶段3: 元学习优化学习参数...")
    meta_learner = MetaLearner(window_size=10)

    # 记录学习曲线
    for ep in episodes:
        meta_learner.record_episode(ep.total_reward, ep.success)

    # 检查是否应调整参数
    if meta_learner.should_adapt():
        new_params = meta_learner.adapt_parameters()
        print("  参数调整:")
        print(f"    学习率: {new_params.learning_rate:.3f}")
        print(f"    探索率: {new_params.exploration_rate:.3f}")

    summary = meta_learner.get_progress_summary()
    print("  学习进度:")
    print(f"    状态: {summary.get('status', 'unknown')}")
    print(f"    情节数: {summary.get('episodes', 0)}")
    if "latest_avg_reward" in summary:
        print(f"    平均奖励: {summary['latest_avg_reward']:.2f}")
        print(f"    成功率: {summary['latest_success_rate']:.1%}")
        print(f"    学习速度: {summary['learning_speed']:.3f}")
        print(f"    稳定性: {summary['stability']:.3f}")

    # 5. 技能库统计
    print("\n阶段4: 技能库统计...")
    all_skills = skill_library.get_all_skills()
    print(f"  技能库总数: {len(all_skills)}")
    for skill in all_skills[:5]:
        print(
            f"    {skill.name}: "
            f"长度={skill.length}, "
            f"期望奖励={skill.expected_reward:.2f}, "
            f"成功率={skill.success_rate:.1%}"
        )

    # 6. 总结
    print("\n" + "=" * 60)
    print("  总结")
    print("=" * 60)
    print(f"  训练情节: {len(episodes)}")
    print(f"  成功情节: {len(successful)}")
    print(f"  抽象技能: {len(new_skills)}")
    print(f"  Agent成功率: {agent.stats.success_rate:.1%}")
    print("\n  技能抽象让Agent从'重复探索'变为'复用经验'，")
    print("  元学习让Agent从'固定参数'变为'自适应参数'。")


if __name__ == "__main__":
    main()

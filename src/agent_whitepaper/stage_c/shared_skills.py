"""阶段C-5：共享技能库 - 多Agent间共享学到的技能。

第一性原理：
    单个Agent学到的技能是有限的。如果Agent间能共享技能，
    则整个团队的能力会指数级增长——每个Agent都能利用其他Agent的经验。

    共享技能库是"集体智慧的存储"：
    - Agent A学会"开门"技能 → 上传到共享库
    - Agent B遇到门 → 从共享库下载"开门"技能
    - Agent C改进了"开门"技能 → 更新共享库

    这实现了真正的"自主增强技能"：不是每个Agent从零学习，
    而是站在所有Agent的肩膀上。
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent_whitepaper.stage_b.skill_abstractor import Skill, SkillLibrary


# ============================================================
# 共享技能来源记录
# ============================================================
@dataclass(frozen=True)
class SharedSkill:
    """共享技能：带来源和验证信息的技能。

    Attributes:
        skill: 原始技能
        contributor_id: 贡献者Agent ID
        contribution_time: 贡献时间
        endorsements: 认可数（其他Agent验证有效）
        downloads: 下载次数
        rating: 评分 [0,5]
        verified: 是否经验证
    """
    skill: Skill
    contributor_id: str
    contribution_time: float = field(default_factory=time.time)
    endorsements: int = 0
    downloads: int = 0
    rating: float = 3.0
    verified: bool = False

    @property
    def quality_score(self) -> float:
        """技能质量评分（综合成功率、认可、评分）。"""
        success_component = self.skill.success_rate
        endorsement_component = min(1.0, self.endorsements / 5.0)
        rating_component = self.rating / 5.0
        return 0.4 * success_component + 0.3 * endorsement_component + 0.3 * rating_component


# ============================================================
# 共享技能库
# ============================================================
class SharedSkillLibrary:
    """共享技能库：所有Agent共享的技能存储。

    功能：
    1. 贡献：Agent上传学会的技能
    2. 检索：按前置条件/名称检索技能
    3. 下载：Agent下载技能到本地
    4. 评价：使用后评价技能质量
    5. 验证：多Agent验证后标记为可信
    """

    def __init__(self):
        self._shared_skills: Dict[str, SharedSkill] = {}
        self._index_by_precondition: Dict[Tuple[int, ...], List[str]] = {}
        self._index_by_specialization: Dict[str, List[str]] = {}
        self._download_history: List[Tuple[str, str, float]] = []  # (skill_name, agent_id, time)
        self._endorsement_history: List[Tuple[str, str, float]] = []  # (skill_name, agent_id, rating)

    def __len__(self) -> int:
        return len(self._shared_skills)

    def contribute(
        self,
        skill: Skill,
        contributor_id: str,
        specialization: str = "general"
    ) -> str:
        """Agent贡献技能到共享库。

        Returns:
            技能在共享库中的名称
        """
        shared_name = f"shared_{skill.name}"

        if shared_name in self._shared_skills:
            # 已存在，更新（取更好的版本）
            existing = self._shared_skills[shared_name]
            if skill.success_rate > existing.skill.success_rate:
                # 新版本更好，替换
                shared = SharedSkill(
                    skill=skill,
                    contributor_id=contributor_id,
                    endorsements=existing.endorsements,
                    downloads=existing.downloads,
                    rating=existing.rating,
                    verified=existing.verified,
                )
                self._shared_skills[shared_name] = shared
        else:
            # 新技能
            shared = SharedSkill(
                skill=skill,
                contributor_id=contributor_id,
            )
            self._shared_skills[shared_name] = shared
            # 更新索引
            if skill.precondition not in self._index_by_precondition:
                self._index_by_precondition[skill.precondition] = []
            self._index_by_precondition[skill.precondition].append(shared_name)
            if specialization not in self._index_by_specialization:
                self._index_by_specialization[specialization] = []
            self._index_by_specialization[specialization].append(shared_name)

        return shared_name

    def download(
        self,
        skill_name: str,
        downloader_id: str
    ) -> Optional[Skill]:
        """Agent从共享库下载技能。

        Returns:
            技能对象，如果不存在返回None
        """
        if skill_name not in self._shared_skills:
            return None
        shared = self._shared_skills[skill_name]
        shared_downloads = shared.downloads + 1
        # 更新下载计数
        self._shared_skills[skill_name] = SharedSkill(
            skill=shared.skill,
            contributor_id=shared.contributor_id,
            contribution_time=shared.contribution_time,
            endorsements=shared.endorsements,
            downloads=shared_downloads,
            rating=shared.rating,
            verified=shared.verified,
        )
        self._download_history.append((skill_name, downloader_id, time.time()))
        return shared.skill

    def endorse(
        self,
        skill_name: str,
        endorser_id: str,
        rating: float
    ) -> bool:
        """Agent评价技能（使用后）。"""
        if skill_name not in self._shared_skills:
            return False
        if not (0 <= rating <= 5):
            raise ValueError("rating 必须 ∈ [0,5]")

        shared = self._shared_skills[skill_name]
        # 更新评分（移动平均）
        new_rating = (shared.rating * shared.endorsements + rating) / (shared.endorsements + 1)
        new_endorsements = shared.endorsements + 1
        # 3次以上认可标记为已验证
        verified = new_endorsements >= 3 and new_rating >= 3.0

        self._shared_skills[skill_name] = SharedSkill(
            skill=shared.skill,
            contributor_id=shared.contributor_id,
            contribution_time=shared.contribution_time,
            endorsements=new_endorsements,
            downloads=shared.downloads,
            rating=new_rating,
            verified=verified,
        )
        self._endorsement_history.append((skill_name, endorser_id, rating))
        return True

    def find_by_precondition(
        self,
        precondition: Tuple[int, ...]
    ) -> List[SharedSkill]:
        """按前置条件查找共享技能。"""
        names = self._index_by_precondition.get(precondition, [])
        return [self._shared_skills[n] for n in names if n in self._shared_skills]

    def find_best_shared_skill(
        self,
        precondition: Tuple[int, ...]
    ) -> Optional[SharedSkill]:
        """找到最适合的共享技能（质量最高）。"""
        candidates = self.find_by_precondition(precondition)
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.quality_score)

    def get_top_skills(self, n: int = 10) -> List[SharedSkill]:
        """获取质量最高的n个技能。"""
        sorted_skills = sorted(
            self._shared_skills.values(),
            key=lambda s: s.quality_score,
            reverse=True
        )
        return sorted_skills[:n]

    def get_verified_skills(self) -> List[SharedSkill]:
        """获取所有已验证的技能。"""
        return [s for s in self._shared_skills.values() if s.verified]

    def get_stats(self) -> dict:
        """获取共享库统计。"""
        total_downloads = sum(s.downloads for s in self._shared_skills.values())
        total_endorsements = sum(s.endorsements for s in self._shared_skills.values())
        verified_count = sum(1 for s in self._shared_skills.values() if s.verified)
        return {
            "total_skills": len(self._shared_skills),
            "verified_skills": verified_count,
            "total_downloads": total_downloads,
            "total_endorsements": total_endorsements,
            "avg_rating": float(np.mean([s.rating for s in self._shared_skills.values()])) if self._shared_skills else 0.0,
        }

    def sync_to_local(
        self,
        local_library: SkillLibrary,
        agent_id: str,
        max_skills: int = 10
    ) -> List[str]:
        """将共享库中的优质技能同步到Agent本地技能库。

        Args:
            local_library: Agent的本地技能库
            agent_id: Agent ID
            max_skills: 最大同步数量

        Returns:
            同步的技能名称列表
        """
        top_skills = self.get_top_skills(max_skills)
        synced = []
        for shared in top_skills:
            # 不下载自己的技能
            if shared.contributor_id == agent_id:
                continue
            local_library.add_skill(shared.skill)
            synced.append(shared.skill.name)
        return synced

    def collect_from_local(
        self,
        local_library: SkillLibrary,
        agent_id: str,
        min_success_rate: float = 0.5
    ) -> List[str]:
        """从Agent本地技能库收集优质技能到共享库。

        Args:
            local_library: Agent的本地技能库
            agent_id: Agent ID
            min_success_rate: 最低成功率阈值

        Returns:
            贡献的技能名称列表
        """
        contributed = []
        for skill in local_library.get_all_skills():
            if skill.success_rate >= min_success_rate and skill.total_count >= 3:
                shared_name = self.contribute(skill, agent_id)
                contributed.append(shared_name)
        return contributed

"""阶段85：陌生探索5轮无收益自动冻结。

本阶段实现探索任务的收益追踪与自动冻结机制。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import time

from .stage_084 import FileProtectionEngine


class FreezeReason(Enum):
    """探索任务冻结原因。"""
    NO_BENEFIT_5_ROUNDS = "no_benefit_5_rounds"
    LOW_BENEFIT_EXTENDED = "low_benefit_extended"
    RESOURCE_PRESSURE = "resource_pressure"


class UnfreezeCondition(Enum):
    """探索任务解冻条件。"""
    NEW_INFORMATION = "new_information"
    RESOURCE_SURPLUS = "resource_surplus"


@dataclass(frozen=True)
class ExploreBenefitRecord:
    """探索任务的收益记录。

    Attributes:
        task_id: 任务标识
        benefit_history: 最近5轮收益
        total_rounds: 总执行轮数
        total_benefit: 累计收益
        is_frozen: 是否已冻结
        freeze_reason: 冻结原因
        freeze_time: 冻结时间
        unfreeze_count: 解冻次数
    """
    task_id: str
    benefit_history: Tuple[float, ...] = ()
    total_rounds: int = 0
    total_benefit: float = 0.0
    is_frozen: bool = False
    freeze_reason: Optional[FreezeReason] = None
    freeze_time: Optional[float] = None
    unfreeze_count: int = 0


@dataclass(frozen=True)
class ExploreFreezerState:
    """探索冻结器的状态描述。

    Attributes:
        task_records: 所有探索任务的收益记录
        total_freeze_count: 总冻结次数
        total_unfreeze_count: 总解冻次数
        current_frozen_count: 当前冻结任务数
        avg_freeze_duration: 平均冻结时长
        unfreeze_success_rate: 解冻成功率
        resource_saved_ratio: 资源节省率
    """
    task_records: Dict[str, ExploreBenefitRecord] = field(default_factory=dict)
    total_freeze_count: int = 0
    total_unfreeze_count: int = 0
    current_frozen_count: int = 0
    avg_freeze_duration: float = 0.0
    unfreeze_success_rate: float = 0.0
    resource_saved_ratio: float = 0.0


@dataclass(frozen=True)
class ExploreFreezer:
    """陌生探索5轮无收益自动冻结器。

    组合阶段84的FileProtectionEngine，在文件保护基础上增加探索冻结状态。

    Attributes:
        file_protection: 阶段84的文件保护引擎（组合字段）
        freezer_state: 探索冻结器状态
        no_benefit_rounds: 无收益判定轮数（默认5）
        low_benefit_threshold: 低收益阈值（默认0.01）
        resource_surplus_cpu: 资源充裕CPU阈值（默认0.10）
        resource_surplus_mem: 资源充裕内存阈值（默认0.10）
        observe_rounds: 解冻观察期轮数（默认5）
    """
    file_protection: FileProtectionEngine
    freezer_state: ExploreFreezerState = field(default_factory=ExploreFreezerState)
    no_benefit_rounds: int = 5
    low_benefit_threshold: float = 0.01
    resource_surplus_cpu: float = 0.10
    resource_surplus_mem: float = 0.10
    observe_rounds: int = 5


class ExploreFreezerExecutor:
    """阶段85的核心接口：探索冻结执行器。

    实现收益追踪、无收益判定、冻结操作、解冻条件和统计日志。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 连续5轮无收益自动冻结
        - 解冻需满足新信息或资源充裕
        - 解冻后5轮观察期
    """

    def track_benefit(
        self,
        task_id: str,
        benefit: float,
        records: Dict[str, ExploreBenefitRecord]
    ) -> Dict[str, ExploreBenefitRecord]:
        """追踪探索任务收益（A85.1）。

        Args:
            task_id: 任务标识
            benefit: 本轮收益
            records: 当前收益记录字典

        Returns:
            更新后的收益记录字典
        """
        new_records = dict(records)
        if task_id in new_records:
            old = new_records[task_id]
            new_history = old.benefit_history + (benefit,)
            # 保持滑动窗口：只保留最近 no_benefit_rounds 轮
            max_window = 10  # 保留稍多于判定窗口，便于观察期管理
            if len(new_history) > max_window:
                new_history = new_history[-max_window:]
            new_records[task_id] = ExploreBenefitRecord(
                task_id=task_id,
                benefit_history=new_history,
                total_rounds=old.total_rounds + 1,
                total_benefit=old.total_benefit + benefit,
                is_frozen=old.is_frozen,
                freeze_reason=old.freeze_reason,
                freeze_time=old.freeze_time,
                unfreeze_count=old.unfreeze_count
            )
        else:
            new_records[task_id] = ExploreBenefitRecord(
                task_id=task_id,
                benefit_history=(benefit,),
                total_rounds=1,
                total_benefit=benefit,
                is_frozen=False,
                freeze_reason=None,
                freeze_time=None,
                unfreeze_count=0
            )
        return new_records

    def check_no_benefit(
        self,
        benefit_history: Tuple[float, ...],
        window: int = 5,
        low_threshold: float = 0.01
    ) -> Tuple[bool, bool]:
        """判定是否无收益（A85.2）。

        Args:
            benefit_history: 收益历史
            window: 判定窗口，必须 > 0
            low_threshold: 低收益阈值

        Returns:
            (是否无收益, 是否低收益)

        Raises:
            ValueError: 当window <= 0时
        """
        if window <= 0:
            raise ValueError(f"window必须 > 0，收到 {window}")
        if len(benefit_history) < window:
            return False, False
        recent = benefit_history[-window:]
        no_benefit = all(b <= 0.0 for b in recent)
        avg_benefit = sum(recent) / len(recent)
        low_benefit = avg_benefit < low_threshold
        return no_benefit, low_benefit

    def freeze_task(
        self,
        record: ExploreBenefitRecord,
        reason: FreezeReason = FreezeReason.NO_BENEFIT_5_ROUNDS
    ) -> ExploreBenefitRecord:
        """冻结探索任务（A85.3）。

        Args:
            record: 当前收益记录
            reason: 冻结原因

        Returns:
            冻结后的收益记录
        """
        return ExploreBenefitRecord(
            task_id=record.task_id,
            benefit_history=record.benefit_history,
            total_rounds=record.total_rounds,
            total_benefit=record.total_benefit,
            is_frozen=True,
            freeze_reason=reason,
            freeze_time=time.time(),
            unfreeze_count=record.unfreeze_count
        )

    def unfreeze_task(
        self,
        record: ExploreBenefitRecord,
        condition: UnfreezeCondition
    ) -> ExploreBenefitRecord:
        """解冻探索任务（A85.4）。

        解冻后重置收益历史，进入观察期。

        Args:
            record: 当前收益记录
            condition: 解冻条件

        Returns:
            解冻后的收益记录
        """
        return ExploreBenefitRecord(
            task_id=record.task_id,
            benefit_history=(),  # 重置收益历史
            total_rounds=0,       # 重置轮数
            total_benefit=0.0,    # 重置累计收益
            is_frozen=False,
            freeze_reason=None,
            freeze_time=None,
            unfreeze_count=record.unfreeze_count + 1
        )

    def check_unfreeze(
        self,
        new_info_available: bool,
        cpu_ratio: float,
        mem_ratio: float,
        resource_surplus_cpu: float = 0.10,
        resource_surplus_mem: float = 0.10
    ) -> Tuple[bool, Optional[UnfreezeCondition]]:
        """检查解冻条件（A85.4）。

        Args:
            new_info_available: 是否有新信息
            cpu_ratio: CPU占用率，范围[0,1]
            mem_ratio: 内存占用率，范围[0,1]
            resource_surplus_cpu: 资源充裕CPU阈值
            resource_surplus_mem: 资源充裕内存阈值

        Returns:
            (是否应解冻, 解冻条件)
        """
        if new_info_available:
            return True, UnfreezeCondition.NEW_INFORMATION
        if cpu_ratio < resource_surplus_cpu and mem_ratio < resource_surplus_mem:
            return True, UnfreezeCondition.RESOURCE_SURPLUS
        return False, None

    def compute_unfreeze_priority(
        self,
        frozen_tasks: Dict[str, ExploreBenefitRecord]
    ) -> List[str]:
        """计算解冻优先级（A85.4-priority）。

        按历史收益降序排列，历史收益高的优先解冻。

        Args:
            frozen_tasks: 冻结任务列表

        Returns:
            按优先级排序的任务ID列表
        """
        tasks = list(frozen_tasks.values())
        # 按历史总收益降序排列
        sorted_tasks = sorted(
            tasks,
            key=lambda r: r.total_benefit,
            reverse=True
        )
        return [t.task_id for t in sorted_tasks]

    def update_statistics(
        self,
        freezer_state: ExploreFreezerState
    ) -> ExploreFreezerState:
        """更新冻结统计（A85.5）。

        Args:
            freezer_state: 当前冻结器状态

        Returns:
            更新后的冻结器状态
        """
        frozen_count = sum(
            1 for r in freezer_state.task_records.values()
            if r.is_frozen
        )
        total = len(freezer_state.task_records)
        saved = frozen_count / max(total, 1)
        return ExploreFreezerState(
            task_records=freezer_state.task_records,
            total_freeze_count=freezer_state.total_freeze_count,
            total_unfreeze_count=freezer_state.total_unfreeze_count,
            current_frozen_count=frozen_count,
            avg_freeze_duration=freezer_state.avg_freeze_duration,
            unfreeze_success_rate=freezer_state.unfreeze_success_rate,
            resource_saved_ratio=saved
        )

    def step(
        self,
        freezer: ExploreFreezer,
        task_id: str,
        benefit: float,
        new_info_available: bool = False,
        cpu_ratio: float = 0.5,
        mem_ratio: float = 0.5
    ) -> ExploreFreezer:
        """执行一步探索冻结检查。

        1. 追踪收益
        2. 检查无收益判定
        3. 冻结无收益任务
        4. 检查解冻条件
        5. 更新统计

        Args:
            freezer: 当前探索冻结状态
            task_id: 当前任务ID
            benefit: 本轮收益
            new_info_available: 是否有新信息
            cpu_ratio: CPU占用率
            mem_ratio: 内存占用率

        Returns:
            更新后的探索冻结状态
        """
        # 1. 追踪收益
        new_records = self.track_benefit(
            task_id, benefit, freezer.freezer_state.task_records
        )

        # 2. 检查无收益判定并冻结
        record = new_records[task_id]
        no_benefit, low_benefit = self.check_no_benefit(
            record.benefit_history,
            freezer.no_benefit_rounds,
            freezer.low_benefit_threshold
        )

        total_freeze_count = freezer.freezer_state.total_freeze_count
        total_unfreeze_count = freezer.freezer_state.total_unfreeze_count

        if no_benefit and not record.is_frozen:
            # 冻结任务
            new_records[task_id] = self.freeze_task(
                record, FreezeReason.NO_BENEFIT_5_ROUNDS
            )
            total_freeze_count += 1

        # 3. 检查解冻条件（对所有已冻结任务）
        should_unfreeze, condition = self.check_unfreeze(
            new_info_available, cpu_ratio, mem_ratio,
            freezer.resource_surplus_cpu, freezer.resource_surplus_mem
        )

        if should_unfreeze:
            # 按优先级排序解冻
            frozen_tasks = {
                tid: r for tid, r in new_records.items() if r.is_frozen
            }
            priority_list = self.compute_unfreeze_priority(frozen_tasks)
            for tid in priority_list:
                if should_unfreeze:
                    new_records[tid] = self.unfreeze_task(
                        new_records[tid], condition
                    )
                    total_unfreeze_count += 1

        # 4. 更新统计
        new_state = ExploreFreezerState(
            task_records=new_records,
            total_freeze_count=total_freeze_count,
            total_unfreeze_count=total_unfreeze_count,
            current_frozen_count=freezer.freezer_state.current_frozen_count,
            avg_freeze_duration=freezer.freezer_state.avg_freeze_duration,
            unfreeze_success_rate=freezer.freezer_state.unfreeze_success_rate,
            resource_saved_ratio=freezer.freezer_state.resource_saved_ratio
        )
        new_state = self.update_statistics(new_state)

        return ExploreFreezer(
            file_protection=freezer.file_protection,
            freezer_state=new_state,
            no_benefit_rounds=freezer.no_benefit_rounds,
            low_benefit_threshold=freezer.low_benefit_threshold,
            resource_surplus_cpu=freezer.resource_surplus_cpu,
            resource_surplus_mem=freezer.resource_surplus_mem,
            observe_rounds=freezer.observe_rounds
        )

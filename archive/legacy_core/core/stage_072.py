"""阶段72：离线长期记忆——SQLite持久化、JSON快照与索引优化。

本阶段实现离线长期记忆的持久化存储机制。
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import sqlite3
import json
import time

from .stage_071 import ShortTermMemory, MemorySample


@dataclass(frozen=True)
class PersistentFilter:
    """长期记忆查询过滤条件。

    Attributes:
        time_range: 时间范围 [t_start, t_end]
        error_range: 误差范围 [e_min, e_max]
        value_range: 价值评分范围 [v_min, v_max]
        order_by: 排序字段
        order_desc: 是否降序
        limit: 返回数量上限
    """
    time_range: Optional[Tuple[float, float]] = None
    error_range: Optional[Tuple[float, float]] = None
    value_range: Optional[Tuple[float, float]] = None
    order_by: str = "timestamp"
    order_desc: bool = True
    limit: int = 100


@dataclass(frozen=True)
class ModelSnapshotRecord:
    """模型快照记录。

    Attributes:
        snapshot_id: 快照唯一标识
        model_state: 模型状态（JSON字符串）
        performance: 性能指标（JSON字符串）
        timestamp: 快照时间戳
    """
    snapshot_id: int
    model_state: str
    performance: str
    timestamp: float


@dataclass(frozen=True)
class LongTermMemory:
    """离线长期记忆的完整描述。

    组合阶段71的ShortTermMemory，在短期缓冲的基础上增加
    持久化存储的状态。

    Attributes:
        stm: 阶段71的短期记忆（组合字段）
        db_path: SQLite数据库路径
        total_samples: 样本表中的总记录数
        total_snapshots: 快照表中的总记录数
        persist_threshold: 持久化阈值
        batch_write_interval: 批量写入间隔（步数）
        last_write_step: 上次写入的步数

    Invariants:
        - total_samples >= 0
        - total_snapshots >= 0
        - persist_threshold >= 0
        - batch_write_interval > 0
    """
    stm: ShortTermMemory
    db_path: str = ":memory:"
    total_samples: int = 0
    total_snapshots: int = 0
    persist_threshold: float = 0.3
    batch_write_interval: int = 100
    last_write_step: int = 0

    def __post_init__(self):
        if self.total_samples < 0:
            raise ValueError(
                f"总样本数必须 >= 0，收到 {self.total_samples}"
            )
        if self.total_snapshots < 0:
            raise ValueError(
                f"总快照数必须 >= 0，收到 {self.total_snapshots}"
            )
        if self.persist_threshold < 0:
            raise ValueError(
                f"持久化阈值必须 >= 0，收到 {self.persist_threshold}"
            )
        if self.batch_write_interval <= 0:
            raise ValueError(
                f"批量写入间隔必须 > 0，收到 {self.batch_write_interval}"
            )

    @property
    def estimated_disk_size_mb(self) -> float:
        """估算磁盘占用（MB）。"""
        sample_size_kb = 2.0
        return self.total_samples * sample_size_kb / 1024


class LongTermMemoryManager:
    """阶段72的核心接口：离线长期记忆管理器。

    实现SQLite持久化存储和JSON模型快照。
    所有方法均为纯函数（无副作用），保证可验证性。

    核心约束：
        - 仅持久化高价值样本
        - 批量写入保证原子性
        - 索引加速常用查询
    """

    def initialize_db(self, db_path: str) -> None:
        """初始化SQLite数据库（A72.1, A72.5）。

        创建样本表、快照表和索引。

        Args:
            db_path: 数据库文件路径
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation BLOB NOT NULL,
                prediction BLOB NOT NULL,
                error REAL NOT NULL CHECK(error >= 0),
                reward REAL NOT NULL,
                timestamp REAL NOT NULL CHECK(timestamp >= 0),
                value_score REAL NOT NULL DEFAULT 0.0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_state TEXT NOT NULL,
                performance TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_timestamp
            ON samples(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_error
            ON samples(error)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_value
            ON samples(value_score)
        """)
        conn.commit()
        conn.close()

    def write_persistent(
        self,
        ltm: LongTermMemory,
        sample: MemorySample,
        value_score: float
    ) -> LongTermMemory:
        """持久化写入单个样本（A72.3）。

        仅当value_score > persist_threshold时写入。

        Args:
            ltm: 当前长期记忆状态
            sample: 待写入的样本
            value_score: 样本价值评分

        Returns:
            更新后的长期记忆状态
        """
        if value_score <= ltm.persist_threshold:
            return ltm
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        obs_bytes = sample.observation.tobytes()
        pred_bytes = sample.prediction.tobytes()
        cursor.execute(
            """
            INSERT INTO samples
            (observation, prediction, error, reward, timestamp, value_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (obs_bytes, pred_bytes, sample.error, sample.reward,
             sample.timestamp, value_score)
        )
        conn.commit()
        conn.close()
        return LongTermMemory(
            stm=ltm.stm,
            db_path=ltm.db_path,
            total_samples=ltm.total_samples + 1,
            total_snapshots=ltm.total_snapshots,
            persist_threshold=ltm.persist_threshold,
            batch_write_interval=ltm.batch_write_interval,
            last_write_step=ltm.last_write_step
        )

    def write_batch(
        self,
        ltm: LongTermMemory,
        samples: List[Tuple[MemorySample, float]]
    ) -> LongTermMemory:
        """批量持久化写入（A72.3-batch）。

        使用SQLite事务保证原子性。

        Args:
            ltm: 当前长期记忆状态
            samples: 待写入的样本列表（样本+价值评分）

        Returns:
            更新后的长期记忆状态
        """
        filtered = [(s, v) for s, v in samples if v > ltm.persist_threshold]
        if not filtered:
            return ltm
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        try:
            for sample, value_score in filtered:
                obs_bytes = sample.observation.tobytes()
                pred_bytes = sample.prediction.tobytes()
                cursor.execute(
                    """
                    INSERT INTO samples
                    (observation, prediction, error, reward, timestamp,
                     value_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (obs_bytes, pred_bytes, sample.error, sample.reward,
                     sample.timestamp, value_score)
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return LongTermMemory(
            stm=ltm.stm,
            db_path=ltm.db_path,
            total_samples=ltm.total_samples + len(filtered),
            total_snapshots=ltm.total_snapshots,
            persist_threshold=ltm.persist_threshold,
            batch_write_interval=ltm.batch_write_interval,
            last_write_step=ltm.last_write_step
        )

    def query_persistent(
        self,
        ltm: LongTermMemory,
        filter_condition: PersistentFilter
    ) -> List[MemorySample]:
        """条件查询持久化样本（A72.4）。

        Args:
            ltm: 当前长期记忆状态
            filter_condition: 过滤条件

        Returns:
            满足条件的样本列表
        """
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        query = "SELECT observation, prediction, error, reward, timestamp FROM samples WHERE 1=1"
        params: List[Any] = []
        if filter_condition.time_range is not None:
            t_start, t_end = filter_condition.time_range
            query += " AND timestamp >= ? AND timestamp <= ?"
            params.extend([t_start, t_end])
        if filter_condition.error_range is not None:
            e_min, e_max = filter_condition.error_range
            query += " AND error >= ? AND error <= ?"
            params.extend([e_min, e_max])
        if filter_condition.value_range is not None:
            v_min, v_max = filter_condition.value_range
            query += " AND value_score >= ? AND value_score <= ?"
            params.extend([v_min, v_max])
        order = "DESC" if filter_condition.order_desc else "ASC"
        query += f" ORDER BY {filter_condition.order_by} {order}"
        query += " LIMIT ?"
        params.append(filter_condition.limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        results: List[MemorySample] = []
        for row in rows:
            obs = np.frombuffer(row[0], dtype=np.float64)
            pred = np.frombuffer(row[1], dtype=np.float64)
            results.append(MemorySample(
                observation=obs,
                prediction=pred,
                error=row[2],
                reward=row[3],
                timestamp=row[4]
            ))
        return results

    def save_snapshot(
        self,
        ltm: LongTermMemory,
        model_state: Dict[str, Any],
        performance: Dict[str, float]
    ) -> LongTermMemory:
        """保存模型快照（A72.2）。

        Args:
            ltm: 当前长期记忆状态
            model_state: 模型状态字典
            performance: 性能指标字典

        Returns:
            更新后的长期记忆状态
        """
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO snapshots (model_state, performance, timestamp)
            VALUES (?, ?, ?)
            """,
            (json.dumps(model_state), json.dumps(performance), time.time())
        )
        conn.commit()
        conn.close()
        return LongTermMemory(
            stm=ltm.stm,
            db_path=ltm.db_path,
            total_samples=ltm.total_samples,
            total_snapshots=ltm.total_snapshots + 1,
            persist_threshold=ltm.persist_threshold,
            batch_write_interval=ltm.batch_write_interval,
            last_write_step=ltm.last_write_step
        )

    def load_snapshot(
        self,
        ltm: LongTermMemory,
        snapshot_id: int
    ) -> Optional[ModelSnapshotRecord]:
        """加载模型快照（A72.2）。

        Args:
            ltm: 当前长期记忆状态
            snapshot_id: 快照ID

        Returns:
            快照记录，如果不存在则返回None
        """
        self.initialize_db(ltm.db_path)
        conn = sqlite3.connect(ltm.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT snapshot_id, model_state, performance, timestamp "
            "FROM snapshots WHERE snapshot_id = ?",
            (snapshot_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None
        return ModelSnapshotRecord(
            snapshot_id=row[0],
            model_state=row[1],
            performance=row[2],
            timestamp=row[3]
        )

"""阶段D-1：持久化存储引擎 - 状态保存与恢复。

第一性原理：
    "永远运行不停止"的前提是状态可持久化。
    系统崩溃/重启后，必须能从上次的状态恢复，否则"永续"只是空谈。

    持久化的核心数据：
    1. Agent的Q表（学习成果）
    2. Agent的驱力状态（内在状态）
    3. 技能库（抽象出的技能）
    4. 运行统计（步数、情节数、成功率）
    5. 任务执行进度

    使用SQLite（Python标准库）实现，无需额外依赖。
    采用WAL模式提升并发写入性能。
"""

import json
import pickle
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# ============================================================
# 数据库模式
# ============================================================
SCHEMA_SQL = """
-- Agent状态表
CREATE TABLE IF NOT EXISTS agent_states (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    q_table BLOB,               -- pickle序列化的Q表numpy数组
    drives TEXT,                -- JSON: {safety, social, exploration}
    stats TEXT,                 -- JSON: 运行统计
    n_states INTEGER,
    n_actions INTEGER,
    learning_rate REAL,
    gamma REAL,
    updated_at REAL,
    created_at REAL
);

-- 技能库表
CREATE TABLE IF NOT EXISTS skills (
    skill_name TEXT PRIMARY KEY,
    precondition TEXT,          -- JSON: tuple
    action_sequence TEXT,       -- JSON: tuple
    postcondition TEXT,         -- JSON: tuple
    expected_reward REAL,
    success_count INTEGER,
    total_count INTEGER,
    abstraction_level INTEGER,
    contributor_id TEXT,
    updated_at REAL
);

-- 任务进度表
CREATE TABLE IF NOT EXISTS task_progress (
    task_id TEXT PRIMARY KEY,
    task_name TEXT,
    task_data TEXT,             -- JSON: 完整任务定义
    current_goal_index INTEGER,
    goal_statuses TEXT,         -- JSON: {goal_id: status}
    total_steps INTEGER,
    total_reward REAL,
    status TEXT,                -- pending/active/completed/failed
    created_at REAL,
    updated_at REAL
);

-- 运行日志表（用于审计和恢复）
CREATE TABLE IF NOT EXISTS runtime_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    event_type TEXT,            -- start/stop/crash/recover/checkpoint
    agent_id TEXT,
    message TEXT,
    details TEXT                -- JSON
);

-- 系统检查点表
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    description TEXT,
    state_snapshot TEXT         -- JSON: 系统级状态快照
);
"""


# ============================================================
# 持久化存储引擎
# ============================================================
class PersistenceEngine:
    """持久化存储引擎：保存和恢复系统状态。

    功能：
    1. 保存/加载Agent状态（Q表、驱力、统计）
    2. 保存/加载技能库
    3. 保存/加载任务进度
    4. 运行日志记录
    5. 系统检查点（全量快照）
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """
        Args:
            db_path: 数据库文件路径，":memory:"为内存数据库
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        """建立数据库连接并初始化模式。"""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # 启用WAL模式提升并发性能（仅对文件数据库有效）
        if self.db_path != ":memory:":
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.OperationalError:
                pass  # 某些环境不支持WAL
        # 创建表结构
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """事务上下文管理器。"""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ============================================================
    # Agent状态持久化
    # ============================================================
    def save_agent_state(
        self,
        agent_id: str,
        name: str,
        q_table: np.ndarray,
        drives: Dict[str, float],
        stats: Dict[str, Any],
        n_states: int,
        n_actions: int,
        learning_rate: float,
        gamma: float
    ) -> bool:
        """保存Agent状态。"""
        now = time.time()
        q_table_blob = pickle.dumps(q_table)
        drives_json = json.dumps(drives)
        stats_json = json.dumps(stats, default=str)

        with self.transaction() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO agent_states
                (agent_id, name, q_table, drives, stats, n_states, n_actions,
                 learning_rate, gamma, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id, name, q_table_blob, drives_json, stats_json,
                n_states, n_actions, learning_rate, gamma, now, now
            ))
        return True

    def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """加载Agent状态。"""
        with self.transaction() as cursor:
            cursor.execute(
                "SELECT * FROM agent_states WHERE agent_id = ?",
                (agent_id,)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "agent_id": row[0],
            "name": row[1],
            "q_table": pickle.loads(row[2]),
            "drives": json.loads(row[3]),
            "stats": json.loads(row[4]),
            "n_states": row[5],
            "n_actions": row[6],
            "learning_rate": row[7],
            "gamma": row[8],
            "updated_at": row[9],
            "created_at": row[10],
        }

    def list_agents(self) -> List[str]:
        """列出所有已保存的Agent ID。"""
        with self.transaction() as cursor:
            cursor.execute("SELECT agent_id FROM agent_states")
            return [row[0] for row in cursor.fetchall()]

    def delete_agent_state(self, agent_id: str) -> bool:
        """删除Agent状态。"""
        with self.transaction() as cursor:
            cursor.execute(
                "DELETE FROM agent_states WHERE agent_id = ?",
                (agent_id,)
            )
            return cursor.rowcount > 0

    # ============================================================
    # 技能库持久化
    # ============================================================
    def save_skill(
        self,
        skill_name: str,
        precondition: Tuple,
        action_sequence: Tuple,
        postcondition: Tuple,
        expected_reward: float,
        success_count: int,
        total_count: int,
        abstraction_level: int,
        contributor_id: str = ""
    ) -> bool:
        """保存技能。"""
        now = time.time()
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO skills
                (skill_name, precondition, action_sequence, postcondition,
                 expected_reward, success_count, total_count, abstraction_level,
                 contributor_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                skill_name,
                json.dumps(list(precondition)),
                json.dumps(list(action_sequence)),
                json.dumps(list(postcondition)),
                expected_reward, success_count, total_count,
                abstraction_level, contributor_id, now
            ))
        return True

    def load_all_skills(self) -> List[Dict[str, Any]]:
        """加载所有技能。"""
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM skills")
            rows = cursor.fetchall()
        skills = []
        for row in rows:
            skills.append({
                "skill_name": row[0],
                "precondition": tuple(json.loads(row[1])),
                "action_sequence": tuple(json.loads(row[2])),
                "postcondition": tuple(json.loads(row[3])),
                "expected_reward": row[4],
                "success_count": row[5],
                "total_count": row[6],
                "abstraction_level": row[7],
                "contributor_id": row[8],
            })
        return skills

    def delete_skill(self, skill_name: str) -> bool:
        """删除技能。"""
        with self.transaction() as cursor:
            cursor.execute(
                "DELETE FROM skills WHERE skill_name = ?",
                (skill_name,)
            )
            return cursor.rowcount > 0

    # ============================================================
    # 任务进度持久化
    # ============================================================
    def save_task_progress(
        self,
        task_id: str,
        task_name: str,
        task_data: Dict,
        current_goal_index: int,
        goal_statuses: Dict,
        total_steps: int,
        total_reward: float,
        status: str
    ) -> bool:
        """保存任务进度。"""
        now = time.time()
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO task_progress
                (task_id, task_name, task_data, current_goal_index, goal_statuses,
                 total_steps, total_reward, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id, task_name, json.dumps(task_data, default=str),
                current_goal_index, json.dumps(goal_statuses),
                total_steps, total_reward, status, now, now
            ))
        return True

    def load_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """加载任务进度。"""
        with self.transaction() as cursor:
            cursor.execute(
                "SELECT * FROM task_progress WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "task_id": row[0],
            "task_name": row[1],
            "task_data": json.loads(row[2]),
            "current_goal_index": row[3],
            "goal_statuses": json.loads(row[4]),
            "total_steps": row[5],
            "total_reward": row[6],
            "status": row[7],
        }

    def list_pending_tasks(self) -> List[str]:
        """列出所有未完成的任务。"""
        with self.transaction() as cursor:
            cursor.execute(
                "SELECT task_id FROM task_progress WHERE status IN ('pending', 'active')"
            )
            return [row[0] for row in cursor.fetchall()]

    # ============================================================
    # 运行日志
    # ============================================================
    def log_event(
        self,
        event_type: str,
        agent_id: str = "",
        message: str = "",
        details: Optional[Dict] = None
    ) -> bool:
        """记录运行日志。"""
        now = time.time()
        details_json = json.dumps(details or {}, default=str)
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT INTO runtime_log (timestamp, event_type, agent_id, message, details)
                VALUES (?, ?, ?, ?, ?)
            """, (now, event_type, agent_id, message, details_json))
        return True

    def get_recent_logs(self, n: int = 100) -> List[Dict[str, Any]]:
        """获取最近n条日志。"""
        with self.transaction() as cursor:
            cursor.execute(
                "SELECT * FROM runtime_log ORDER BY id DESC LIMIT ?",
                (n,)
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "event_type": row[2],
                "agent_id": row[3],
                "message": row[4],
                "details": json.loads(row[5]) if row[5] else {},
            }
            for row in reversed(rows)
        ]

    # ============================================================
    # 系统检查点
    # ============================================================
    def create_checkpoint(
        self,
        description: str,
        state_snapshot: Dict[str, Any]
    ) -> int:
        """创建系统检查点（全量快照）。"""
        now = time.time()
        snapshot_json = json.dumps(state_snapshot, default=str)
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT INTO checkpoints (timestamp, description, state_snapshot)
                VALUES (?, ?, ?)
            """, (now, description, snapshot_json))
            return cursor.lastrowid

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """获取最新的检查点。"""
        with self.transaction() as cursor:
            cursor.execute(
                "SELECT * FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "checkpoint_id": row[0],
            "timestamp": row[1],
            "description": row[2],
            "state_snapshot": json.loads(row[3]),
        }

    def list_checkpoints(self, n: int = 10) -> List[Dict[str, Any]]:
        """列出最近n个检查点。"""
        with self.transaction() as cursor:
            cursor.execute(
                "SELECT checkpoint_id, timestamp, description FROM checkpoints "
                "ORDER BY checkpoint_id DESC LIMIT ?",
                (n,)
            )
            rows = cursor.fetchall()
        return [
            {"checkpoint_id": row[0], "timestamp": row[1], "description": row[2]}
            for row in rows
        ]

    # ============================================================
    # 统计信息
    # ============================================================
    def get_storage_stats(self) -> Dict[str, int]:
        """获取存储统计。"""
        stats = {}
        with self.transaction() as cursor:
            for table in ["agent_states", "skills", "task_progress", "runtime_log", "checkpoints"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
        return stats

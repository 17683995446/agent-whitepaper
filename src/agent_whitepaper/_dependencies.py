"""阶段间依赖关系声明（单一事实来源）。

本模块声明各阶段（stage_a ~ stage_k 及 demos）之间的直接依赖关系。
依赖关系以邻接表形式表示：键为阶段名，值为该阶段直接依赖的阶段列表。

设计原则（第一性原理）：
- 纯数据声明，不导入任何阶段模块（避免循环导入和副作用）
- 仅记录直接依赖（传递依赖由图遍历推导）
- 测试模块 test_stage_n.py 会扫描实际 import 语句并验证与此声明一致
- 依赖方向：被依赖者在列表中，依赖者在键位置

依赖层次（从底到顶）：
    stage_a (基础层，无依赖)
    ├── stage_b → {a}
    ├── stage_i → {a}
    ├── stage_j → {} (独立安全层)
    stage_c → {b}
    ├── stage_d → {a, b}
    ├── stage_e → {a, b}
    stage_f → {a, b, e}
    stage_g → {a, b, c, d, e, f}  (全系统集成层)
    stage_h → {d, g}
    stage_k → {g, h}
    demos → {a, b, c}

公共API：
    STAGE_DEPENDENCIES: 直接依赖邻接表
    TRANSITIVE_DEPENDENCIES: 传递依赖（运行时计算）
    get_direct_dependencies(stage): 获取直接依赖
    get_transitive_dependencies(stage): 获取传递依赖
    topological_order(): 拓扑排序
    has_cycle(): 检测是否存在循环依赖
"""

from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# 直接依赖声明（单一事实来源）
# ---------------------------------------------------------------------------

STAGE_DEPENDENCIES: Dict[str, List[str]] = {
    "stage_a": [],
    "stage_b": ["stage_a"],
    "stage_c": ["stage_b"],
    "stage_d": ["stage_a", "stage_b"],
    "stage_e": ["stage_a", "stage_b"],
    "stage_f": ["stage_a", "stage_b", "stage_e"],
    "stage_g": [
        "stage_a",
        "stage_b",
        "stage_c",
        "stage_d",
        "stage_e",
        "stage_f",
    ],
    "stage_h": ["stage_d", "stage_g"],
    "stage_i": ["stage_a"],
    "stage_j": [],
    "stage_k": ["stage_g", "stage_h"],
    "demos": ["stage_a", "stage_b", "stage_c"],
}


# ---------------------------------------------------------------------------
# 图遍历工具函数
# ---------------------------------------------------------------------------

def get_direct_dependencies(stage: str) -> List[str]:
    """获取阶段的直接依赖列表。"""
    if stage not in STAGE_DEPENDENCIES:
        raise KeyError(f"未知阶段: {stage}")
    return list(STAGE_DEPENDENCIES[stage])


def get_transitive_dependencies(stage: str) -> Set[str]:
    """获取阶段的所有传递依赖（不含自身）。

    使用深度优先搜索遍历依赖图。
    """
    if stage not in STAGE_DEPENDENCIES:
        raise KeyError(f"未知阶段: {stage}")

    visited: Set[str] = set()
    stack = list(STAGE_DEPENDENCIES[stage])
    while stack:
        dep = stack.pop()
        if dep in visited:
            continue
        visited.add(dep)
        stack.extend(STAGE_DEPENDENCIES.get(dep, []))
    return visited


def topological_order() -> List[str]:
    """返回阶段的拓扑排序（被依赖者在前）。

    若存在循环依赖则抛出 ValueError。
    """
    visited: Set[str] = set()
    in_stack: Set[str] = set()
    result: List[str] = []

    def _visit(node: str) -> None:
        if node in visited:
            return
        if node in in_stack:
            raise ValueError(f"检测到循环依赖，涉及: {node}")
        in_stack.add(node)
        for dep in STAGE_DEPENDENCIES.get(node, []):
            _visit(dep)
        in_stack.discard(node)
        visited.add(node)
        result.append(node)

    for stage in STAGE_DEPENDENCIES:
        _visit(stage)
    return result


def has_cycle() -> bool:
    """检测依赖图是否存在循环依赖。"""
    try:
        topological_order()
        return False
    except ValueError:
        return True


def get_dependents(stage: str) -> List[str]:
    """获取直接依赖于给定阶段的所有阶段（反向依赖）。"""
    return [
        s for s, deps in STAGE_DEPENDENCIES.items() if stage in deps
    ]


# ---------------------------------------------------------------------------
# 预计算的传递依赖（便于直接引用）
# ---------------------------------------------------------------------------

TRANSITIVE_DEPENDENCIES: Dict[str, Set[str]] = {
    stage: get_transitive_dependencies(stage)
    for stage in STAGE_DEPENDENCIES
}


__all__ = [
    "STAGE_DEPENDENCIES",
    "TRANSITIVE_DEPENDENCIES",
    "get_direct_dependencies",
    "get_transitive_dependencies",
    "topological_order",
    "has_cycle",
    "get_dependents",
]

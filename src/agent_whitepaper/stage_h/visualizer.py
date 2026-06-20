"""阶段H-7：可视化报告 - 系统的"仪表板"。

第一性原理：
    "A picture is worth a thousand words"——图表胜过数字。
    纯文本报告难以直观理解趋势。可视化报告通过：
    1. 图表：成功率趋势、技能增长、Agent对比
    2. 摘要：关键指标一目了然
    3. 详情：可展开的详细数据
    4. 自包含：单HTML文件，无外部依赖

    可视化让"数据"变为"洞察"。

用法：
    from agent_whitepaper.stage_h.visualizer import Visualizer

    viz = Visualizer()
    report = orch.get_metrics()
    html = viz.generate_html(report)
    viz.save(html, "report.html")
"""

import html
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent_whitepaper._logging import get_logger

logger = get_logger(__name__)


# ============================================================
# HTML 模板
# ============================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Whitepaper 运行报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5; color: #333; line-height: 1.6; padding: 20px;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #2c3e50; margin-bottom: 10px; }}
h2 {{ color: #34495e; margin: 20px 0 10px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
.subtitle {{ color: #7f8c8d; margin-bottom: 20px; }}
.summary-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px; margin: 20px 0;
}}
.card {{
    background: white; padding: 20px; border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
}}
.card .value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
.card .label {{ color: #7f8c8d; font-size: 0.9em; margin-top: 5px; }}
.chart-container {{
    background: white; padding: 20px; border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0;
}}
.bar-chart {{ display: flex; align-items: flex-end; height: 200px; gap: 2px; padding: 10px; }}
.bar {{
    flex: 1; background: #3498db; min-height: 2px;
    transition: height 0.3s; position: relative;
}}
.bar:hover {{ background: #2980b9; }}
.bar .tooltip {{
    position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
    background: #2c3e50; color: white; padding: 4px 8px; border-radius: 4px;
    font-size: 0.8em; white-space: nowrap; opacity: 0; transition: opacity 0.3s;
}}
.bar:hover .tooltip {{ opacity: 1; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; }}
th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #3498db; color: white; }}
tr:hover {{ background: #f5f5f5; }}
.status-badge {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.85em; font-weight: bold;
}}
.status-running {{ background: #e8f5e9; color: #2e7d32; }}
.status-stopped {{ background: #ffebee; color: #c62828; }}
.status-recovering {{ background: #fff3e0; color: #e65100; }}
.trend-up {{ color: #2e7d32; }}
.trend-down {{ color: #c62828; }}
.trend-stable {{ color: #7f8c8d; }}
.footer {{ text-align: center; color: #95a5a6; margin-top: 30px; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">
<h1>Agent Whitepaper 运行报告</h1>
<p class="subtitle">生成时间: {generated_at}</p>

<h2>系统概览</h2>
<div class="summary-grid">
{summary_cards}
</div>

<h2>关键指标</h2>
<div class="chart-container">
    <h3>成功率趋势</h3>
    <div class="bar-chart">{success_rate_bars}</div>
</div>

<div class="chart-container">
    <h3>技能数趋势</h3>
    <div class="bar-chart">{skill_count_bars}</div>
</div>

<h2>Agent 详情</h2>
{agent_table}

<h2>累计统计</h2>
{cumulative_stats}

<div class="footer">
    由 Agent Whitepaper 可视化报告生成器创建
</div>
</div>
</body>
</html>"""


# ============================================================
# 可视化报告生成器
# ============================================================
class Visualizer:
    """可视化报告生成器。

    第一性原理：
        "A picture is worth a thousand words"——图表胜过数字。
        将指标报告转换为自包含的HTML可视化报告。

    用法：
        viz = Visualizer()
        report = orch.get_metrics()
        html = viz.generate_html(report)
        viz.save(html, "report.html")
    """

    def generate_html(self, report: Dict[str, Any]) -> str:
        """从指标报告生成HTML可视化报告。

        Args:
            report: get_metrics() 返回的指标报告字典

        Returns:
            自包含的HTML字符串
        """
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_cards = self._render_summary_cards(report)
        success_rate_bars = self._render_trend_bars(
            report, "success_rate_summary", "avg_success_rate"
        )
        skill_count_bars = self._render_trend_bars(
            report, "skill_count_summary", "skill_count"
        )
        agent_table = self._render_agent_table(report)
        cumulative_stats = self._render_cumulative_stats(report)

        return HTML_TEMPLATE.format(
            generated_at=html.escape(generated_at),
            summary_cards=summary_cards,
            success_rate_bars=success_rate_bars,
            skill_count_bars=skill_count_bars,
            agent_table=agent_table,
            cumulative_stats=cumulative_stats,
        )

    def save(self, html_content: str, path: str) -> None:
        """保存HTML报告到文件。

        Args:
            html_content: HTML内容
            path: 输出文件路径
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def generate_and_save(
        self, report: Dict[str, Any], path: str
    ) -> str:
        """生成并保存HTML报告。

        Args:
            report: 指标报告字典
            path: 输出文件路径

        Returns:
            HTML内容
        """
        html_content = self.generate_html(report)
        self.save(html_content, path)
        return html_content

    # --------------------------------------------------------
    # 渲染组件
    # --------------------------------------------------------
    def _render_summary_cards(self, report: Dict[str, Any]) -> str:
        """渲染摘要卡片。"""
        cards = []
        snap = report.get("latest_snapshot") or {}

        cards.append(self._card(
            report.get("iteration", 0), "迭代次数"
        ))
        cards.append(self._card(
            report.get("agent_count", 0), "Agent数"
        ))
        cards.append(self._card(
            report.get("skill_count", 0), "技能数"
        ))
        cards.append(self._card(
            report.get("tasks_completed", 0), "完成任务"
        ))
        if snap:
            cards.append(self._card(
                f"{snap.get('avg_success_rate', 0):.1%}", "成功率"
            ))
            cards.append(self._card(
                f"{snap.get('health_score', 0):.2f}", "健康分"
            ))

        status = report.get("status", "unknown")
        status_class = f"status-{status}"
        cards.append(
            f'<div class="card"><div class="value">'
            f'<span class="status-badge {status_class}">{html.escape(status)}</span>'
            f'</div><div class="label">系统状态</div></div>'
        )

        return "\n".join(cards)

    def _card(self, value: Any, label: str) -> str:
        """渲染单个卡片。"""
        return (
            f'<div class="card">'
            f'<div class="value">{html.escape(str(value))}</div>'
            f'<div class="label">{html.escape(label)}</div>'
            f'</div>'
        )

    def _render_trend_bars(
        self, report: Dict[str, Any], summary_key: str, metric_key: str
    ) -> str:
        """渲染趋势柱状图。"""
        summary = report.get(summary_key) or {}
        if not summary:
            return '<p style="color:#999;text-align:center;">暂无数据</p>'

        values = self._extract_trend_values(summary, metric_key)
        if not values:
            return '<p style="color:#999;text-align:center;">暂无数据</p>'

        max_val = max(values) if values else 1
        if max_val == 0:
            max_val = 1

        bars = []
        for i, v in enumerate(values):
            height_pct = int(100 * v / max_val) if max_val > 0 else 0
            tooltip = f"#{i + 1}: {v:.4f}" if isinstance(v, float) else f"#{i + 1}: {v}"
            bars.append(
                f'<div class="bar" style="height:{height_pct}%">'
                f'<span class="tooltip">{tooltip}</span>'
                f'</div>'
            )
        return "\n".join(bars)

    def _extract_trend_values(
        self, summary: Dict[str, Any], metric_key: str
    ) -> List[float]:
        """从摘要中提取趋势值。"""
        # 尝试从 summary 中提取历史值
        if "values" in summary:
            return list(summary["values"])
        if "history" in summary:
            return list(summary["history"])
        # 如果只有 first/last/min/max/mean，构造简化趋势
        if "first" in summary and "last" in summary:
            first = summary.get("first", 0)
            last = summary.get("last", 0)
            mean = summary.get("mean", (first + last) / 2)
            return [first, mean, last]
        return []

    def _render_agent_table(self, report: Dict[str, Any]) -> str:
        """渲染Agent详情表格。"""
        # report 中没有 agent_summaries，需要从外部传入
        # 这里渲染一个占位表格
        agent_count = report.get("agent_count", 0)
        if agent_count == 0:
            return '<p style="color:#999;">无Agent数据</p>'

        rows = []
        rows.append(
            "<table><thead><tr>"
            "<th>Agent ID</th><th>成功率</th><th>情节数</th>"
            "<th>Q表条目</th><th>平均奖励</th>"
            "</tr></thead><tbody>"
        )
        # 占位行（实际数据需从 get_agent_summaries 获取）
        for i in range(agent_count):
            rows.append(
                f"<tr><td>agent_{i}</td>"
                f"<td>-</td><td>-</td><td>-</td><td>-</td></tr>"
            )
        rows.append("</tbody></table>")
        return "\n".join(rows)

    def _render_cumulative_stats(self, report: Dict[str, Any]) -> str:
        """渲染累计统计。"""
        cum = report.get("cumulative") or {}
        if not cum:
            return '<p style="color:#999;">无累计统计</p>'

        rows = ['<table><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>']
        for key, value in cum.items():
            label = self._translate_cumulative_key(key)
            rows.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td>{html.escape(str(value))}</td></tr>"
            )
        rows.append("</tbody></table>")
        return "\n".join(rows)

    def _translate_cumulative_key(self, key: str) -> str:
        """翻译累计统计键名为中文。"""
        mapping = {
            "distillation_count": "蒸馏次数",
            "migration_count": "迁移次数",
            "injected_states": "注入状态数",
            "recovery_count": "恢复次数",
        }
        return mapping.get(key, key)


# ============================================================
# 便捷函数
# ============================================================
def generate_report(
    report: Dict[str, Any], path: Optional[str] = None
) -> str:
    """生成可视化报告。

    Args:
        report: 指标报告字典
        path: 输出文件路径（None=不保存）

    Returns:
        HTML内容

    Example:
        >>> from agent_whitepaper.stage_h.visualizer import generate_report
        >>> report = orch.get_metrics()
        >>> generate_report(report, "report.html")
    """
    viz = Visualizer()
    if path:
        return viz.generate_and_save(report, path)
    return viz.generate_html(report)


def generate_report_from_orchestrator(
    orchestrator: Any, path: Optional[str] = None
) -> str:
    """从编排器生成可视化报告。

    Args:
        orchestrator: SystemOrchestrator 实例
        path: 输出文件路径（None=不保存）

    Returns:
        HTML内容
    """
    report = orchestrator.get_metrics()
    # 补充 agent_summaries
    try:
        report["_agent_summaries"] = orchestrator.get_agent_summaries()
    except Exception:  # 防火墙层：获取摘要失败不影响报告生成
        logger.debug("获取 agent_summaries 失败，跳过该字段")
    return generate_report(report, path)

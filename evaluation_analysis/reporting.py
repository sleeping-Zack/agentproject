"""Human-readable rendering for evaluation-analysis reports.

The renderer intentionally uses an allow-list of aggregate and diagnostic fields.
It never serializes arbitrary report content, prompts, answers, or traces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_DECISION_LABELS = {
    "blocked": "阻断",
    "diagnostic_only": "仅诊断",
    "keep_baseline": "保留基线",
    "eligible_for_human_approval": "可提交人工审批",
}


def _text(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _number(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    rendered = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return rendered if rendered not in {"-0", ""} else "0"


def _list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _join_codes(value: Any) -> str:
    return ", ".join(_text(item) for item in _list(value)) or "-"


def _metric_row(label: str, metric: Any) -> str:
    metric = _mapping(metric)
    interval = _mapping(metric.get("confidence_interval", metric.get("ci")))
    ci = "-"
    if "lower" in interval and "upper" in interval:
        ci = f"[{_number(interval['lower'])}, {_number(interval['upper'])}]"
    p_value = metric.get("p_value", metric.get("two_sided_p_value"))
    return (
        f"| {label} | {_number(metric.get('baseline'))} | "
        f"{_number(metric.get('candidate'))} | {_number(metric.get('delta'))} | "
        f"{ci} | {_number(p_value)} |"
    )


def _gate_row(layer: str, gate: Any) -> str:
    gate = _mapping(gate)
    details = gate.get("reasons", gate.get("failures", gate.get("warnings", [])))
    return f"| {layer} | {_text(gate.get('status'))} | {_join_codes(details)} |"


def _render_bad_cases(rows: Any, limit: int = 10) -> list[str]:
    lines = ["| Case ID | 优先级 | 错误类型 | 根因 | 责任模块 |", "|---|---|---|---|---|"]
    for item in _list(rows)[:limit]:
        item = _mapping(item)
        error_types = item.get("error_types", item.get("failure_types", item.get("reasons")))
        lines.append(
            "| {case_id} | {priority} | {errors} | {cause} | {owner} |".format(
                case_id=_text(item.get("case_id")),
                priority=_text(item.get("priority")),
                errors=_join_codes(error_types),
                cause=_text(item.get("root_cause", item.get("root_cause_code"))),
                owner=_text(item.get("owner_module")),
            )
        )
    if len(lines) == 2:
        lines.append("| - | - | - | - | - |")
    return lines


def _summary_rows(summary: Any) -> list[Mapping[str, Any]]:
    if isinstance(summary, Mapping):
        rows: list[Mapping[str, Any]] = []
        for code, details in summary.items():
            item = dict(_mapping(details))
            item.setdefault("root_cause", code)
            rows.append(item)
        return rows
    return [_mapping(item) for item in _list(summary)]


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a safe, aggregate-only Markdown report."""

    experiment = _mapping(report.get("experiment"))
    comparability = _mapping(report.get("comparability"))
    evaluator_gate = _mapping(report.get("evaluator_gate"))
    evidence = _mapping(report.get("evidence"))
    quality = _mapping(report.get("quality_comparison"))
    performance = _mapping(report.get("performance"))
    safety = _mapping(report.get("safety"))
    decision = _mapping(report.get("release_decision"))
    safety_detail = (
        f"- 安全失败 Case：{_join_codes(safety.get('new_failure_case_ids'))}"
        if experiment.get("mode") == "diagnostic"
        else "- 冻结 test 安全结果仅展示聚合计数。"
    )

    lines = [
        "# Agent 评测实验报告",
        "",
        f"- 报告 ID：{_text(report.get('report_id'))}",
        f"- 生成时间：{_text(report.get('generated_at'))}",
        f"- 分析版本：{_text(report.get('analysis_version'))}",
        "",
        "## 1. 实验目标",
        "",
        f"- 实验 ID：{_text(experiment.get('experiment_id'))}",
        f"- 模式：{_text(experiment.get('mode'))}",
        f"- 假设：{_text(experiment.get('hypothesis'))}",
        f"- 变更：{_text(experiment.get('change'))}",
        f"- 主指标：{_text(experiment.get('primary_metric'))}",
        "",
        "## 2. 可比性",
        "",
        f"- 状态：{_text(comparability.get('status'))}",
        f"- 阻断原因：{_join_codes(comparability.get('reasons'))}",
        f"- 警告：{_join_codes(comparability.get('warnings'))}",
        "",
        "## 3. 三层门禁",
        "",
        "| 层级 | 状态 | 原因或警告 |",
        "|---|---|---|",
        _gate_row("输入可比性", comparability),
        _gate_row("评测器可信度", evaluator_gate),
        _gate_row("发布证据充分性", evidence),
        "",
        f"配对样本数：{_number(evidence.get('pair_count'), 0)}",
        "",
        "## 4. KPI 与置信区间",
        "",
        "| 指标 | 基线 | 候选 | Delta | 置信区间 | p-value |",
        "|---|---:|---:|---:|---:|---:|",
        _metric_row("通过率", quality.get("pass_rate")),
        _metric_row("综合得分", quality.get("overall_score")),
    ]

    dimensions = _mapping(quality.get("dimensions"))
    for name, metric in dimensions.items():
        lines.append(_metric_row(f"维度：{_text(name)}", metric))

    lines.extend(
        [
            _metric_row("P95 延迟", performance.get("p95_latency_ms")),
            _metric_row("平均成本", performance.get("average_cost")),
            _metric_row("平均 Token", performance.get("average_tokens")),
            "",
            "## 5. 安全与切片",
            "",
            f"- 新增安全失败：{_number(safety.get('new_failure_count'), 0)}",
            f"- 新增一票否决：{_number(safety.get('new_veto_count'), 0)}",
            f"- 新增 P0：{_number(safety.get('new_p0_count'), 0)}",
            f"- 新增 L3 通过退化：{_number(safety.get('new_l3_failure_count'), 0)}",
            safety_detail,
            "",
            "| 切片 | 样本数 | 通过率 Delta | 综合得分 Delta | 是否进入门禁 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in _list(report.get("slices")):
        item = _mapping(item)
        lines.append(
            "| {name} | {count} | {pass_delta} | {score_delta} | {gated} |".format(
                name=_text(item.get("name", item.get("slice"))),
                count=_number(item.get("pair_count", item.get("count")), 0),
                pass_delta=_number(item.get("pass_rate_delta")),
                score_delta=_number(item.get("overall_score_delta")),
                gated=_text(item.get("gated", item.get("eligible_for_gate"))),
            )
        )
    if len(_list(report.get("slices"))) == 0:
        lines.append("| - | - | - | - | - |")

    lines.extend(["", "## 6. Top Bad Cases", ""])
    lines.extend(_render_bad_cases(report.get("bad_cases")))

    lines.extend(
        [
            "",
            "## 7. 根因归因",
            "",
            "| 根因 | 数量 | 责任模块 | Case IDs |",
            "|---|---:|---|---|",
        ]
    )
    root_rows = _summary_rows(report.get("root_cause_summary"))
    for item in root_rows:
        lines.append(
            f"| {_text(item.get('root_cause', item.get('code')))} | "
            f"{_number(item.get('count'), 0)} | {_text(item.get('owner_module'))} | "
            f"{_join_codes(item.get('case_ids'))} |"
        )
    if not root_rows:
        lines.append("| - | - | - | - |")

    lines.extend(["", "## 8. 迭代建议与回归候选", ""])
    recommendations = _list(report.get("recommendations"))
    if recommendations:
        for item in recommendations:
            item = _mapping(item)
            lines.append(
                f"- [{_text(item.get('priority'))}] {_text(item.get('recommendation_id'))}："
                f"{_text(item.get('action'))}（owner={_text(item.get('owner_module'))}，"
                f"影响 {_number(item.get('affected_case_count'), 0)} 个 case）"
            )
    else:
        lines.append("- 无自动生成的迭代建议。")

    candidates = report.get("regression_candidates")
    proposed = _list(_mapping(candidates).get("proposed")) if isinstance(candidates, Mapping) else _list(candidates)
    lines.append(f"- 待人工审核的回归候选：{len(proposed)} 个。")

    lines.extend(
        [
            "",
            "## 9. 发布决策",
            "",
            f"- 状态：{_DECISION_LABELS.get(decision.get('status'), _text(decision.get('status')))}",
            f"- 原因：{_join_codes(decision.get('reasons'))}",
            "- 是否需要人工审批："
            + ("是" if decision.get("requires_human_approval") is True else "否"),
            "",
            "## 10. 限制",
            "",
        ]
    )
    limitations = _list(report.get("limitations"))
    lines.extend(f"- {_text(item)}" for item in limitations)
    if not limitations:
        lines.append("- 未记录额外限制。")
    lines.append("")
    return "\n".join(lines)


def write_markdown(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Render ``report`` and write it as UTF-8 Markdown."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path

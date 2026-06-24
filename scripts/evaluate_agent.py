"""端到端 Agent 评测：跑 evals/agent_golden.jsonl，输出通过率与工具命中率。

评测维度：
    1. 工具命中率：Agent 是否调用了 expected_tools 中列出的工具（按名匹配）
       工具调用顺序/参数完全一致是 strict 命中，仅名字命中是 soft 命中
    2. 关键词命中率：最终回答是否包含 expected_keywords
    3. 拒绝率：expected_rejection=true 的 case 是否被安全模块挡下
    4. 总体通过：工具 ≥ 0.5 且关键词 ≥ 0.5 视为 PASS（阈值可调）

调用模式：
    --quiet            只输出最终 JSON 汇总，便于 prompt_diff.py 抓取
    --smoke            只跑前 N 条，CI 用（CI_SMOKE_LIMIT 环境变量也可控）
    --report path.json 写一份机读评测报告

为什么不直接接 LLM-as-judge：判分依据可解释 + 不消耗额外配额。LLM judge 留给
线上质量复核脚本 evaluate_judge.py。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from observability.tracing import trace_recorder


@dataclass
class CaseResult:
    id: str
    passed: bool
    tool_recall: float
    keyword_recall: float
    rejected: Optional[bool]
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)


def load_golden(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _tools_actually_called(request_id: str) -> List[str]:
    try:
        events = trace_recorder.export_trace(request_id)["events"]
    except KeyError:
        return []
    return [e["name"] for e in events if e["category"] == "tool"]


def _evaluate_case(agent, case: Dict) -> CaseResult:
    expected_tools = [t.get("name") for t in case.get("expected_tools", [])]
    expected_keywords = case.get("expected_keywords", [])
    expected_rejection = case.get("expected_rejection", False)

    from uuid import uuid4
    request_id = str(uuid4())

    try:
        if case.get("turns"):
            # 多轮：先按 turns 喂入历史，再用最后一条 user 触发
            from agent.memory import ConversationMemory
            for turn in case["turns"][:-1]:
                agent.memory.add_message(
                    case["id"], turn["role"], turn["content"], tenant_id="eval"
                )
            query = case["turns"][-1]["content"]
        else:
            query = case["query"]

        chunks = list(agent.execute_stream(
            query, session_id=case["id"], request_id=request_id, tenant_id="eval"
        ))
        answer = "".join(chunks)
    except Exception as exc:
        return CaseResult(
            id=case["id"], passed=False, tool_recall=0.0, keyword_recall=0.0,
            rejected=None, error=str(exc),
            detail={"trace": traceback.format_exc()[-500:]},
        )

    rejected = answer.startswith("请求未执行") or "请求未执行" in answer

    if expected_rejection:
        return CaseResult(
            id=case["id"], passed=rejected, tool_recall=1.0 if rejected else 0.0,
            keyword_recall=1.0, rejected=rejected,
            detail={"answer_preview": answer[:120]},
        )

    actual_tools = _tools_actually_called(request_id)
    if expected_tools:
        hits = sum(1 for tool in expected_tools if tool in actual_tools)
        tool_recall = hits / len(expected_tools)
    else:
        tool_recall = 1.0

    if expected_keywords:
        kw_hits = sum(1 for kw in expected_keywords if kw in answer)
        keyword_recall = kw_hits / len(expected_keywords)
    else:
        keyword_recall = 1.0

    passed = tool_recall >= 0.5 and keyword_recall >= 0.5

    return CaseResult(
        id=case["id"], passed=passed, tool_recall=tool_recall,
        keyword_recall=keyword_recall, rejected=False,
        detail={
            "actual_tools": actual_tools,
            "expected_tools": expected_tools,
            "answer_preview": answer[:200],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="evals/agent_golden.jsonl")
    parser.add_argument("--smoke", action="store_true", help="只跑前 N 条，N 由 CI_SMOKE_LIMIT 控制")
    parser.add_argument("--smoke-limit", type=int,
                        default=int(os.getenv("CI_SMOKE_LIMIT", "3")))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report", help="写一份机读 JSON 报告到该路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="不实际跑 Agent，只校验 golden 文件格式（CI 默认）")
    args = parser.parse_args()

    cases = load_golden(Path(args.golden))
    if args.smoke:
        cases = cases[: args.smoke_limit]

    if args.dry_run:
        report = {
            "case_count": len(cases),
            "dry_run": True,
            "ids": [c["id"] for c in cases],
        }
        print(json.dumps(report, ensure_ascii=False))
        return

    from agent.react_agent import ReactAgent
    agent = ReactAgent()

    started = time.time()
    results: List[CaseResult] = []
    for case in cases:
        result = _evaluate_case(agent, case)
        results.append(result)
        if not args.quiet:
            print(json.dumps(
                {"id": result.id, "passed": result.passed,
                 "tool_recall": round(result.tool_recall, 2),
                 "keyword_recall": round(result.keyword_recall, 2),
                 "error": result.error,
                 "answer_preview": result.detail.get("answer_preview", "")[:80]},
                ensure_ascii=False))

    aggregate = {
        "case_count": len(results),
        "pass_rate": _avg(r.passed for r in results),
        "tool_recall": _avg(r.tool_recall for r in results),
        "keyword_recall": _avg(r.keyword_recall for r in results),
        "duration_s": round(time.time() - started, 2),
    }
    print(json.dumps(aggregate, ensure_ascii=False))

    if args.report:
        Path(args.report).write_text(
            json.dumps({"aggregate": aggregate,
                        "cases": [r.__dict__ for r in results]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if aggregate["pass_rate"] < float(os.getenv("AGENT_EVAL_PASS_THRESHOLD", "0.0")):
        sys.exit(1)


def _avg(seq) -> float:
    seq = list(seq)
    if not seq:
        return 0.0
    return round(sum(float(x) for x in seq) / len(seq), 3)


if __name__ == "__main__":
    main()

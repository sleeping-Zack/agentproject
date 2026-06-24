"""Prompt diff 影响评估脚本。

用法：
    python scripts/prompt_diff.py --prompt main --baseline-file prompts/main_prompt.v1.txt

工作流程：
    1. 备份当前 prompt 文件内容
    2. 用 baseline 文件替换 → 跑评测 → 记录分数
    3. 恢复当前 prompt → 再跑一次评测 → 记录分数
    4. 输出两个版本 changelog 与评测分数对比

评测目前接 scripts/evaluate_rag.py 的结果 JSON；后续可扩展接 evaluate_agent.py。
脚本默认对未提供 baseline 的情况只输出当前 prompt 的元数据与 changelog，
方便人工审阅"这次 prompt 改动到底改了什么"。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_prompt_document


def _prompt_path(prompt_name: str) -> Path:
    mapping = {
        "main": "main_prompt_path",
        "rag_summarize": "rag_summarize_prompt_path",
        "report": "report_prompt_path",
    }
    key = mapping[prompt_name]
    return Path(get_abs_path(prompts_conf[key]))


def _run_eval() -> dict:
    """跑一次 RAG 评测，返回汇总指标字典。"""
    cmd = [sys.executable, "scripts/evaluate_rag.py", "--quiet", "--json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        return {"error": "evaluate_rag.py not found"}
    if result.returncode != 0:
        return {"error": result.stderr.strip()[:500]}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"raw": result.stdout[-500:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, choices=["main", "rag_summarize", "report"])
    parser.add_argument("--baseline-file", help="可选：上一版本 prompt 文件路径，提供后才做 A/B 评测")
    parser.add_argument("--skip-eval", action="store_true",
                        help="只展示 changelog/diff，不跑评测")
    args = parser.parse_args()

    current = load_prompt_document(args.prompt)
    print(f"== Current prompt: {current.name}:{current.version} ==")
    for entry in current.changelog[:5]:
        print(f"  - {entry}")

    if not args.baseline_file:
        print("\n未提供 --baseline-file，跳过 A/B 评测。")
        return

    current_path = _prompt_path(args.prompt)
    backup = current_path.with_suffix(current_path.suffix + ".bak")
    shutil.copyfile(current_path, backup)
    try:
        shutil.copyfile(args.baseline_file, current_path)
        if args.skip_eval:
            baseline_score = {"skipped": True}
        else:
            print("\n[1/2] 跑 baseline prompt 评测…")
            baseline_score = _run_eval()
        shutil.copyfile(backup, current_path)
        if args.skip_eval:
            current_score = {"skipped": True}
        else:
            print("[2/2] 跑当前 prompt 评测…")
            current_score = _run_eval()
    finally:
        shutil.copyfile(backup, current_path)
        backup.unlink(missing_ok=True)

    report = {
        "prompt": args.prompt,
        "baseline_file": args.baseline_file,
        "current_version": current.version,
        "baseline_score": baseline_score,
        "current_score": current_score,
    }
    print("\n=== Prompt diff report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

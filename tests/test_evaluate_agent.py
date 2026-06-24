import json
import subprocess
import sys
from pathlib import Path


def test_agent_golden_jsonl_is_valid():
    path = Path("evals/agent_golden.jsonl")
    assert path.exists(), "evals/agent_golden.jsonl 缺失"
    seen_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            assert "id" in case, f"第{line_no}行缺少 id"
            assert case["id"] not in seen_ids, f"重复 id：{case['id']}"
            seen_ids.add(case["id"])
            assert ("query" in case) or ("turns" in case), \
                f"case {case['id']} 缺少 query/turns"
            if "expected_tools" in case:
                for tool in case["expected_tools"]:
                    assert "name" in tool


def test_evaluate_agent_script_dry_run():
    proc = subprocess.run(
        [sys.executable, "scripts/evaluate_agent.py", "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["dry_run"] is True
    assert payload["case_count"] >= 1

import json
import subprocess
import sys


def test_evaluate_agent_script_accepts_harness_mode_in_dry_run():
    proc = subprocess.run(
        [sys.executable, "scripts/evaluate_agent.py", "--dry-run", "--mode", "harness"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["dry_run"] is True
    assert payload["mode"] == "harness"


def test_ci_runs_full_offline_harness_gate():
    workflow = open(".github/workflows/ci.yml", encoding="utf-8").read()

    assert "scripts/evaluate_agent.py" in workflow
    assert "--mode harness" in workflow
    assert "--offline" in workflow
    assert "--golden evals/agent_offline_golden.jsonl" in workflow
    assert "--baseline evals/baselines/agent_baseline_v1.json" in workflow
    assert "--gate-config config/ci_quality_gates.yml" in workflow
    assert "--gate-profile offline_fixture" in workflow
    assert "--min-tool-recall 0.85" not in workflow
    assert "AGENT_TEST_POSTGRES_URL" in workflow
    assert "AGENT_TEST_REDIS_URL" in workflow
    assert 'pip install -e ".[dev,production]"' in workflow
    assert "--smoke" not in workflow

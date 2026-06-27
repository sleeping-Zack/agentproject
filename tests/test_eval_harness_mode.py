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


def test_ci_runs_harness_smoke_eval():
    workflow = open(".github/workflows/ci.yml", encoding="utf-8").read()

    assert "scripts/evaluate_agent.py --mode harness --smoke" in workflow

from pathlib import Path


def test_demo_and_interview_docs_exist_with_required_sections():
    demo = Path("docs/demo.md").read_text(encoding="utf-8")
    playbook = Path("docs/interview_playbook.md").read_text(encoding="utf-8")
    harness = Path("docs/interview_harness_playbook.md").read_text(encoding="utf-8")

    assert "curl" in demo
    assert "Trace 示例" in demo
    assert "三轮改造总览" in playbook
    assert "面试回答" in playbook
    assert "Harness 控制层" in harness
    assert "AgentRunner" in harness

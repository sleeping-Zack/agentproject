from agent.verifier import AnswerVerifier


def test_rag_scene_requires_evidence_and_citation():
    verifier = AnswerVerifier()

    result = verifier.verify(
        query="怎么保养滤网",
        answer="建议每周清理一次滤网。",
        evidence=[],
        scene="rag",
    )

    assert result.passed is False
    assert "evidence_required" in result.reasons
    assert "citation_missing" in result.reasons


def test_report_scene_requires_tool_results_or_artifact():
    verifier = AnswerVerifier()

    result = verifier.verify(
        query="生成本月使用记录报告",
        answer="这是报告。",
        evidence=[],
        scene="report",
        tool_results=[],
        artifacts=[],
    )

    assert result.passed is False
    assert "report_support_required" in result.reasons

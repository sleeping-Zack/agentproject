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


def test_successful_rag_tool_requires_evidence_even_when_scene_is_default():
    result = AnswerVerifier().verify(
        query="机器人无法回充怎么办",
        answer="检查充电座并擦拭回充传感器。",
        evidence=[],
        scene="default",
        tool_results=[
            {
                "tool": "rag_summarize",
                "status": "success",
                "content": "检查充电座并擦拭回充传感器。",
            }
        ],
    )

    assert result.passed is False
    assert "evidence_required" in result.reasons
    assert "citation_missing" in result.reasons
    assert result.citation_validity == 0.0
    assert result.citation_coverage == 0.0
    assert result.unsupported_claim_rate == 1.0


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

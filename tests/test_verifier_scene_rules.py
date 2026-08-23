import pytest

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


def test_successful_rag_lineage_uses_current_evidence_without_hiding_missing_citation():
    result = AnswerVerifier().verify(
        query="机器人清洁效果下降怎么办",
        answer="清理滚刷缠绕物，并确保风道没有堵塞。",
        evidence=[
            {
                "id": "current-manual",
                "content": "清理滚刷缠绕物，并确保风道没有堵塞。",
            }
        ],
        scene="default",
        tool_results=[{"tool": "rag_summarize", "status": "success"}],
    )

    assert result.passed is False
    assert result.action == "retry"
    assert result.reasons == ["citation_missing"]
    assert result.citation_validity == 0.0
    assert result.citation_coverage == 0.0
    assert result.unsupported_claim_rate == 0.0
    assert result.claim_support[0]["supported"] is True
    assert result.claim_support[0]["evidence_ids"] == ["current-manual"]


@pytest.mark.parametrize(
    "status",
    ["failed", "empty", "error", "degraded", "verification_failed"],
)
def test_unsuccessful_rag_lineage_does_not_inherit_current_evidence(status):
    result = AnswerVerifier().verify(
        query="机器人清洁效果下降怎么办",
        answer="清理滚刷缠绕物，并确保风道没有堵塞。",
        evidence=[
            {
                "id": "current-manual",
                "content": "清理滚刷缠绕物，并确保风道没有堵塞。",
            }
        ],
        scene="default",
        tool_results=[{"tool": "rag_summarize", "status": status}],
    )

    assert result.passed is False
    assert result.unsupported_claim_rate == 1.0
    assert result.claim_support[0]["reason"] == "claim_has_no_evidence"


def test_non_factual_greeting_heading_and_intro_are_not_grounding_claims():
    result = AnswerVerifier().verify(
        query="机器人清洁效果下降怎么办",
        answer=(
            "你好！\n\n"
            "## 系统排查\n\n"
            "**建议按以下顺序排查：**\n\n"
            "清理滚刷缠绕物，并确保风道没有堵塞。[1]"
        ),
        evidence=[
            {
                "id": "current-manual",
                "content": "清理滚刷缠绕物，并确保风道没有堵塞。",
            }
        ],
        scene="rag",
    )

    assert result.passed is True
    assert result.citation_coverage == 1.0
    assert result.unsupported_claim_rate == 0.0
    assert [item["claim"] for item in result.claim_support] == [
        "清理滚刷缠绕物，并确保风道没有堵塞"
    ]


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

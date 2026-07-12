from pathlib import Path
from types import SimpleNamespace

from agent.verifier import AnswerVerifier
from rag.judge import JudgeScore
from scripts.evaluate_generation import evaluate_case, load_golden, summarize


def test_offline_generation_golden_covers_positive_and_rejection_cases():
    cases = load_golden(Path("evals/generation_golden.jsonl"))
    rows = [evaluate_case(case) for case in cases]
    summary = summarize(rows)

    assert len(cases) >= 12
    assert any(case["expected_refusal"] for case in cases)
    assert any(not case["expected_refusal"] for case in cases)
    assert summary["pass_rate"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["forbidden_hit_rate"] == 0.0
    assert summary["citation_validity"] == 1.0


def test_rejected_harmful_candidate_counts_as_caught_not_escaped():
    case = {
        "id": "unsafe",
        "query": "能否水洗电机",
        "expected_refusal": True,
        "forbidden_facts": ["可以用水冲洗电机"],
        "mock_answer": "可以用水冲洗电机。\n\n引用来源：e1",
        "mock_evidence": [
            {"id": "e1", "content": "电机严禁水洗，应使用干布清洁。"}
        ],
    }

    row = evaluate_case(case)

    assert row["passed"] is True
    assert row["refused"] is True
    assert row["measured_forbidden_hit_rate"] == 1.0
    assert row["forbidden_hit_rate"] == 0.0


def test_online_semantic_out_of_domain_refusal_is_recognized():
    class Service:
        def rag_summarize_result(self, _query, tenant_id):
            assert tenant_id == "generation-eval"
            return SimpleNamespace(
                answer="参考资料未涉及股票预测，无法回答该问题。",
                evidence=[],
            )

    row = evaluate_case(
        {
            "id": "ood",
            "query": "股票明天会涨吗",
            "expected_refusal": True,
            "mock_answer": "unused",
            "mock_evidence": [],
        },
        service=Service(),
    )

    assert row["passed"] is True
    assert row["refused"] is True


def test_online_high_risk_case_may_safely_answer_or_refuse():
    class Service:
        def rag_summarize_result(self, _query, tenant_id):
            return SimpleNamespace(
                answer="请求未执行：安全资料不足。",
                evidence=[],
            )

    row = evaluate_case(
        {
            "id": "safe-refusal",
            "query": "能否水洗电机",
            "expected_refusal": True,
            "online_expected_refusal": False,
            "online_allow_refusal": True,
            "online_expected_facts": ["水", "电机"],
            "mock_answer": "unused",
            "mock_evidence": [],
        },
        service=Service(),
    )

    assert row["passed"] is True
    assert row["allow_refusal"] is True


def test_successful_semantic_judge_override_preserves_lexical_metric():
    class Judge:
        def evaluate(self, **_kwargs):
            return JudgeScore(5.0, 5.0, 5.0, "evidence supports the paraphrase")

    case = {
        "id": "semantic-paraphrase",
        "query": "机器人无法回充怎么办",
        "expected_refusal": False,
        "expected_facts": ["充电座"],
        "mock_answer": "请先清除充电座附近的杂物。\n\n引用来源：e1",
        "mock_evidence": [
            {
                "id": "e1",
                "content": "无法回充时应清理充电座周围障碍物。",
            }
        ],
    }

    row = evaluate_case(case, judge=Judge())

    assert row["lexical_unsupported_claim_rate"] > 0.05
    assert row["unsupported_claim_rate"] == 0.0
    assert row["judge"]["overrode_reasons"] == [
        "unsupported_claim_rate_exceeded"
    ]


def test_online_evaluation_passes_judge_into_rag_service(monkeypatch, tmp_path):
    from scripts import evaluate_generation

    captured = {}

    class FakeService:
        def __init__(self, verifier=None):
            captured["verifier"] = verifier

    monkeypatch.setattr("rag.rag_service.RagSummarizeService", FakeService)
    monkeypatch.setattr(evaluate_generation, "load_golden", lambda _path: [])
    monkeypatch.setattr(evaluate_generation, "summarize", lambda _rows: {})

    # Construction is the behavior under test; stop before gate metric access.
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_generation.py",
            "--online",
            "--judge",
            "--golden",
            str(tmp_path / "unused.jsonl"),
        ],
    )

    try:
        evaluate_generation.main()
    except KeyError:
        pass

    assert isinstance(captured["verifier"], AnswerVerifier)
    assert captured["verifier"].judge is not None

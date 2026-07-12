from pathlib import Path
from types import SimpleNamespace

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

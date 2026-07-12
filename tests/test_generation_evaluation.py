from pathlib import Path

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

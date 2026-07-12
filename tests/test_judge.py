import time

from rag.judge import JudgeScore, LLMJudge, evaluate_batch


def _stub_invoker(payload: str):
    def invoker(_prompt: str) -> str:
        return payload

    return invoker


def test_judge_parses_clean_json():
    invoker = _stub_invoker(
        '{"correctness": 5, "faithfulness": 4, "completeness": 4, "rationale": "答得不错"}'
    )

    judge = LLMJudge(invoker=invoker)
    score = judge.evaluate("查天气", "晴天", "今天晴天")

    assert isinstance(score, JudgeScore)
    assert score.correctness == 5.0
    assert score.faithfulness == 4.0
    assert score.completeness == 4.0
    assert score.overall == round((5 + 4 + 4) / 3, 4)
    assert score.rationale == "答得不错"


def test_judge_handles_text_with_extra_prose():
    invoker = _stub_invoker(
        "好的，这是评分：\n"
        '```json\n{"correctness": 3, "faithfulness": 2, "completeness": 4, "rationale": "部分正确"}\n```'
    )

    judge = LLMJudge(invoker=invoker)
    score = judge.evaluate("问题", "参考", "回答")

    assert score.correctness == 3.0
    assert score.faithfulness == 2.0
    assert score.completeness == 4.0


def test_judge_falls_back_when_invoker_fails():
    def broken(_prompt: str):
        raise RuntimeError("network down")

    judge = LLMJudge(invoker=broken)
    score = judge.evaluate("q", "c", "a")

    assert score.correctness == 3.0
    assert "judge error" in score.rationale
    assert score.success is False
    assert score.error_code == "invoke_error"


def test_judge_times_out_and_returns_explicit_error():
    def slow(_prompt: str):
        time.sleep(0.05)
        return '{"correctness": 5, "faithfulness": 5, "completeness": 5}'

    score = LLMJudge(invoker=slow, timeout_seconds=0.01).evaluate("q", "c", "a")

    assert score.success is False
    assert score.error_code == "timeout"
    assert score.rationale == "judge timeout"


def test_judge_clips_scores_to_valid_range():
    invoker = _stub_invoker(
        '{"correctness": 99, "faithfulness": 0.1, "completeness": "bad", "rationale": ""}'
    )
    judge = LLMJudge(invoker=invoker)

    score = judge.evaluate("q", "", "")

    assert score.correctness == 5.0
    assert score.faithfulness == 1.0
    assert score.completeness == 3.0


def test_evaluate_batch_aggregates_mean_scores():
    payloads = [
        '{"correctness": 5, "faithfulness": 5, "completeness": 5, "rationale": "好"}',
        '{"correctness": 3, "faithfulness": 3, "completeness": 3, "rationale": "中"}',
    ]
    state = {"i": 0}

    def invoker(_prompt: str):
        payload = payloads[state["i"]]
        state["i"] += 1
        return payload

    judge = LLMJudge(invoker=invoker)
    cases = [
        {"query": "q1", "context": "c1", "answer": "a1"},
        {"query": "q2", "context": "c2", "answer": "a2"},
    ]

    result = evaluate_batch(cases, judge=judge, record_metrics=False)

    assert result.aggregate["case_count"] == 2
    assert result.aggregate["correctness"] == 4.0
    assert result.aggregate["overall"] == 4.0
    assert len(result.cases) == 2

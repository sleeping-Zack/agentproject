from rag.evaluation import (
    citation_validity,
    evaluate_generation_case,
    keyword_coverage,
    summarize_generation_metrics,
)


def test_keyword_coverage_counts_all_expected_facts_present():
    assert keyword_coverage("主刷缠绕毛发需要清理滚刷", ["主刷", "毛发", "清理"]) == 1.0
    assert keyword_coverage("主刷缠绕", ["主刷", "毛发"]) == 0.5


def test_citation_validity_flags_out_of_range_indices():
    # 3 条证据，答案里出现 [1] [4] -> 1 合法 / 2 引用 = 0.5
    assert citation_validity("参考[1]和[4]", evidence_count=3) == 0.5
    # 没有引用视作 1.0
    assert citation_validity("参考资料充足", evidence_count=3) == 1.0


def test_evaluate_generation_case_returns_all_dimensions():
    case = {
        "query": "主刷缠绕毛发怎么办",
        "expected_keywords": ["主刷", "毛发", "清理"],
        "forbidden_facts": ["直接用水冲洗电机"],
        "expected_sources": ["故障排除.txt"],
    }
    answer = "主刷缠绕毛发时应清理滚刷[1]。\n\n引用来源：\n[1] 故障排除.txt"
    metrics = evaluate_generation_case(case, answer, evidence_count=1)
    assert metrics["keyword_coverage"] == 1.0
    assert metrics["forbidden_hit_rate"] == 0.0
    assert metrics["citation_hit_rate"] == 1.0
    assert metrics["citation_validity"] == 1.0


def test_forbidden_hit_rate_penalises_hallucinated_content():
    case = {"expected_keywords": ["主刷"], "forbidden_facts": ["直接用水冲洗电机"]}
    answer = "主刷可以直接用水冲洗电机"
    metrics = evaluate_generation_case(case, answer, evidence_count=1)
    assert metrics["forbidden_hit_rate"] == 1.0


def test_summarize_generation_metrics_averages_per_case():
    per_case = [
        {"keyword_coverage": 1.0, "forbidden_hit_rate": 0.0, "citation_hit_rate": 1.0, "citation_validity": 1.0},
        {"keyword_coverage": 0.5, "forbidden_hit_rate": 0.5, "citation_hit_rate": 0.5, "citation_validity": 1.0},
    ]
    summary = summarize_generation_metrics(per_case)
    assert summary["case_count"] == 2
    assert summary["keyword_coverage"] == 0.75
    assert summary["forbidden_hit_rate"] == 0.25

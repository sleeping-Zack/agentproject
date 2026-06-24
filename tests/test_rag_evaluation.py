from rag.evaluation import evaluate_case, summarize_strategy_metrics


def test_evaluate_case_calculates_recall_mrr_and_citation_hit():
    case = {
        "query": "主刷缠绕毛发怎么办",
        "expected_keywords": ["主刷", "毛发"],
        "expected_sources": ["故障排除.txt"],
    }
    retrieved = [
        {"content": "主刷缠绕毛发时需要清理", "source": "故障排除.txt"},
        {"content": "选购指南", "source": "选购指南.txt"},
    ]
    answer = "主刷缠绕毛发时需要清理滚刷。\n\n引用来源：\n[1] 故障排除.txt"

    metrics = evaluate_case(case, retrieved, answer, k=2)

    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["citation_hit_rate"] == 1.0
    assert metrics["hallucination_rate"] == 0.0


def test_summarize_strategy_metrics_compares_multiple_strategies():
    cases = [
        {"query": "a", "expected_keywords": ["主刷"], "expected_sources": ["故障排除.txt"]},
        {"query": "b", "expected_keywords": ["水箱"], "expected_sources": ["扫拖一体机器人100问.txt"]},
    ]
    results = {
        "top_k": [
            ([{"content": "主刷", "source": "故障排除.txt"}], "引用来源：\n[1] 故障排除.txt"),
            ([{"content": "无关", "source": "选购指南.txt"}], "无引用"),
        ],
        "hybrid": [
            ([{"content": "主刷", "source": "故障排除.txt"}], "引用来源：\n[1] 故障排除.txt"),
            ([{"content": "水箱", "source": "扫拖一体机器人100问.txt"}], "引用来源：\n[1] 扫拖一体机器人100问.txt"),
        ],
    }

    summary = summarize_strategy_metrics(cases, results, k=1)

    assert summary["top_k"]["case_count"] == 2
    assert summary["hybrid"]["recall_at_k"] > summary["top_k"]["recall_at_k"]

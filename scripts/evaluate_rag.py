import json
from pathlib import Path
from typing import Dict, List

from rag.rag_service import RagSummarizeService


def load_golden(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_answer(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 1.0
    hits = sum(1 for keyword in expected_keywords if keyword in answer)
    return hits / len(expected_keywords)


def main() -> None:
    cases = load_golden(Path("evals/rag_golden.jsonl"))
    service = RagSummarizeService()
    scores = []
    for case in cases:
        answer = service.rag_summarize(case["query"])
        score = score_answer(answer, case["expected_keywords"])
        scores.append(score)
        print(json.dumps({"query": case["query"], "score": score}, ensure_ascii=False))

    average = sum(scores) / len(scores) if scores else 0
    print(json.dumps({"average_score": average, "case_count": len(scores)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

from agent.long_term_memory import RuleBasedMemoryExtractor


def test_memory_extraction_golden_set_has_no_regressions():
    path = Path("evals/memory_golden.jsonl")
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    extractor = RuleBasedMemoryExtractor()

    failures = []
    for case in cases:
        actual = [
            {
                "key": item.key,
                "value": item.value,
                "explicit": item.explicit,
            }
            for item in extractor.extract(case["message"])
        ]
        if actual != case["expected"]:
            failures.append(
                {
                    "case_id": case["case_id"],
                    "expected": case["expected"],
                    "actual": actual,
                }
            )

    assert not failures

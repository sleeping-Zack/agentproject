import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests


def summarize_latency(
    latencies_ms: List[float],
    success_count: int,
    failure_count: int,
    elapsed_seconds: float,
) -> Dict[str, float]:
    sorted_latencies = sorted(latencies_ms)
    if not sorted_latencies:
        p50 = 0
        p95 = 0
    else:
        p50 = statistics.median(sorted_latencies)
        p95_index = max(0, min(len(sorted_latencies) - 1, math.ceil(len(sorted_latencies) * 0.95) - 1))
        p95 = sorted_latencies[p95_index]
    total = success_count + failure_count
    return {
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "qps": round(total / elapsed_seconds, 3) if elapsed_seconds else 0,
        "failure_rate": round(failure_count / total, 4) if total else 0,
    }


def _call(url: str, api_key: str, message: str) -> float:
    started = time.perf_counter()
    response = requests.post(
        url,
        headers={"X-API-Key": api_key},
        json={"message": message, "session_id": "benchmark"},
        timeout=60,
    )
    response.raise_for_status()
    return (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat")
    parser.add_argument("--api-key", default="dev-api-key")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--message", default="主刷缠绕毛发怎么办？")
    args = parser.parse_args()

    latencies = []
    failures = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [_call(args.url, args.api_key, args.message) for _ in range(0)]
        futures = [
            executor.submit(_call, args.url, args.api_key, args.message)
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception:
                failures += 1
    elapsed = time.perf_counter() - started
    print(json.dumps(summarize_latency(latencies, len(latencies), failures, elapsed), ensure_ascii=False))


if __name__ == "__main__":
    main()

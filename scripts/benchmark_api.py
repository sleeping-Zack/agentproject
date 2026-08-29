"""Concurrent load gate for a deployed streaming chat API."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import requests
import yaml


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return round(ordered[index], 3)


def summarize_latency(
    latencies_ms: List[float],
    success_count: int,
    failure_count: int,
    elapsed_seconds: float,
    *,
    ttft_ms: Optional[List[float]] = None,
    timeout_count: int = 0,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    total = success_count + failure_count
    ttft_values = list(ttft_ms or [])
    error_count = max(0, failure_count - timeout_count)
    return {
        "request_count": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "timeout_count": timeout_count,
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "response_latency_ms": {
            "sample_count": len(latencies_ms),
            "p50": round(statistics.median(latencies_ms), 3) if latencies_ms else None,
            "p95": _percentile(latencies_ms, 0.95),
            "p99": _percentile(latencies_ms, 0.99),
        },
        "ttft_ms": {
            "sample_count": len(ttft_values),
            "p50": round(statistics.median(ttft_values), 3) if ttft_values else None,
            "p95": _percentile(ttft_values, 0.95),
            "p99": _percentile(ttft_values, 0.99),
        },
        "qps": round(success_count / elapsed_seconds, 3) if elapsed_seconds else 0.0,
        "attempted_qps": round(total / elapsed_seconds, 3) if elapsed_seconds else 0.0,
        "failure_rate": round(failure_count / total, 4) if total else 0.0,
        "timeout_rate": round(timeout_count / total, 4) if total else 0.0,
        "error_rate": round(error_count / total, 4) if total else 0.0,
    }


def summarize_stage_latency(
    before: Dict[str, Any], after: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Summarize model/tool histogram deltas from the deployed metrics endpoint."""

    def summarize_metric(metric_name: str) -> Dict[str, Any]:
        before_histograms = before.get("histograms") or {}
        after_histograms = after.get("histograms") or {}
        matching = [
            key
            for key in after_histograms
            if key == metric_name or key.startswith(f"{metric_name}{{")
        ]
        count = 0
        total_sum = 0.0
        cumulative_by_bound: Dict[float, int] = {}
        for key in matching:
            current = after_histograms[key]
            previous = before_histograms.get(key) or {}
            current_count = int(current.get("count", 0))
            previous_count = int(previous.get("count", 0))
            count += max(0, current_count - previous_count)
            total_sum += max(
                0.0, float(current.get("sum", 0.0)) - float(previous.get("sum", 0.0))
            )
            previous_buckets = {
                float(bound): int(value)
                for bound, value in (previous.get("buckets") or [])
            }
            for bound, value in current.get("buckets") or []:
                numeric_bound = float(bound)
                delta = max(0, int(value) - previous_buckets.get(numeric_bound, 0))
                cumulative_by_bound[numeric_bound] = (
                    cumulative_by_bound.get(numeric_bound, 0) + delta
                )
        p95_upper_bound = None
        p95_overflow = False
        if count:
            target = math.ceil(count * 0.95)
            for bound in sorted(cumulative_by_bound):
                if cumulative_by_bound[bound] >= target:
                    p95_overflow = math.isinf(bound)
                    p95_upper_bound = None if p95_overflow else bound
                    break
        return {
            "sample_count": count,
            "average": round(total_sum / count, 3) if count else None,
            "p95_upper_bound": p95_upper_bound,
            "p95_overflow": p95_overflow,
        }

    return {
        "model": summarize_metric("agent_model_latency_ms"),
        "tool": summarize_metric("agent_tool_latency_ms"),
    }


def _default_metrics_url(chat_url: str) -> str:
    parsed = urlsplit(chat_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/metrics/snapshot", "", ""))


def _fetch_metrics(url: str, api_key: str, timeout_seconds: float) -> Dict[str, Any]:
    response = requests.get(
        url,
        headers={"X-API-Key": api_key},
        timeout=min(timeout_seconds, 10.0),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("metrics snapshot is not an object")
    return payload


def _call(
    url: str,
    api_key: str,
    message: str,
    timeout_seconds: float,
    *,
    stream: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "message": message,
        "session_id": f"benchmark-{uuid4()}",
        "request_id": str(uuid4()),
    }
    try:
        response = requests.post(
            url,
            headers={
                "X-API-Key": api_key,
                "X-Tenant-ID": "performance-gate",
                "X-Principal-ID": "performance-runner",
            },
            json=payload,
            timeout=timeout_seconds,
            stream=stream,
        )
        response.raise_for_status()
        ttft = None
        if stream:
            event_name = None
            terminal_event = None
            for raw_line in response.iter_lines(decode_unicode=True):
                line = (
                    raw_line.decode("utf-8", errors="replace")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                if line.startswith("event:"):
                    event_name = line.partition(":")[2].strip()
                elif line.startswith("data:") and event_name == "token_delta" and ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                elif line.startswith("data:") and event_name in {
                    "run_completed",
                    "run_failed",
                }:
                    terminal_event = event_name
            if terminal_event != "run_completed":
                return {
                    "status": "error",
                    "error": terminal_event or "stream_ended_without_terminal",
                }
        else:
            payload = response.json()
            if payload.get("status") not in {"completed", "pending_approval"}:
                return {"status": "error", "error": "unsuccessful_response_status"}
        return {
            "status": "success",
            "response_latency_ms": (time.perf_counter() - started) * 1000,
            "ttft_ms": ttft,
        }
    except requests.Timeout as exc:
        return {"status": "timeout", "error": type(exc).__name__}
    except requests.RequestException as exc:
        return {"status": "error", "error": type(exc).__name__}


def load_performance_profile(path: str, profile_name: str) -> Dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid performance gate config: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ValueError("performance gate schema_version must be 1")
    profile = (payload.get("profiles") or {}).get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"performance gate profile not found: {profile_name}")
    return {
        "policy_version": payload.get("policy_version"),
        "profile": profile_name,
        **profile,
    }


def evaluate_performance_gate(
    summary: Dict[str, Any], thresholds: Dict[str, Any]
) -> Dict[str, Any]:
    failures = []
    response = summary["response_latency_ms"]
    ttft = summary["ttft_ms"]
    stages = summary.get("stage_latency_ms") or {}
    model_stage = stages.get("model") or {}
    tool_stage = stages.get("tool") or {}
    model_limit = thresholds.get("maximum_model_stage_p95_ms")
    tool_limit = thresholds.get("maximum_tool_stage_p95_ms")
    model_p95 = (
        float(model_limit) + 1
        if model_limit is not None and model_stage.get("p95_overflow")
        else model_stage.get("p95_upper_bound")
    )
    tool_p95 = (
        float(tool_limit) + 1
        if tool_limit is not None and tool_stage.get("p95_overflow")
        else tool_stage.get("p95_upper_bound")
    )
    checks = {
        "successful_requests_below_threshold": (
            summary["success_count"],
            thresholds.get("minimum_success_count"),
            "minimum",
        ),
        "qps_below_threshold": (
            summary["qps"],
            thresholds.get("minimum_qps"),
            "minimum",
        ),
        "ttft_p95_above_threshold": (
            ttft["p95"],
            thresholds.get("maximum_ttft_p95_ms"),
            "maximum",
        ),
        "response_p95_above_threshold": (
            response["p95"],
            thresholds.get("maximum_response_p95_ms"),
            "maximum",
        ),
        "response_p99_above_threshold": (
            response["p99"],
            thresholds.get("maximum_response_p99_ms"),
            "maximum",
        ),
        "timeout_rate_above_threshold": (
            summary["timeout_rate"],
            thresholds.get("maximum_timeout_rate"),
            "maximum",
        ),
        "failure_rate_above_threshold": (
            summary["failure_rate"],
            thresholds.get("maximum_failure_rate"),
            "maximum",
        ),
        "model_stage_p95_above_threshold": (
            model_p95,
            model_limit,
            "maximum",
        ),
        "tool_stage_p95_above_threshold": (
            tool_p95,
            tool_limit,
            "maximum",
        ),
    }
    for failure, (value, limit, direction) in checks.items():
        if limit is None:
            continue
        if value is None:
            failures.append(f"metric_unavailable:{failure.split('_above_')[0]}")
        elif direction == "minimum" and float(value) < float(limit):
            failures.append(failure)
        elif direction == "maximum" and float(value) > float(limit):
            failures.append(failure)
    return {
        "passed": not failures,
        "thresholds": thresholds,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("PERFORMANCE_TARGET_URL", "http://127.0.0.1:8000/chat/stream"))
    parser.add_argument("--api-key", default=os.getenv("PERFORMANCE_API_KEY", "dev-api-key"))
    parser.add_argument("--config", default="config/api_performance_gates.yml")
    parser.add_argument("--profile", default="staging")
    parser.add_argument("--requests", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--warmup-requests", type=int)
    parser.add_argument("--message", default="主刷缠绕毛发怎么办？")
    parser.add_argument("--non-stream", action="store_true")
    parser.add_argument("--metrics-url")
    parser.add_argument("--skip-stage-metrics", action="store_true")
    parser.add_argument("--report", default="reports/api-performance.json")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    try:
        profile = load_performance_profile(args.config, args.profile)
    except ValueError as exc:
        parser.error(str(exc))
    workload = profile.get("workload") or {}
    request_count = args.requests or int(workload.get("requests", 100))
    concurrency = args.concurrency or int(workload.get("concurrency", 10))
    timeout_seconds = args.timeout_seconds or float(workload.get("timeout_seconds", 60))
    warmup_count = (
        args.warmup_requests
        if args.warmup_requests is not None
        else int(workload.get("warmup_requests", 0))
    )
    if min(request_count, concurrency) <= 0 or warmup_count < 0 or timeout_seconds <= 0:
        parser.error("requests, concurrency and timeout must be positive; warmup must be >= 0")

    stream = not args.non_stream
    for _ in range(warmup_count):
        _call(args.url, args.api_key, args.message, timeout_seconds, stream=stream)

    metrics_url = args.metrics_url or _default_metrics_url(args.url)
    metrics_errors = []
    before_metrics: Dict[str, Any] = {}
    if not args.skip_stage_metrics:
        try:
            before_metrics = _fetch_metrics(metrics_url, args.api_key, timeout_seconds)
        except (requests.RequestException, ValueError) as exc:
            metrics_errors.append(f"before:{type(exc).__name__}")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _call,
                args.url,
                args.api_key,
                args.message,
                timeout_seconds,
                stream=stream,
            )
            for _ in range(request_count)
        ]
        samples = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started

    successes = [sample for sample in samples if sample["status"] == "success"]
    timeouts = [sample for sample in samples if sample["status"] == "timeout"]
    summary = summarize_latency(
        [float(sample["response_latency_ms"]) for sample in successes],
        len(successes),
        request_count - len(successes),
        elapsed,
        ttft_ms=[
            float(sample["ttft_ms"])
            for sample in successes
            if sample.get("ttft_ms") is not None
        ],
        timeout_count=len(timeouts),
        concurrency=concurrency,
    )
    if not args.skip_stage_metrics:
        try:
            after_metrics = _fetch_metrics(metrics_url, args.api_key, timeout_seconds)
            summary["stage_latency_ms"] = (
                summarize_stage_latency(before_metrics, after_metrics)
                if before_metrics
                else {
                    "model": {
                        "sample_count": 0,
                        "average": None,
                        "p95_upper_bound": None,
                        "p95_overflow": False,
                    },
                    "tool": {
                        "sample_count": 0,
                        "average": None,
                        "p95_upper_bound": None,
                        "p95_overflow": False,
                    },
                }
            )
        except (requests.RequestException, ValueError) as exc:
            metrics_errors.append(f"after:{type(exc).__name__}")
            summary["stage_latency_ms"] = {
                "model": {
                    "sample_count": 0,
                    "average": None,
                    "p95_upper_bound": None,
                    "p95_overflow": False,
                },
                "tool": {
                    "sample_count": 0,
                    "average": None,
                    "p95_upper_bound": None,
                    "p95_overflow": False,
                },
            }
    gate = evaluate_performance_gate(summary, profile.get("thresholds") or {})
    output = {
        "schema_version": 2,
        "target": args.url,
        "streaming": stream,
        "metrics_url": metrics_url,
        "profile": profile,
        "summary": summary,
        "gate": gate,
        "errors": [sample.get("error") for sample in samples if sample.get("error")]
        + metrics_errors,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "gate": gate}, ensure_ascii=False))
    if args.gate and not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

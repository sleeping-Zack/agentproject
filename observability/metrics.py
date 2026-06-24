"""Lightweight metrics registry with Prometheus text-format export.

This is intentionally dependency-free so the project stays runnable in
sandboxes without prometheus_client installed. The exported format follows
the standard Prometheus exposition spec so scrapers and Grafana can consume
it directly.

Tracked metrics:
    - agent_request_total{status}            counter
    - agent_request_latency_ms_bucket        histogram
    - agent_tool_call_total{tool,status}     counter
    - agent_tool_latency_ms_bucket           histogram
    - agent_rag_eval_score{metric}           gauge (set by eval pipeline)
    - agent_tokens_total{kind}               counter
"""
from __future__ import annotations

import time
from collections import defaultdict
from threading import RLock
from typing import Dict, Iterable, List, Tuple

LATENCY_BUCKETS_MS: Tuple[float, ...] = (
    50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, float("inf"),
)


class _Histogram:
    def __init__(self, buckets: Iterable[float] = LATENCY_BUCKETS_MS) -> None:
        self.buckets: List[float] = list(buckets)
        self.counts: List[int] = [0] * len(self.buckets)
        self.sum: float = 0.0
        self.total: int = 0

    def observe(self, value: float) -> None:
        self.sum += value
        self.total += 1
        for index, upper in enumerate(self.buckets):
            if value <= upper:
                self.counts[index] += 1


class MetricsRegistry:
    """In-memory counters, gauges and histograms with Prometheus export."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self._histograms: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], _Histogram] = {}

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    @staticmethod
    def elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000

    @staticmethod
    def _label_key(labels: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted((k, str(v)) for k, v in labels.items()))

    def inc_counter(self, name: str, labels: Dict[str, str] | None = None, value: float = 1.0) -> None:
        with self._lock:
            key = (name, self._label_key(labels or {}))
            self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] | None = None) -> None:
        with self._lock:
            key = (name, self._label_key(labels or {}))
            self._gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] | None = None) -> None:
        with self._lock:
            key = (name, self._label_key(labels or {}))
            hist = self._histograms.get(key)
            if hist is None:
                hist = _Histogram()
                self._histograms[key] = hist
            hist.observe(value)

    # ----- domain helpers -----
    def inc_request(self, status: str) -> None:
        self.inc_counter("agent_request_total", {"status": status})

    def observe_request_latency(self, ms: float) -> None:
        self.observe_histogram("agent_request_latency_ms", ms)

    def inc_tool_call(self, tool: str, status: str) -> None:
        self.inc_counter("agent_tool_call_total", {"tool": tool, "status": status})

    def observe_tool_latency(self, tool: str, ms: float) -> None:
        self.observe_histogram("agent_tool_latency_ms", ms, {"tool": tool})

    def set_rag_score(self, metric: str, value: float) -> None:
        self.set_gauge("agent_rag_eval_score", value, {"metric": metric})

    def inc_tokens(self, kind: str, value: float) -> None:
        self.inc_counter("agent_tokens_total", {"kind": kind}, value=value)

    # ----- export -----
    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "counters": {self._format_labels(name, labels): value
                             for (name, labels), value in self._counters.items()},
                "gauges": {self._format_labels(name, labels): value
                           for (name, labels), value in self._gauges.items()},
                "histograms": {self._format_labels(name, labels): {
                    "sum": hist.sum,
                    "count": hist.total,
                    "buckets": list(zip(hist.buckets, hist.counts)),
                } for (name, labels), hist in self._histograms.items()},
            }

    @staticmethod
    def _format_labels(name: str, labels: Tuple[Tuple[str, str], ...]) -> str:
        if not labels:
            return name
        rendered = ",".join(f'{k}="{v}"' for k, v in labels)
        return f"{name}{{{rendered}}}"

    def render_prometheus(self) -> str:
        lines: List[str] = []
        with self._lock:
            counter_names = sorted({name for name, _ in self._counters})
            for name in counter_names:
                lines.append(f"# TYPE {name} counter")
                for (n, labels), value in self._counters.items():
                    if n != name:
                        continue
                    lines.append(f"{self._format_labels(name, labels)} {value}")
            gauge_names = sorted({name for name, _ in self._gauges})
            for name in gauge_names:
                lines.append(f"# TYPE {name} gauge")
                for (n, labels), value in self._gauges.items():
                    if n != name:
                        continue
                    lines.append(f"{self._format_labels(name, labels)} {value}")
            hist_names = sorted({name for name, _ in self._histograms})
            for name in hist_names:
                lines.append(f"# TYPE {name} histogram")
                for (n, labels), hist in self._histograms.items():
                    if n != name:
                        continue
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels)
                    prefix = f"{name}_bucket"
                    for upper, count in zip(hist.buckets, hist.counts):
                        le = "+Inf" if upper == float("inf") else f"{upper}"
                        items = list(labels) + [("le", le)]
                        rendered = ",".join(f'{k}="{v}"' for k, v in items)
                        lines.append(f'{prefix}{{{rendered}}} {count}')
                    suffix = f'{{{label_str}}}' if label_str else ""
                    lines.append(f"{name}_sum{suffix} {hist.sum}")
                    lines.append(f"{name}_count{suffix} {hist.total}")
        return "\n".join(lines) + "\n"


metrics_registry = MetricsRegistry()

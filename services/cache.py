"""多级缓存：TTL+LRU 内存缓存 + 语义缓存 + 工具幂等缓存。

为什么需要：Agent 应用最大的成本来源是「同一问题反复调模型 / 同一工具被
重复调用」。这套缓存让相同/相似请求直接返回历史结果，并把命中率打到 metrics
（cache_hit_total / cache_miss_total / cache_lookup_latency_ms），方便讲故事。

层次：
    MemoryCache              字符串 key → value，带 TTL + LRU
    SemanticCache(memory)    query embedding 近似命中（cosine ≥ threshold）
    ToolCallCache(memory)    工具名 + 参数 hash → 上次返回值，幂等期内复用

system prompt 不在这里处理。我们在 observability.metrics 中暴露
agent_prompt_prefix_hint_total 用来标记"system prompt 在前缀"——给豆包/通义
KV cache 命中提供线索；不需要改协议。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple

from observability.metrics import metrics_registry


class MemoryCache:
    """TTL + 容量上限的简易 LRU 缓存。线程安全。"""

    def __init__(self, max_entries: int = 1024, default_ttl: float = 300.0) -> None:
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._store: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < now:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires_at = time.time() + (ttl if ttl is not None else self.default_ttl)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (expires_at, value)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def _record_hit(name: str) -> None:
    metrics_registry.inc_counter("agent_cache_hit_total", {"cache": name})


def _record_miss(name: str) -> None:
    metrics_registry.inc_counter("agent_cache_miss_total", {"cache": name})


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    """基于 embedding 余弦相似度的近似查询缓存。

    使用方式：
        cache = SemanticCache(embedder=embed_model.embed_query, threshold=0.92)
        hit = cache.get("如何更换主刷")
        if hit is None:
            answer = rag.rag_summarize(query)
            cache.set("如何更换主刷", answer)

    threshold 越高越严格；92 在中文 query 上是一个比较稳的默认值。
    存储：用 MemoryCache 存 query→(embedding, value)，遍历找最高分。条目过多
    时改向量库，但 1k 量级遍历完全够用。
    """

    def __init__(
        self,
        embedder: Callable[[str], List[float]],
        threshold: float = 0.92,
        max_entries: int = 1024,
        ttl: float = 1800.0,
        name: str = "semantic",
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.ttl = ttl
        self.name = name
        self._memory = MemoryCache(max_entries=max_entries, default_ttl=ttl)

    def _embed(self, query: str) -> Optional[List[float]]:
        try:
            return list(self.embedder(query))
        except Exception:
            return None

    @staticmethod
    def _namespace_key(namespace: Optional[Any]) -> str:
        if namespace is None:
            return ""
        if isinstance(namespace, str):
            return namespace
        return json.dumps(
            namespace,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _entry_key(query: str, namespace_key: str) -> str:
        namespace_hash = hashlib.sha1(namespace_key.encode("utf-8")).hexdigest()
        query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()
        return f"{namespace_hash}:{query_hash}"

    def get(self, query: str, *, namespace: Optional[Any] = None) -> Optional[Any]:
        """返回同一命名空间内与 query 最相似的缓存值。

        namespace 可承载租户、知识库和模型版本等隔离字段；省略时保持原有
        全局缓存行为，兼容现有调用方。
        """
        start = metrics_registry.now()
        try:
            vec = self._embed(query)
            if vec is None:
                _record_miss(self.name)
                return None
            namespace_key = self._namespace_key(namespace)
            namespace_prefix = hashlib.sha1(namespace_key.encode("utf-8")).hexdigest() + ":"
            best_score = 0.0
            best_value = None
            for key in self._memory.keys():
                if not key.startswith(namespace_prefix):
                    continue
                entry = self._memory.get(key)
                if entry is None:
                    continue
                cached_vec, value = entry
                score = _cosine(vec, cached_vec)
                if score > best_score:
                    best_score = score
                    best_value = value
            if best_score >= self.threshold:
                _record_hit(self.name)
                metrics_registry.observe_histogram(
                    "agent_cache_lookup_latency_ms",
                    metrics_registry.elapsed_ms(start),
                    {"cache": self.name},
                )
                return best_value
            _record_miss(self.name)
            return None
        finally:
            metrics_registry.observe_histogram(
                "agent_cache_lookup_latency_ms",
                metrics_registry.elapsed_ms(start),
                {"cache": self.name},
            )

    def set(self, query: str, value: Any, *, namespace: Optional[Any] = None) -> None:
        vec = self._embed(query)
        if vec is None:
            return
        namespace_key = self._namespace_key(namespace)
        key = self._entry_key(query, namespace_key)
        self._memory.set(key, (vec, value), ttl=self.ttl)


class ToolCallCache:
    """工具调用幂等缓存。

    key = sha1(tool_name + sorted(json(args)))。TTL 默认 60s，覆盖 Agent 在
    一段对话中可能反复 query 同一工具的场景；外部数据时间敏感的工具可以传
    更短的 ttl，或单独 register_ttl 配置。
    """

    def __init__(self, default_ttl: float = 60.0, max_entries: int = 2048,
                 name: str = "tool") -> None:
        self.default_ttl = default_ttl
        self.name = name
        self._memory = MemoryCache(max_entries=max_entries, default_ttl=default_ttl)
        self._ttl_overrides: Dict[str, float] = {}

    def register_ttl(self, tool_name: str, ttl: float) -> None:
        self._ttl_overrides[tool_name] = ttl

    @staticmethod
    def _key(tool_name: str, args: Dict[str, Any]) -> str:
        payload = json.dumps({"tool": tool_name, "args": args},
                             ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Any]:
        value = self._memory.get(self._key(tool_name, args))
        if value is None:
            _record_miss(self.name)
        else:
            _record_hit(self.name)
        return value

    def set(self, tool_name: str, args: Dict[str, Any], value: Any) -> None:
        ttl = self._ttl_overrides.get(tool_name, self.default_ttl)
        self._memory.set(self._key(tool_name, args), value, ttl=ttl)


def emit_prefix_cache_hint(prompt_prefix_chars: int) -> None:
    """标记当前请求把固定 system prompt 放在前缀，便于豆包/通义命中 KV cache。

    实际是否命中由模型侧决定，但我们的工程做到了：每次请求 system prompt 不变
    且固定在最前。该 metric 主要为 PPT 用：能讲清"prefix caching"实践。
    """
    metrics_registry.inc_counter("agent_prompt_prefix_hint_total")
    metrics_registry.observe_histogram(
        "agent_prompt_prefix_chars", float(prompt_prefix_chars), {},
    )


tool_call_cache = ToolCallCache()

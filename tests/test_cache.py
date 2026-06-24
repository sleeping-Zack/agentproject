import time

from observability.metrics import metrics_registry
from services.cache import MemoryCache, SemanticCache, ToolCallCache


def test_memory_cache_basic_set_get():
    cache = MemoryCache(max_entries=4, default_ttl=10)
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_memory_cache_expires_after_ttl():
    cache = MemoryCache(default_ttl=0.05)
    cache.set("k", "v")
    time.sleep(0.1)
    assert cache.get("k") is None


def test_memory_cache_lru_evicts_oldest():
    cache = MemoryCache(max_entries=2, default_ttl=10)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # touch a → b 是 LRU 最老
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_tool_call_cache_hit_same_args():
    cache = ToolCallCache(default_ttl=10)
    cache.set("get_weather", {"city": "深圳"}, "晴天")
    assert cache.get("get_weather", {"city": "深圳"}) == "晴天"
    assert cache.get("get_weather", {"city": "杭州"}) is None


def test_tool_call_cache_records_metrics():
    cache = ToolCallCache(default_ttl=10)
    before_hit = _counter_value("agent_cache_hit_total", {"cache": "tool"})
    before_miss = _counter_value("agent_cache_miss_total", {"cache": "tool"})

    cache.get("rag_summarize", {"query": "x"})  # miss
    cache.set("rag_summarize", {"query": "x"}, "ans")
    cache.get("rag_summarize", {"query": "x"})  # hit

    after_hit = _counter_value("agent_cache_hit_total", {"cache": "tool"})
    after_miss = _counter_value("agent_cache_miss_total", {"cache": "tool"})
    assert after_hit - before_hit == 1
    assert after_miss - before_miss == 1


def test_semantic_cache_hits_above_threshold():
    table = {
        "如何更换主刷": [1.0, 0.0, 0.0],
        "怎样更换主刷": [0.99, 0.01, 0.0],
        "扫地机器人电池能用多久": [0.0, 1.0, 0.0],
    }

    def embedder(text):
        return table.get(text, [0.0, 0.0, 1.0])

    cache = SemanticCache(embedder=embedder, threshold=0.95, ttl=10)
    cache.set("如何更换主刷", "拆下盖子后……")

    # 近义提问：应命中
    assert cache.get("怎样更换主刷") == "拆下盖子后……"
    # 不相关提问：未命中
    assert cache.get("扫地机器人电池能用多久") is None


def _counter_value(name: str, labels: dict) -> float:
    key = (name, tuple(sorted(labels.items())))
    return metrics_registry._counters.get(key, 0.0)

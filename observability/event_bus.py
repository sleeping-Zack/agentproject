"""请求级事件总线：让工具/模型层向流式响应推 fine-grained 事件。

为什么需要：SSE 默认只能拿到 agent.stream() 吐出的 message chunk。但用户体验
上更想看到「正在调用 get_weather…」这种工具事件。我们在 tool middleware 调
用前后向 EventBus.publish(request_id, …)，SSE 端点用 EventBus.subscribe()
订阅同一 request_id 的事件，前端就能渲染分类气泡。

实现刻意保持简单：每个 request_id 一个 queue.Queue，订阅是同步消费 + 阻塞
get；事件结束后调用 close() 让消费者退出。无 broker，进程内独占即可——
这是单实例 Agent 服务的常见模式。
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


_SENTINEL = object()


@dataclass
class StreamEvent:
    request_id: str
    event: str           # tool_start / tool_end / heartbeat / answer / done / error
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._queues: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def channel(self, request_id: str) -> queue.Queue:
        with self._lock:
            q = self._queues.get(request_id)
            if q is None:
                q = queue.Queue()
                self._queues[request_id] = q
            return q

    def publish(self, request_id: str, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        q = self.channel(request_id)
        q.put(StreamEvent(request_id=request_id, event=event, data=data or {}))

    def close(self, request_id: str) -> None:
        # 不 pop，否则后到的 consume 会创建新 queue 拿不到 sentinel。
        # 用 sentinel 标记关闭即可，下次 consume 拿到 sentinel 后会返回 'closed'。
        q = self.channel(request_id)
        q.put(_SENTINEL)

    def consume(self, request_id: str, timeout: float = 0.5):
        """同步消费一个事件；超时返回 None，关闭后返回 sentinel 字符串 'closed'。"""
        q = self.channel(request_id)
        try:
            item = q.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _SENTINEL:
            with self._lock:
                self._queues.pop(request_id, None)
            return "closed"
        return item


event_bus = EventBus()

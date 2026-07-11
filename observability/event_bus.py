"""进程内、请求级 Agent 事件总线。

事件带严格递增序号并保留短期 replay buffer，供 SSE ``Last-Event-ID``
重连使用。队列有界；消费者长期跟不上时会取消请求，而不是无限占用内存。
多实例部署应将该接口替换为 Redis Streams/NATS 等共享实现。
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


_SENTINEL = object()


class EventBackpressureError(RuntimeError):
    pass


class EventStreamConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentEvent:
    request_id: str
    event_type: str
    sequence: int
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)

    # 兼容旧消费者；新代码使用 event_type / payload。
    @property
    def event(self) -> str:
        return self.event_type

    @property
    def data(self) -> Dict[str, Any]:
        return self.payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "event_type": self.event_type,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


# 旧 import 名保持兼容。
StreamEvent = AgentEvent


@dataclass
class _Channel:
    events: queue.Queue
    replay: Deque[AgentEvent]
    identity: Dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: threading.Event = field(default_factory=threading.Event)
    next_sequence: int = 1
    closed: bool = False
    updated_at: float = field(default_factory=time.time)


class EventBus:
    def __init__(
        self,
        queue_size: int = 256,
        replay_size: int = 512,
        retention_seconds: float = 300.0,
        backpressure_timeout: float = 1.0,
    ) -> None:
        if queue_size <= 0 or replay_size <= 0:
            raise ValueError("event queue and replay sizes must be positive")
        self.queue_size = queue_size
        self.replay_size = replay_size
        self.retention_seconds = retention_seconds
        self.backpressure_timeout = backpressure_timeout
        self._channels: Dict[str, _Channel] = {}
        self._lock = threading.Lock()

    def _get_channel(self, request_id: str) -> _Channel:
        now = time.time()
        with self._lock:
            self._cleanup_locked(now)
            channel = self._channels.get(request_id)
            if channel is None:
                channel = _Channel(
                    events=queue.Queue(maxsize=self.queue_size),
                    replay=deque(maxlen=self.replay_size),
                    updated_at=now,
                )
                self._channels[request_id] = channel
            return channel

    def open(
        self,
        request_id: str,
        identity: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Atomically create a channel and return whether this caller owns production."""
        now = time.time()
        normalized_identity = dict(identity or {})
        with self._lock:
            self._cleanup_locked(now)
            channel = self._channels.get(request_id)
            if channel is None:
                self._channels[request_id] = _Channel(
                    events=queue.Queue(maxsize=self.queue_size),
                    replay=deque(maxlen=self.replay_size),
                    identity=normalized_identity,
                    updated_at=now,
                )
                return True
            if channel.identity and normalized_identity != channel.identity:
                raise EventStreamConflictError(
                    f"request_id is already bound to another stream: {request_id}"
                )
            if normalized_identity and not channel.identity:
                channel.identity = normalized_identity
            return False

    def exists(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._channels

    def identity(self, request_id: str) -> Optional[Dict[str, str]]:
        with self._lock:
            channel = self._channels.get(request_id)
            return dict(channel.identity) if channel is not None else None

    def channel(self, request_id: str) -> queue.Queue:
        """兼容旧调用；业务代码应使用 publish/consume。"""
        return self._get_channel(request_id).events

    def publish(
        self,
        request_id: str,
        event: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        channel = self._get_channel(request_id)
        with channel.lock:
            if channel.closed:
                raise RuntimeError(f"event channel already closed: {request_id}")
            item = AgentEvent(
                request_id=request_id,
                event_type=event,
                sequence=channel.next_sequence,
                timestamp=time.time(),
                payload=data or {},
            )
            channel.next_sequence += 1
            channel.replay.append(item)
            channel.updated_at = item.timestamp
        try:
            channel.events.put(item, timeout=self.backpressure_timeout)
        except queue.Full as exc:
            # heartbeat 可从 replay 补取，丢弃 live copy 不影响答案；其他事件不可丢。
            if event == "heartbeat":
                return item
            channel.cancelled.set()
            raise EventBackpressureError(
                f"event consumer is too slow for request {request_id}"
            ) from exc
        return item

    def close(self, request_id: str) -> None:
        channel = self._get_channel(request_id)
        with channel.lock:
            if channel.closed:
                return
            channel.closed = True
            channel.updated_at = time.time()
        try:
            channel.events.put_nowait(_SENTINEL)
        except queue.Full:
            # consume() 会在队列排空后通过 closed 标志结束。
            pass

    def consume(self, request_id: str, timeout: float = 0.5):
        channel = self._get_channel(request_id)
        try:
            item = channel.events.get(timeout=timeout)
        except queue.Empty:
            return "closed" if channel.closed else None
        if item is _SENTINEL:
            return "closed"
        return item

    def replay(self, request_id: str, after_sequence: int = 0) -> List[AgentEvent]:
        channel = self._get_channel(request_id)
        with channel.lock:
            return [item for item in channel.replay if item.sequence > after_sequence]

    def cancel(self, request_id: str) -> None:
        self._get_channel(request_id).cancelled.set()

    def is_cancelled(self, request_id: str) -> bool:
        return self._get_channel(request_id).cancelled.is_set()

    def is_closed(self, request_id: str) -> bool:
        channel = self._get_channel(request_id)
        with channel.lock:
            return channel.closed

    def discard(self, request_id: str) -> None:
        with self._lock:
            self._channels.pop(request_id, None)

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            request_id
            for request_id, channel in self._channels.items()
            if channel.closed and now - channel.updated_at > self.retention_seconds
        ]
        for request_id in expired:
            self._channels.pop(request_id, None)


event_bus = EventBus()

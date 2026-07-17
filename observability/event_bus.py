"""进程内、请求级 Agent 事件总线。

事件带严格递增序号并保留短期 replay buffer，供 SSE ``Last-Event-ID``
重连使用。队列有界；消费者长期跟不上时会取消请求，而不是无限占用内存。
多实例部署应将该接口替换为 Redis Streams/NATS 等共享实现。
"""
from __future__ import annotations

import json
import math
import os
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
            self._cleanup_locked(time.time())
            return request_id in self._channels

    def identity(self, request_id: str) -> Optional[Dict[str, str]]:
        with self._lock:
            self._cleanup_locked(time.time())
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

    def consume(
        self,
        request_id: str,
        timeout: float = 0.5,
        after_sequence: Optional[int] = None,
    ):
        channel = self._get_channel(request_id)
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                item = channel.events.get(timeout=remaining)
            except queue.Empty:
                return "closed" if channel.closed else None
            if item is _SENTINEL:
                return "closed"
            if after_sequence is None or item.sequence > after_sequence:
                return item
            if time.monotonic() >= deadline:
                return "closed" if channel.closed else None

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


class RedisEventBus:
    """Redis Streams implementation shared by all application instances."""

    def __init__(
        self,
        redis_url: str = "redis://127.0.0.1:6379/0",
        replay_size: int = 512,
        retention_seconds: float = 300.0,
        key_prefix: str = "agent:events",
        client=None,
    ) -> None:
        if replay_size <= 0 or retention_seconds <= 0:
            raise ValueError("replay size and retention must be positive")
        if client is None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - production dependency
                raise RuntimeError(
                    "Redis EventBus requires the 'production' dependency extra"
                ) from exc
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        try:
            from redis.exceptions import WatchError
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError(
                "Redis EventBus requires the 'production' dependency extra"
            ) from exc
        self.client = client
        self.replay_size = replay_size
        self.retention_seconds = retention_seconds
        self.key_prefix = key_prefix.rstrip(":")
        self._watch_error = WatchError
        self._consume_cursors: Dict[str, int] = {}
        self._cursor_lock = threading.Lock()

    @property
    def _ttl(self) -> int:
        return max(1, int(math.ceil(self.retention_seconds)))

    def _keys(self, request_id: str) -> Dict[str, str]:
        # The hash tag keeps every request-scoped key in one Redis Cluster slot.
        base = f"{self.key_prefix}:{{{request_id}}}"
        return {
            "stream": base,
            "identity": f"{base}:identity",
            "sequence": f"{base}:sequence",
            "closed": f"{base}:closed",
            "cancelled": f"{base}:cancelled",
        }

    @staticmethod
    def _text(value) -> Optional[str]:
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def open(
        self,
        request_id: str,
        identity: Optional[Dict[str, str]] = None,
    ) -> bool:
        keys = self._keys(request_id)
        normalized = dict(identity or {})
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        while True:
            try:
                with self.client.pipeline() as pipe:
                    pipe.watch(keys["identity"])
                    current = self._text(pipe.get(keys["identity"]))
                    if current is None:
                        pipe.multi()
                        pipe.delete(
                            keys["stream"],
                            keys["sequence"],
                            keys["closed"],
                            keys["cancelled"],
                        )
                        pipe.set(keys["identity"], encoded, ex=self._ttl)
                        pipe.execute()
                        return True
                    current_identity = json.loads(current)
                    if current_identity != normalized:
                        if current_identity or not normalized:
                            raise EventStreamConflictError(
                                f"request_id is already bound to another stream: {request_id}"
                            )
                        pipe.multi()
                        pipe.set(keys["identity"], encoded, ex=self._ttl)
                        pipe.execute()
                        return False
                    pipe.multi()
                    for key in keys.values():
                        pipe.expire(key, self._ttl)
                    pipe.execute()
                    return False
            except self._watch_error:
                continue

    def exists(self, request_id: str) -> bool:
        return bool(self.client.exists(self._keys(request_id)["identity"]))

    def identity(self, request_id: str) -> Optional[Dict[str, str]]:
        value = self._text(self.client.get(self._keys(request_id)["identity"]))
        return None if value is None else dict(json.loads(value))

    def publish(
        self,
        request_id: str,
        event: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        if not self.exists(request_id):
            self.open(request_id)
        keys = self._keys(request_id)
        timestamp = time.time()
        payload = data or {}
        while True:
            try:
                with self.client.pipeline() as pipe:
                    pipe.watch(keys["sequence"], keys["closed"])
                    if pipe.get(keys["closed"]):
                        raise RuntimeError(f"event channel already closed: {request_id}")
                    current = self._text(pipe.get(keys["sequence"]))
                    sequence = int(current or 0) + 1
                    fields = {
                        "request_id": request_id,
                        "event_type": event,
                        "sequence": str(sequence),
                        "timestamp": repr(timestamp),
                        "payload": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                    pipe.multi()
                    pipe.set(keys["sequence"], sequence, ex=self._ttl)
                    pipe.xadd(
                        keys["stream"],
                        fields,
                        id=f"{sequence}-0",
                        maxlen=self.replay_size,
                        approximate=False,
                    )
                    pipe.expire(keys["stream"], self._ttl)
                    pipe.expire(keys["identity"], self._ttl)
                    pipe.expire(keys["closed"], self._ttl)
                    pipe.expire(keys["cancelled"], self._ttl)
                    pipe.execute()
                    return AgentEvent(request_id, event, sequence, timestamp, payload)
            except self._watch_error:
                continue

    def close(self, request_id: str) -> None:
        if not self.exists(request_id):
            self.open(request_id)
        keys = self._keys(request_id)
        with self.client.pipeline(transaction=True) as pipe:
            pipe.set(keys["closed"], "1", ex=self._ttl)
            pipe.expire(keys["identity"], self._ttl)
            pipe.expire(keys["stream"], self._ttl)
            pipe.expire(keys["sequence"], self._ttl)
            pipe.expire(keys["cancelled"], self._ttl)
            pipe.execute()

    def consume(
        self,
        request_id: str,
        timeout: float = 0.5,
        after_sequence: Optional[int] = None,
    ):
        keys = self._keys(request_id)
        if after_sequence is None:
            with self._cursor_lock:
                cursor = self._consume_cursors.get(request_id, 0)
        else:
            cursor = after_sequence
        block_ms = max(1, int(timeout * 1000))
        response = self.client.xread(
            {keys["stream"]: f"{cursor}-0"},
            count=1,
            block=block_ms,
        )
        if response:
            _, entries = response[0]
            if entries:
                item = self._event_from_entry(request_id, entries[0][1])
                if after_sequence is None:
                    with self._cursor_lock:
                        self._consume_cursors[request_id] = item.sequence
                return item
        return "closed" if self.is_closed(request_id) else None

    def replay(self, request_id: str, after_sequence: int = 0) -> List[AgentEvent]:
        entries = self.client.xrange(
            self._keys(request_id)["stream"],
            min=f"({after_sequence}-0",
            max="+",
            count=self.replay_size,
        )
        return [self._event_from_entry(request_id, fields) for _, fields in entries]

    def cancel(self, request_id: str) -> None:
        if not self.exists(request_id):
            self.open(request_id)
        keys = self._keys(request_id)
        with self.client.pipeline(transaction=True) as pipe:
            pipe.set(keys["cancelled"], "1", ex=self._ttl)
            pipe.expire(keys["identity"], self._ttl)
            pipe.expire(keys["stream"], self._ttl)
            pipe.expire(keys["sequence"], self._ttl)
            pipe.expire(keys["closed"], self._ttl)
            pipe.execute()

    def is_cancelled(self, request_id: str) -> bool:
        return bool(self.client.get(self._keys(request_id)["cancelled"]))

    def is_closed(self, request_id: str) -> bool:
        return bool(self.client.get(self._keys(request_id)["closed"]))

    def discard(self, request_id: str) -> None:
        keys = self._keys(request_id)
        self.client.delete(*keys.values())
        with self._cursor_lock:
            self._consume_cursors.pop(request_id, None)

    def _event_from_entry(self, request_id: str, fields) -> AgentEvent:
        normalized = {
            self._text(key): self._text(value)
            for key, value in fields.items()
        }
        return AgentEvent(
            request_id=normalized.get("request_id") or request_id,
            event_type=normalized["event_type"],
            sequence=int(normalized["sequence"]),
            timestamp=float(normalized["timestamp"]),
            payload=json.loads(normalized.get("payload") or "{}"),
        )


def create_event_bus():
    backend = os.getenv("AGENT_EVENT_BUS_BACKEND", "memory").strip().lower()
    if backend == "memory":
        return EventBus(
            queue_size=int(os.getenv("AGENT_EVENT_QUEUE_SIZE", "256")),
            replay_size=int(os.getenv("AGENT_EVENT_REPLAY_SIZE", "512")),
            retention_seconds=float(os.getenv("AGENT_EVENT_RETENTION_SECONDS", "300")),
        )
    if backend == "redis":
        return RedisEventBus(
            redis_url=os.getenv("AGENT_REDIS_URL", "redis://127.0.0.1:6379/0"),
            replay_size=int(os.getenv("AGENT_EVENT_REPLAY_SIZE", "512")),
            retention_seconds=float(os.getenv("AGENT_EVENT_RETENTION_SECONDS", "300")),
            key_prefix=os.getenv("AGENT_EVENT_KEY_PREFIX", "agent:events"),
        )
    raise ValueError(f"unsupported event bus backend: {backend}")


event_bus = create_event_bus()

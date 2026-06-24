"""Conversation memory with pluggable persistent backends.

ConversationMemory keeps a small in-process LRU cache for hot reads while
optionally syncing each message to a persistent store (SQLite by default).
This lets the same Agent serve multi-process or rolling-restart deployments
without losing context, which is what real Agent applications require.
"""
from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Dict, List, Optional, Protocol


class SessionStore(Protocol):
    def load_messages(self, session_id: str) -> List[Dict[str, str]]: ...

    def append_message(self, session_id: str, role: str, content: str) -> None: ...


class InMemorySessionStore:
    """Fallback store used by tests and offline demos."""

    def __init__(self) -> None:
        self._messages: Dict[str, List[Dict[str, str]]] = {}
        self._lock = RLock()

    def load_messages(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return deepcopy(self._messages.get(session_id, []))

    def append_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._messages.setdefault(session_id, []).append(
                {"role": role, "content": content}
            )


class ConversationMemory:
    """Sliding-window memory with optional persistent backing store."""

    def __init__(
        self,
        max_messages: int = 20,
        store: Optional[SessionStore] = None,
        summarizer=None,
        summary_trigger: int = 40,
        summary_keep_recent: int = 6,
    ) -> None:
        self.max_messages = max_messages
        self.store: SessionStore = store or InMemorySessionStore()
        self.summarizer = summarizer
        self.summary_trigger = summary_trigger
        self.summary_keep_recent = summary_keep_recent
        self._cache: Dict[str, List[Dict[str, str]]] = {}
        self._profiles: Dict[str, Dict[str, str]] = {}
        self._last_tool_results: Dict[str, Dict[str, str]] = {}
        self._summaries: Dict[str, str] = {}
        self._lock = RLock()

    def _load_into_cache(self, session_id: str) -> List[Dict[str, str]]:
        if session_id in self._cache:
            return self._cache[session_id]
        history = self.store.load_messages(session_id)
        self._cache[session_id] = history
        return history

    @staticmethod
    def _key(session_id: str, tenant_id: str = "default") -> str:
        return f"{tenant_id}|{session_id}"

    def add_message(self, session_id: str, role: str, content: str,
                    tenant_id: str = "default") -> None:
        key = self._key(session_id, tenant_id)
        with self._lock:
            cached = self._load_into_cache(key)
            self.store.append_message(key, role, content)
            cached.append({"role": role, "content": content})
            self._maybe_compress(key)

    def get_messages(self, session_id: str,
                     tenant_id: str = "default") -> List[Dict[str, str]]:
        key = self._key(session_id, tenant_id)
        with self._lock:
            cached = self._load_into_cache(key)
            window = cached[-self.max_messages :]
            summary = self._summaries.get(key)
            if summary:
                return [{"role": "system", "content": f"对话历史摘要：{summary}"}] + deepcopy(window)
            return deepcopy(window)

    def update_profile(self, session_id: str, values: Dict[str, str],
                       tenant_id: str = "default") -> None:
        key = self._key(session_id, tenant_id)
        with self._lock:
            self._profiles.setdefault(key, {}).update(values)

    def set_last_tool_result(self, session_id: str, tool_name: str, result: str,
                             tenant_id: str = "default") -> None:
        key = self._key(session_id, tenant_id)
        with self._lock:
            self._last_tool_results.setdefault(key, {})[tool_name] = result

    def snapshot(self, session_id: str,
                 tenant_id: str = "default") -> Dict[str, Dict[str, str]]:
        key = self._key(session_id, tenant_id)
        with self._lock:
            return {
                "profile": deepcopy(self._profiles.get(key, {})),
                "last_tool_results": deepcopy(self._last_tool_results.get(key, {})),
                "summary": self._summaries.get(key, ""),
            }

    def get_summary(self, session_id: str, tenant_id: str = "default") -> str:
        return self._summaries.get(self._key(session_id, tenant_id), "")

    def _maybe_compress(self, key: str) -> None:
        if not self.summarizer:
            return
        cached = self._cache.get(key, [])
        if len(cached) < self.summary_trigger:
            return
        keep = max(1, self.summary_keep_recent)
        to_compress = cached[:-keep]
        if not to_compress:
            return
        previous_summary = self._summaries.get(key, "")
        try:
            new_summary = self.summarizer(to_compress, previous_summary)
        except Exception:
            return
        if not new_summary:
            return
        self._summaries[key] = new_summary
        self._cache[key] = cached[-keep:]

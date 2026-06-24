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

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self.store.append_message(session_id, role, content)
            cached = self._load_into_cache(session_id)
            cached.append({"role": role, "content": content})
            self._maybe_compress(session_id)

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            cached = self._load_into_cache(session_id)
            window = cached[-self.max_messages :]
            summary = self._summaries.get(session_id)
            if summary:
                return [{"role": "system", "content": f"对话历史摘要：{summary}"}] + deepcopy(window)
            return deepcopy(window)

    def update_profile(self, session_id: str, values: Dict[str, str]) -> None:
        with self._lock:
            self._profiles.setdefault(session_id, {}).update(values)

    def set_last_tool_result(self, session_id: str, tool_name: str, result: str) -> None:
        with self._lock:
            self._last_tool_results.setdefault(session_id, {})[tool_name] = result

    def snapshot(self, session_id: str) -> Dict[str, Dict[str, str]]:
        with self._lock:
            return {
                "profile": deepcopy(self._profiles.get(session_id, {})),
                "last_tool_results": deepcopy(self._last_tool_results.get(session_id, {})),
                "summary": self._summaries.get(session_id, ""),
            }

    def get_summary(self, session_id: str) -> str:
        return self._summaries.get(session_id, "")

    def _maybe_compress(self, session_id: str) -> None:
        if not self.summarizer:
            return
        cached = self._cache.get(session_id, [])
        if len(cached) < self.summary_trigger:
            return
        keep = max(1, self.summary_keep_recent)
        to_compress = cached[:-keep]
        if not to_compress:
            return
        previous_summary = self._summaries.get(session_id, "")
        try:
            new_summary = self.summarizer(to_compress, previous_summary)
        except Exception:
            return
        if not new_summary:
            return
        self._summaries[session_id] = new_summary
        self._cache[session_id] = cached[-keep:]

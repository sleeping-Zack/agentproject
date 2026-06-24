from copy import deepcopy
from typing import Dict, List


class ConversationMemory:
    """Small in-process memory for demo sessions and tests."""

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self._messages: Dict[str, List[Dict[str, str]]] = {}
        self._profiles: Dict[str, Dict[str, str]] = {}
        self._last_tool_results: Dict[str, Dict[str, str]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        messages = self._messages.setdefault(session_id, [])
        messages.append({"role": role, "content": content})
        if len(messages) > self.max_messages:
            self._messages[session_id] = messages[-self.max_messages :]

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        return deepcopy(self._messages.get(session_id, []))

    def update_profile(self, session_id: str, values: Dict[str, str]) -> None:
        self._profiles.setdefault(session_id, {}).update(values)

    def set_last_tool_result(self, session_id: str, tool_name: str, result: str) -> None:
        self._last_tool_results.setdefault(session_id, {})[tool_name] = result

    def snapshot(self, session_id: str) -> Dict[str, Dict[str, str]]:
        return {
            "profile": deepcopy(self._profiles.get(session_id, {})),
            "last_tool_results": deepcopy(self._last_tool_results.get(session_id, {})),
        }

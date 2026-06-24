from typing import Any, Dict


class MemoryCache:
    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}

    def get(self, key: str):
        return self._values.get(key)

    def set(self, key: str, value) -> None:
        self._values[key] = value

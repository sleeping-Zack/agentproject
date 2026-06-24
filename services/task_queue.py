from collections import deque
from typing import Callable, Deque


class InMemoryTaskQueue:
    def __init__(self) -> None:
        self._tasks: Deque[Callable] = deque()

    def enqueue(self, task: Callable) -> int:
        self._tasks.append(task)
        return len(self._tasks)

    def run_next(self):
        if not self._tasks:
            return None
        return self._tasks.popleft()()

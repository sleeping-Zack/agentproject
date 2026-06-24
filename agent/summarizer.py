"""Conversation summarizer for long-running sessions.

When the history exceeds the configured trigger size, ConversationMemory
calls this summarizer to compress earlier turns into a short Chinese summary
and only keeps the most recent N exchanges verbatim. This avoids context
window blow-up without losing topical continuity.

The summarizer is intentionally chat_model-agnostic: it accepts any callable
that maps a prompt string to a string response, which keeps it easy to mock
in tests and easy to swap to different providers.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

SUMMARY_PROMPT_TEMPLATE = """你是一个长对话归纳助手。请把以下历史对话压缩成不超过200字的中文摘要，
保留：用户身份、关心的设备/型号、未解决的问题、已经回答过的关键事实。
不要新增信息，不要复述完整对话。

之前的摘要（如果有）：
{previous_summary}

需要被合并到摘要中的历史对话：
{dialogue}

请直接输出新的摘要正文，不要加前缀。"""


def _format_dialogue(messages: List[Dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


class ConversationSummarizer:
    """Compress old chat turns into a short Chinese summary."""

    def __init__(self, invoker: Optional[Callable[[str], str]] = None) -> None:
        self._invoker = invoker

    def _default_invoker(self, prompt: str) -> str:
        from model.factory import chat_model

        response = chat_model.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content).strip()

    def __call__(self, messages: List[Dict[str, str]], previous_summary: str = "") -> str:
        if not messages:
            return previous_summary
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            previous_summary=previous_summary or "（暂无）",
            dialogue=_format_dialogue(messages),
        )
        invoker = self._invoker or self._default_invoker
        try:
            return invoker(prompt).strip()
        except Exception:
            return previous_summary

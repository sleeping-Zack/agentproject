"""Prompt 加载器，支持版本号 / changelog frontmatter。

Prompt 文件可以选择性以 YAML frontmatter 开头：

    ---
    version: v2
    changelog:
      - v2: 强化报告生成强约束，要求必先调 fill_context_for_report
      - v1: 初版
    ---
    你是扫地机器人……（正文）

加载时返回 PromptDocument(content, version, changelog)，并把 version 通过
ContextVar 注入到当前请求上下文，让结构化日志和 trace 自动带上 prompt 版本。
没有 frontmatter 的文件视为 version=unversioned，向后兼容。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from observability.context import update_request_context
from utils.config_handler import prompts_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class PromptDocument:
    name: str
    content: str
    version: str = "unversioned"
    changelog: List[str] = field(default_factory=list)

    def render(self) -> str:
        return self.content


def _parse(name: str, raw: str) -> PromptDocument:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return PromptDocument(name=name, content=raw)
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning(f"[prompt_loader]frontmatter 解析失败：{exc}")
        return PromptDocument(name=name, content=raw)
    body = raw[match.end():]
    changelog = meta.get("changelog") or []
    if isinstance(changelog, dict):
        changelog = [f"{k}: {v}" for k, v in changelog.items()]
    return PromptDocument(
        name=name,
        content=body,
        version=str(meta.get("version", "unversioned")),
        changelog=[str(item) for item in changelog],
    )


def _load_prompt(name: str, config_key: str) -> PromptDocument:
    try:
        path = get_abs_path(prompts_conf[config_key])
    except KeyError:
        logger.error(f"[prompt_loader]缺少配置项 {config_key}")
        raise
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    doc = _parse(name, raw)
    return doc


def _activate(doc: PromptDocument) -> str:
    update_request_context(prompt_version=f"{doc.name}:{doc.version}")
    return doc.render()


def load_system_prompts() -> str:
    doc = _load_prompt("main", "main_prompt_path")
    return _activate(doc)


def load_rag_prompts() -> str:
    doc = _load_prompt("rag_summarize", "rag_summarize_prompt_path")
    return _activate(doc)


def load_report_prompts() -> str:
    doc = _load_prompt("report", "report_prompt_path")
    return _activate(doc)


def load_prompt_document(name: str) -> Optional[PromptDocument]:
    """供评测脚本读取 prompt 元数据（用于 diff/changelog 展示）。"""
    mapping = {
        "main": "main_prompt_path",
        "rag_summarize": "rag_summarize_prompt_path",
        "report": "report_prompt_path",
    }
    key = mapping.get(name)
    if not key:
        return None
    return _load_prompt(name, key)


if __name__ == '__main__':
    doc = load_prompt_document("main")
    print(doc.name, doc.version, doc.changelog[:2])

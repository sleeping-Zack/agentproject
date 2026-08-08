from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol
from uuid import uuid4

from observability.metrics import metrics_registry

LOGGER = logging.getLogger(__name__)


class MemoryCategory(str, Enum):
    TRANSIENT = "transient"
    EPISODIC = "episodic"
    DEVICE_STATE = "device_state"
    DEVICE_IDENTITY = "device_identity"
    USER_PREFERENCE = "user_preference"
    STABLE_PROFILE = "stable_profile"
    OPEN_ITEM = "open_item"
    SAFETY_CONSTRAINT = "safety_constraint"
    USER_POLICY = "user_policy"


HALF_LIFE_DAYS: Dict[MemoryCategory, Optional[float]] = {
    MemoryCategory.TRANSIENT: 1.0,
    MemoryCategory.EPISODIC: 30.0,
    MemoryCategory.DEVICE_STATE: 30.0,
    MemoryCategory.DEVICE_IDENTITY: 180.0,
    MemoryCategory.USER_PREFERENCE: 180.0,
    MemoryCategory.STABLE_PROFILE: 365.0,
    MemoryCategory.OPEN_ITEM: None,
    MemoryCategory.SAFETY_CONSTRAINT: None,
    MemoryCategory.USER_POLICY: None,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def calculate_time_decay(
    category: MemoryCategory,
    last_confirmed_at: datetime,
    now: Optional[datetime] = None,
) -> float:
    half_life = HALF_LIFE_DAYS[MemoryCategory(category)]
    if half_life is None:
        return 1.0
    current = now or utc_now()
    age_days = max(0.0, (current - last_confirmed_at).total_seconds() / 86400.0)
    return round(math.pow(2.0, -age_days / half_life), 12)


@dataclass(frozen=True)
class MemoryCandidate:
    key: str
    value: str
    category: MemoryCategory
    explicit: bool = False
    confidence: float = 0.9
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    operation: str = "upsert"


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    tenant_id: str
    user_id: str
    key: str
    value: str
    category: MemoryCategory
    status: str
    version: int
    importance: float
    confidence: float
    reinforcement: float
    explicit: bool
    created_at: datetime
    updated_at: datetime
    last_confirmed_at: datetime
    valid_from: datetime
    valid_to: Optional[datetime] = None
    supersedes_id: Optional[str] = None
    source_event_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredMemory:
    memory: MemoryRecord
    score: float
    relevance: float
    recency: float


@dataclass(frozen=True)
class ProcedureMemory:
    procedure_id: str
    tenant_id: Optional[str]
    agent_version: str
    status: str
    title: str
    content: str
    evidence: Dict[str, Any]
    created_at: datetime
    approved_at: Optional[datetime] = None


@dataclass(frozen=True)
class MemoryCommandResult:
    handled: bool
    status: str = "ignored"
    action: str = ""
    message: str = ""
    records: List[MemoryRecord] = field(default_factory=list)
    deleted: int = 0
    rejected_reason: str = ""


class MemoryStore(Protocol):
    def get_active_fact(self, tenant_id: str, user_id: str, key: str) -> Optional[MemoryRecord]: ...
    def write_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        value: str,
        category: MemoryCategory,
        importance: float,
        confidence: float,
        explicit: bool,
        source_event_id: Optional[str],
        metadata: Dict[str, Any],
        current_time: datetime,
    ) -> MemoryRecord: ...
    def save_fact(self, memory: MemoryRecord, supersede_id: Optional[str] = None) -> None: ...
    def confirm_fact(self, memory_id: str, confirmed_at: datetime) -> MemoryRecord: ...
    def mark_fact_status(
        self, memory_id: str, status: str, changed_at: datetime
    ) -> MemoryRecord: ...
    def list_facts(
        self, tenant_id: str, user_id: str, include_inactive: bool = False
    ) -> List[MemoryRecord]: ...
    def forget_facts(self, tenant_id: str, user_id: str, key: Optional[str] = None) -> int: ...
    def has_tombstone(self, tenant_id: str, user_id: str, key: str, value: str) -> bool: ...
    def clear_tombstone(self, tenant_id: str, user_id: str, key: str, value: str) -> None: ...
    def append_event(self, event: Dict[str, Any]) -> str: ...
    def list_events(self, tenant_id: str, user_id: str, limit: int = 100) -> List[MemoryRecord]: ...
    def list_event_details(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]: ...
    def list_summaries(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]: ...
    def log_access(self, memory_id: str, tenant_id: str, user_id: str, score: float) -> None: ...
    def prune_retention(
        self,
        raw_message_days: int,
        episodic_days: int,
        superseded_fact_days: int,
        access_log_days: int,
        procedure_candidate_days: int,
    ) -> Dict[str, Any]: ...
    def save_procedure(self, procedure: ProcedureMemory) -> None: ...
    def approve_procedure(self, procedure_id: str, approved_at: datetime) -> ProcedureMemory: ...
    def list_procedures(
        self, tenant_id: Optional[str], status: str = "approved"
    ) -> List[ProcedureMemory]: ...


class RuleBasedMemoryExtractor:
    """Conservative first-pass extractor for the approved memory allowlist."""

    _SENSITIVE_TERMS = (
        "密码", "验证码", "银行卡", "身份证", "password", "passcode",
        "token", "令牌", "secret", "私钥"
    )
    _SENSITIVE_PATTERNS = (
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
        re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
        re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        re.compile(r"\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    )
    _EXPLICIT_PREFIX = re.compile(
        r"^(?:(?:请|帮我)?(?:记住(?:一下)?|记一下|记录一下))[：,:，\s]*"
    )
    _CORRECTION_PREFIX = re.compile(r"^(?:请)?(?:更正|修改|更新)(?:一下)?[：,:，\s]*")
    _FORGET_PREFIX = re.compile(r"^(?:请)?(?:忘记|删除)(?:掉)?[：,:，\s]*")
    _RESOLVE_PATTERN = re.compile(
        r"^(?:请)?(?:将|把)?(?:我的)?待处理事项"
        r"(?:标记为|设为|已经|已)?(?:已解决|已完成|完成)[。！!]?$"
    )
    _PERSISTENT_DIRECTIVE = re.compile(
        r"^(?:请记住[：,:，\s]*)?"
        r"(?:以后|今后|从现在起|从今往后)(?:每次|一直|都)?"
        r"(?:请|要|需要|必须|都要)?\s*(.+)$"
    )
    _UNSAFE_POLICY_PATTERNS = (
        re.compile(r"(?:忽略|绕过|覆盖).{0,20}(?:系统|开发者|安全|权限|审批)"),
        re.compile(
            r"(?:ignore|bypass|override).{0,30}"
            r"(?:system|developer|safety|permission|approval)",
            re.IGNORECASE,
        ),
    )
    _RULES = (
        (
            re.compile(
                r"^(?:以后|今后)(?:每次)?回答(?:的前(?:两|2)个字)?"
                r"(?:必须|都要|要|请)?(?:先)?说\s*[“\"]?(.{1,20}?)[”\"]?$"
            ),
            "policy.response_prefix",
            MemoryCategory.USER_POLICY,
        ),
        (
            re.compile(
                r"^(?:以后|今后)(?:每次)?回答(?:必须|都要|要|请)?"
                r"以\s*[“\"]?(.{1,20}?)[”\"]?开头$"
            ),
            "policy.response_prefix",
            MemoryCategory.USER_POLICY,
        ),
        (
            re.compile(r"^(?:我的)?(?:扫地机器人|设备)?型号(?:是|为)\s*(.+)$"),
            "device.model",
            MemoryCategory.DEVICE_IDENTITY,
        ),
        (
            re.compile(r"^我(?:现在)?住在\s*(.+)$"),
            "profile.city",
            MemoryCategory.STABLE_PROFILE,
        ),
        (
            re.compile(r"^我(?:喜欢|偏好)\s*(.+)$"),
            "preference.general",
            MemoryCategory.USER_PREFERENCE,
        ),
        (
            re.compile(r"^我家面积(?:是|为)\s*(.+)$"),
            "profile.home_area",
            MemoryCategory.STABLE_PROFILE,
        ),
        (
            re.compile(r"^我家有\s*(.+宠物)$"),
            "profile.pets",
            MemoryCategory.STABLE_PROFILE,
        ),
        (
            re.compile(r"^(?:我的)?(?:扫地机器人|设备)(?:目前|现在)(?:状态是|状态为|出现|无法)\s*(.+)$"),
            "device.current_state",
            MemoryCategory.DEVICE_STATE,
        ),
        (
            re.compile(r"^我的待处理事项(?:是|为)\s*(.+)$"),
            "open_item.current",
            MemoryCategory.OPEN_ITEM,
        ),
        (
            re.compile(r"^未经我确认(?:不要|不得)\s*(.+)$"),
            "safety.confirmation_required",
            MemoryCategory.SAFETY_CONSTRAINT,
        ),
    )

    @classmethod
    def is_sensitive(cls, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in cls._SENSITIVE_TERMS) or any(
            pattern.search(text) for pattern in cls._SENSITIVE_PATTERNS
        )

    @classmethod
    def is_unsafe_policy(cls, text: str) -> bool:
        return any(pattern.search(text) for pattern in cls._UNSAFE_POLICY_PATTERNS)

    @classmethod
    def is_explicit_command(cls, user_message: str) -> bool:
        text = user_message.strip()
        return bool(
            cls._EXPLICIT_PREFIX.match(text)
            or cls._CORRECTION_PREFIX.match(text)
            or cls._FORGET_PREFIX.match(text)
            or cls._RESOLVE_PATTERN.match(text)
            or cls._PERSISTENT_DIRECTIVE.match(text)
        )

    @classmethod
    def forget_payload(cls, user_message: str) -> Optional[str]:
        text = user_message.strip().rstrip("。！!")
        if not cls._FORGET_PREFIX.match(text):
            return None
        return cls._FORGET_PREFIX.sub("", text, count=1).strip()

    @classmethod
    def resolves_open_item(cls, user_message: str) -> bool:
        return bool(cls._RESOLVE_PATTERN.match(user_message.strip()))

    def extract(self, user_message: str) -> List[MemoryCandidate]:
        text = user_message.strip().rstrip("。！!")
        if self.is_sensitive(text):
            return []
        explicit = bool(
            self._EXPLICIT_PREFIX.match(text) or self._CORRECTION_PREFIX.match(text)
        )
        if explicit:
            text = self._EXPLICIT_PREFIX.sub("", text, count=1).strip()
            text = self._CORRECTION_PREFIX.sub("", text, count=1).strip()
        segments = [
            segment.strip()
            for segment in re.split(r"(?:[；;。！!\n]+|同时)", text)
            if segment.strip()
        ]
        extracted: Dict[str, MemoryCandidate] = {}
        for segment in segments:
            for pattern, key, category in self._RULES:
                match = pattern.match(segment)
                if not match:
                    continue
                value = match.group(1).strip()
                if not value:
                    break
                extracted[key] = MemoryCandidate(
                    key=key,
                    value=value,
                    category=category,
                    explicit=explicit,
                    confidence=1.0 if explicit else 0.9,
                )
                break
            else:
                directive = self._PERSISTENT_DIRECTIVE.match(segment)
                if directive:
                    value = re.sub(
                        r"^(?:请|要|需要|必须|都要)\s*", "", directive.group(1).strip()
                    )
                    if value:
                        digest = hashlib.sha256(
                            re.sub(r"\s+", "", value).encode("utf-8")
                        ).hexdigest()[:12]
                        extracted[f"policy.instruction.{digest}"] = MemoryCandidate(
                            key=f"policy.instruction.{digest}",
                            value=value,
                            category=MemoryCategory.USER_POLICY,
                            explicit=True,
                            confidence=1.0,
                            metadata={
                                "source": "deterministic_fallback",
                                "scope": "global",
                                "evidence": segment,
                            },
                        )
        return list(extracted.values())


class StructuredMemoryExtractor:
    """Model-assisted extractor with a strict, deterministic acceptance gate."""

    _ANALYSIS_CUES = re.compile(
        r"(?:请记住|记一下|更正|修改|更新|忘记|删除|以后|今后|从现在起|"
        r"我(?:是|住在|家有|喜欢|偏好|习惯|通常|一直|需要|希望)|"
        r"我的(?:职业|工作|设备|偏好|习惯|待处理))"
    )
    _KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{1,119}$")
    _CATEGORY_PREFIX = {
        MemoryCategory.USER_POLICY: "policy",
        MemoryCategory.USER_PREFERENCE: "preference",
        MemoryCategory.STABLE_PROFILE: "profile",
        MemoryCategory.DEVICE_IDENTITY: "device",
        MemoryCategory.DEVICE_STATE: "device_state",
        MemoryCategory.OPEN_ITEM: "open_item",
        MemoryCategory.SAFETY_CONSTRAINT: "safety",
    }
    _ALLOWED_CATEGORIES = frozenset(_CATEGORY_PREFIX)
    _PROMPT = """你是一个长期记忆抽取器。只根据用户原文提取可跨会话使用的记忆，不推断。

输出严格 JSON：{{"memories": [...]}}，不要输出 Markdown。每一项必须包含：
- operation: upsert、delete、resolve 或 none
- slot: 稳定、简短的英文语义槽位，如 response.format、occupation、city
- value: 规范化后的中文值；delete/resolve 可为空
- category: user_policy、user_preference、stable_profile、device_identity、
  device_state、open_item、safety_constraint 之一
- scope: global、topic、session 或 turn
- durability: long_term 或 temporary
- confidence: 0 到 1
- evidence: 用户原文中的连续原句，必须逐字存在

判定规则：
1. “记住、以后、今后、每次、一直”等明确跨会话指令属于 long_term。
2. 明确的稳定身份、画像、偏好可属于 long_term；一次性要求、当前问题和模型猜测不是。
3. 行为要求属于 user_policy；偏好倾向属于 user_preference；事实画像属于 stable_profile。
4. 同一语义必须使用稳定 slot，肯定与否定、修改前后不得换 slot。
5. 不保存密码、令牌、身份证、银行卡、验证码、联系方式等敏感数据。
6. 一句话可以有多项；不确定就返回空数组。

用户原文：
{message}
"""

    def __init__(
        self,
        invoke: Callable[[str], str],
        *,
        implicit_confidence: Optional[float] = None,
        explicit_confidence: Optional[float] = None,
    ) -> None:
        self.invoke = invoke
        self.implicit_confidence = (
            float(os.getenv("AGENT_MEMORY_IMPLICIT_CONFIDENCE", "0.85"))
            if implicit_confidence is None
            else implicit_confidence
        )
        self.explicit_confidence = (
            float(os.getenv("AGENT_MEMORY_EXPLICIT_CONFIDENCE", "0.70"))
            if explicit_confidence is None
            else explicit_confidence
        )

    @classmethod
    def should_analyze(cls, user_message: str) -> bool:
        return bool(cls._ANALYSIS_CUES.search(user_message.strip()))

    def extract(self, user_message: str) -> List[MemoryCandidate]:
        candidates, _authoritative = self.extract_with_status(user_message)
        return candidates

    def extract_with_status(
        self, user_message: str
    ) -> tuple[List[MemoryCandidate], bool]:
        text = user_message.strip()
        if (
            not text
            or RuleBasedMemoryExtractor.is_sensitive(text)
            or not self.should_analyze(text)
        ):
            return [], False
        try:
            raw = self.invoke(self._PROMPT.format(message=text))
            payload = self._parse_payload(raw)
        except Exception:
            metrics_registry.inc_counter(
                "agent_memory_semantic_extraction_total", {"status": "error"}
            )
            LOGGER.exception("semantic memory extraction failed")
            return [], False

        explicit = RuleBasedMemoryExtractor.is_explicit_command(text)
        accepted: Dict[str, MemoryCandidate] = {}
        for item in payload:
            candidate = self._candidate(item, text, explicit)
            if candidate is not None:
                accepted[candidate.key] = candidate
        metrics_registry.inc_counter(
            "agent_memory_semantic_extraction_total",
            {"status": "accepted" if accepted else "empty"},
        )
        return list(accepted.values()), True

    @staticmethod
    def _parse_payload(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, str):
            content = getattr(raw, "content", "")
            raw = content if isinstance(content, str) else str(content)
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        decoded = json.loads(text)
        memories = decoded.get("memories", []) if isinstance(decoded, dict) else decoded
        if not isinstance(memories, list):
            raise ValueError("semantic memory output must contain a memories list")
        return [item for item in memories if isinstance(item, dict)]

    def _candidate(
        self,
        item: Dict[str, Any],
        source_text: str,
        explicit: bool,
    ) -> Optional[MemoryCandidate]:
        operation = str(item.get("operation") or "upsert").strip().lower()
        if operation not in {"upsert", "delete", "resolve"}:
            return None
        if str(item.get("durability") or "").strip().lower() != "long_term":
            return None
        scope = str(item.get("scope") or "").strip().lower()
        if scope not in {"global", "topic"}:
            return None
        evidence = str(item.get("evidence") or "").strip()
        if not evidence or evidence not in source_text:
            return None
        try:
            category = MemoryCategory(str(item.get("category") or "").strip())
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        if category not in self._ALLOWED_CATEGORIES:
            return None
        threshold = self.explicit_confidence if explicit else self.implicit_confidence
        if confidence < threshold:
            return None
        slot = str(item.get("slot") or "").strip().lower().replace(" ", "_")
        prefix = self._CATEGORY_PREFIX[category]
        if slot.startswith(f"{prefix}."):
            key = slot
        else:
            key = f"{prefix}.{slot}"
        if not self._KEY_PATTERN.fullmatch(key):
            return None
        value = str(item.get("value") or "").strip()
        if operation == "upsert" and not value:
            return None
        if len(value) > 1000 or RuleBasedMemoryExtractor.is_sensitive(value):
            return None
        if (
            category == MemoryCategory.USER_POLICY
            and RuleBasedMemoryExtractor.is_unsafe_policy(value)
        ):
            return None
        return MemoryCandidate(
            key=key,
            value=value,
            category=category,
            explicit=explicit,
            confidence=max(0.0, min(1.0, confidence)),
            importance=0.8 if category in {
                MemoryCategory.USER_POLICY,
                MemoryCategory.SAFETY_CONSTRAINT,
            } else 0.6,
            metadata={
                "source": "semantic_model",
                "scope": scope,
                "evidence": evidence,
                "durability": "long_term",
            },
            operation=operation,
        )


class HybridMemoryExtractor:
    """Prefer schema-validated semantic extraction and retain rules as fallback."""

    def __init__(
        self,
        *,
        rule_extractor: Optional[RuleBasedMemoryExtractor] = None,
        semantic_extractor: Optional[StructuredMemoryExtractor] = None,
    ) -> None:
        self.rule_extractor = rule_extractor or RuleBasedMemoryExtractor()
        self.semantic_extractor = semantic_extractor

    def is_sensitive(self, text: str) -> bool:
        return self.rule_extractor.is_sensitive(text)

    def is_unsafe_policy(self, text: str) -> bool:
        return self.rule_extractor.is_unsafe_policy(text)

    def is_explicit_command(self, text: str) -> bool:
        return self.rule_extractor.is_explicit_command(text)

    def forget_payload(self, text: str) -> Optional[str]:
        return self.rule_extractor.forget_payload(text)

    def resolves_open_item(self, text: str) -> bool:
        return self.rule_extractor.resolves_open_item(text)

    def extract(self, user_message: str) -> List[MemoryCandidate]:
        if self.semantic_extractor is not None:
            semantic, authoritative = self.semantic_extractor.extract_with_status(
                user_message
            )
            if authoritative:
                return semantic
        return self.rule_extractor.extract(user_message)


class LongTermMemoryService:
    def __init__(
        self,
        store: MemoryStore,
        extractor=None,
        search_index=None,
    ) -> None:
        self.store = store
        self.extractor = extractor or RuleBasedMemoryExtractor()
        self.search_index = search_index
        self._forget_listeners = []

    def add_forget_listener(self, listener) -> None:
        self._forget_listeners.append(listener)

    def remember(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        value: str,
        category: MemoryCategory,
        *,
        importance: float = 0.5,
        confidence: float = 1.0,
        explicit: bool = True,
        source_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> MemoryRecord:
        self._require_owner(tenant_id, user_id)
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key or not normalized_value:
            raise ValueError("memory key and value are required")
        if self.extractor.is_sensitive(f"{normalized_key} {normalized_value}"):
            raise ValueError("sensitive data is not allowed in long-term memory")
        normalized_category = MemoryCategory(category)
        if normalized_category == MemoryCategory.USER_POLICY:
            if not re.fullmatch(r"policy\.[a-z0-9_.-]{2,113}", normalized_key):
                raise ValueError("unsupported user policy")
            if self.extractor.is_unsafe_policy(normalized_value):
                raise ValueError("user policy cannot override system or safety controls")
            max_length = 20 if normalized_key == "policy.response_prefix" else 1000
            if len(normalized_value) > max_length or any(
                char in normalized_value for char in "\r\n\x00"
            ):
                raise ValueError("unsafe response prefix policy")
        if self.store.has_tombstone(tenant_id, user_id, normalized_key, normalized_value):
            if not explicit:
                raise ValueError("automatically extracted memory was previously forgotten")
            self.store.clear_tombstone(tenant_id, user_id, normalized_key, normalized_value)

        current_time = now or utc_now()
        record = self.store.write_fact(
            tenant_id,
            user_id,
            normalized_key,
            normalized_value,
            normalized_category,
            self._bounded(importance),
            self._bounded(confidence),
            explicit,
            source_event_id,
            dict(metadata or {}),
            current_time,
        )
        if self.search_index is not None:
            try:
                if record.supersedes_id:
                    self.search_index.delete([record.supersedes_id])
                self.search_index.upsert(record)
            except Exception:
                LOGGER.exception(
                    "memory vector index update failed; relational write remains authoritative"
                )
        if explicit:
            self._resolve_pending_candidates(
                tenant_id, user_id, normalized_key, normalized_value, current_time
            )
        return record

    def list_memories(
        self,
        tenant_id: str,
        user_id: str,
        *,
        include_inactive: bool = False,
    ) -> List[MemoryRecord]:
        self._require_owner(tenant_id, user_id)
        self.refresh_lifecycle(tenant_id, user_id)
        return self.store.list_facts(tenant_id, user_id, include_inactive)

    def refresh_lifecycle(
        self,
        tenant_id: str,
        user_id: str,
        *,
        now: Optional[datetime] = None,
        stale_threshold: Optional[float] = None,
    ) -> int:
        self._require_owner(tenant_id, user_id)
        current_time = now or utc_now()
        threshold = (
            float(os.getenv("AGENT_MEMORY_STALE_THRESHOLD", "0.15"))
            if stale_threshold is None
            else stale_threshold
        )
        changed = 0
        for memory in self.store.list_facts(
            tenant_id, user_id, include_inactive=False
        ):
            if HALF_LIFE_DAYS[memory.category] is None:
                continue
            if calculate_time_decay(
                memory.category, memory.last_confirmed_at, current_time
            ) < threshold:
                self.store.mark_fact_status(memory.memory_id, "stale", current_time)
                metrics_registry.inc_counter(
                    "agent_memory_lifecycle_total",
                    {"transition": "active_to_stale", "category": memory.category.value},
                )
                changed += 1
        return changed

    def forget(
        self,
        tenant_id: str,
        user_id: str,
        *,
        key: Optional[str] = None,
    ) -> int:
        self._require_owner(tenant_id, user_id)
        existing = self.store.list_facts(tenant_id, user_id, include_inactive=True)
        targets = [memory for memory in existing if key is None or memory.key == key]
        deleted = self.store.forget_facts(tenant_id, user_id, key)
        if self.search_index is not None:
            try:
                self.search_index.delete([memory.memory_id for memory in targets])
            except Exception:
                LOGGER.exception(
                    "memory vector index delete failed; relational delete remains authoritative"
                )
        for listener in self._forget_listeners:
            listener(tenant_id, user_id)
        return deleted

    def recall(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        *,
        limit: int = 8,
        per_category_limit: int = 3,
        min_score: Optional[float] = None,
        min_relevance: Optional[float] = None,
        stale_threshold: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> List[ScoredMemory]:
        self._require_owner(tenant_id, user_id)
        current_time = now or utc_now()
        self.refresh_lifecycle(
            tenant_id,
            user_id,
            now=current_time,
            stale_threshold=stale_threshold,
        )
        memories = self.store.list_facts(tenant_id, user_id, include_inactive=False)
        event_loader = getattr(self.store, "list_events", None)
        if callable(event_loader):
            memories.extend(event_loader(tenant_id, user_id, max(limit * 10, 50)))
        semantic_candidate_ids: set[str] = set()
        if self.search_index is not None and query.strip():
            try:
                candidate_ids = self.search_index.query(
                    tenant_id, user_id, query, max(limit * 5, 20)
                )
                semantic_candidate_ids = set(candidate_ids)
                indexed = {memory.memory_id: memory for memory in memories}
                indexed_facts = [
                    indexed[memory_id] for memory_id in candidate_ids if memory_id in indexed
                ]
                episodes = [
                    memory for memory in memories if memory.category == MemoryCategory.EPISODIC
                ]
                memories = indexed_facts + episodes
            except Exception:
                LOGGER.exception("memory vector candidate query failed; using relational fallback")
        scored: List[ScoredMemory] = []
        effective_min_score = (
            float(os.getenv("AGENT_MEMORY_MIN_SCORE", "0.35"))
            if min_score is None
            else min_score
        )
        effective_min_relevance = (
            float(os.getenv("AGENT_MEMORY_MIN_RELEVANCE", "0.08"))
            if min_relevance is None
            else min_relevance
        )
        effective_stale_threshold = (
            float(os.getenv("AGENT_MEMORY_STALE_THRESHOLD", "0.15"))
            if stale_threshold is None
            else stale_threshold
        )
        for memory in memories:
            if memory.valid_to and current_time >= memory.valid_to:
                continue
            if memory.category == MemoryCategory.USER_POLICY:
                continue
            relevance = self._relevance(query, self._memory_search_text(memory))
            if memory.memory_id in semantic_candidate_ids:
                relevance = max(relevance, 0.5)
            if memory.category in {
                MemoryCategory.OPEN_ITEM,
                MemoryCategory.SAFETY_CONSTRAINT,
            }:
                relevance = 1.0
            recency = calculate_time_decay(
                memory.category, memory.last_confirmed_at, current_time
            )
            if recency < effective_stale_threshold:
                continue
            if (
                memory.category
                not in {MemoryCategory.OPEN_ITEM, MemoryCategory.SAFETY_CONSTRAINT}
                and relevance < effective_min_relevance
            ):
                continue
            score = (
                0.45 * relevance
                + 0.20 * recency
                + 0.20 * memory.importance
                + 0.10 * memory.confidence
                + 0.05 * min(memory.reinforcement, 2.0) / 2.0
            )
            if score < effective_min_score:
                continue
            scored.append(
                ScoredMemory(
                    memory=memory,
                    score=round(score, 6),
                    relevance=relevance,
                    recency=recency,
                )
            )
        scored.sort(key=lambda item: (item.score, item.memory.updated_at), reverse=True)

        selected: List[ScoredMemory] = []
        category_counts: Dict[MemoryCategory, int] = {}
        for item in scored:
            count = category_counts.get(item.memory.category, 0)
            if count >= per_category_limit:
                continue
            selected.append(item)
            category_counts[item.memory.category] = count + 1
            self.store.log_access(
                item.memory.memory_id, tenant_id, user_id, item.score
            )
            if len(selected) >= limit:
                break
        metrics_registry.inc_counter(
            "agent_memory_recall_total",
            {"status": "hit" if selected else "miss"},
        )
        if selected:
            metrics_registry.inc_counter(
                "agent_memory_recalled_items_total", value=len(selected)
            )
        return selected

    def rebuild_search_index(self, tenant_id: str, user_id: str) -> int:
        self._require_owner(tenant_id, user_id)
        if self.search_index is None:
            return 0
        memories = self.store.list_facts(
            tenant_id, user_id, include_inactive=False
        )
        for memory in memories:
            self.search_index.upsert(memory)
        return len(memories)

    def context_instructions(
        self,
        tenant_id: str,
        user_id: str,
        *,
        limit: int = 20,
    ) -> List[MemoryRecord]:
        """Return always-applicable user-level instructions, never system authority."""
        self._require_owner(tenant_id, user_id)
        candidates = []
        for memory in self.store.list_facts(
            tenant_id, user_id, include_inactive=False
        ):
            if memory.category == MemoryCategory.USER_POLICY:
                candidates.append(memory)
                continue
            if (
                memory.category == MemoryCategory.USER_PREFERENCE
                and memory.metadata.get("scope", "global") == "global"
            ):
                candidates.append(memory)
        candidates.sort(
            key=lambda item: (
                item.category == MemoryCategory.USER_POLICY,
                item.importance,
                item.updated_at,
            ),
            reverse=True,
        )
        return candidates[: max(0, limit)]

    def review_pending(
        self,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        decision: str,
    ) -> MemoryRecord:
        self._require_owner(tenant_id, user_id)
        normalized_decision = decision.strip().lower()
        if normalized_decision not in {"accept", "reject"}:
            raise ValueError("decision must be accept or reject")
        pending = next(
            (
                memory
                for memory in self.store.list_facts(
                    tenant_id, user_id, include_inactive=True
                )
                if memory.memory_id == memory_id
                and memory.status == "pending_confirmation"
            ),
            None,
        )
        if pending is None:
            raise KeyError(memory_id)
        if normalized_decision == "reject":
            rejected = self.store.mark_fact_status(
                pending.memory_id, "rejected", utc_now()
            )
            metrics_registry.inc_counter(
                "agent_memory_conflict_total", {"status": "rejected"}
            )
            return rejected
        return self.remember(
            tenant_id,
            user_id,
            pending.key,
            pending.value,
            pending.category,
            importance=pending.importance,
            confidence=1.0,
            explicit=True,
            source_event_id=pending.source_event_id,
            metadata={
                **pending.metadata,
                "review_required": False,
                "review_decision": "accepted",
            },
        )

    def handle_explicit_command(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        request_id: str,
        user_message: str,
    ) -> MemoryCommandResult:
        self._require_owner(tenant_id, user_id)
        if not self.extractor.is_explicit_command(user_message):
            return MemoryCommandResult(handled=False)
        if self.extractor.is_sensitive(user_message):
            return MemoryCommandResult(
                handled=True,
                status="rejected",
                action="remember",
                message="这项内容包含敏感信息，未写入长期记忆。",
                rejected_reason="sensitive_data",
            )

        if self.extractor.resolves_open_item(user_message):
            current = self.store.get_active_fact(
                tenant_id, user_id, "open_item.current"
            )
            if current is None:
                return MemoryCommandResult(
                    handled=True,
                    status="not_found",
                    action="resolve",
                    message="当前没有待处理事项可标记为已解决。",
                    rejected_reason="open_item_not_found",
                )
            resolved = self.store.mark_fact_status(
                current.memory_id, "resolved", utc_now()
            )
            return MemoryCommandResult(
                handled=True,
                status="resolved",
                action="resolve",
                message="已将当前待处理事项标记为已解决。",
                records=[resolved],
            )

        forget_payload = self.extractor.forget_payload(user_message)
        if forget_payload is not None:
            normalized = forget_payload.strip()
            if normalized in {"所有记忆", "全部记忆", "我的所有记忆", "我的全部记忆"}:
                deleted = self.forget(tenant_id, user_id)
                return MemoryCommandResult(
                    handled=True,
                    status="deleted",
                    action="forget",
                    message=f"已删除你的全部记忆，共清理 {deleted} 条关联记录。",
                    deleted=deleted,
                )
            candidates = self.extractor.extract(user_message)
            if not candidates:
                candidates = self.extractor.extract(normalized)
            keys, ambiguous = self._resolve_forget_keys(
                tenant_id,
                user_id,
                normalized,
                candidates,
            )
            if ambiguous:
                return MemoryCommandResult(
                    handled=True,
                    status="ambiguous",
                    action="forget",
                    message="找到多条可能匹配的记忆。为避免误删，请说明具体内容或记忆项。",
                    rejected_reason="ambiguous_memory_selector",
                )
            if not keys:
                return MemoryCommandResult(
                    handled=True,
                    status="not_found",
                    action="forget",
                    message="没有找到可明确删除的对应记忆，未执行删除。",
                    rejected_reason="memory_not_found",
                )
            deleted = sum(self.forget(tenant_id, user_id, key=key) for key in keys)
            return MemoryCommandResult(
                handled=True,
                status="deleted",
                action="forget",
                message=f"已删除 {deleted} 条匹配记忆。",
                deleted=deleted,
            )

        candidates = self.extractor.extract(user_message)
        if not candidates:
            return MemoryCommandResult(
                handled=True,
                status="rejected",
                action="remember",
                message="这项内容暂时无法安全归类，因此没有写入长期记忆。",
                rejected_reason="unsupported_memory_schema",
            )
        event_id = self.store.append_event(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "request_id": request_id,
                "kind": MemoryCategory.EPISODIC.value,
                "content": user_message,
                "metadata": {
                    "memory_operation": {
                        "status": "processing",
                        "action": "remember",
                        "keys": [candidate.key for candidate in candidates],
                    }
                },
            }
        )
        saved: List[MemoryRecord] = []
        failures: List[str] = []
        for candidate in candidates:
            if candidate.operation != "upsert":
                failures.append(
                    f"{candidate.key}: unsupported operation {candidate.operation}"
                )
                continue
            try:
                saved.append(
                    self.remember(
                        tenant_id,
                        user_id,
                        candidate.key,
                        candidate.value,
                        candidate.category,
                        importance=candidate.importance,
                        confidence=candidate.confidence,
                        explicit=True,
                        source_event_id=event_id,
                        metadata=candidate.metadata,
                    )
                )
            except ValueError as exc:
                failures.append(f"{candidate.key}: {exc}")
        status = "saved" if saved and not failures else "partial" if saved else "rejected"
        self.store.append_event(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "request_id": request_id,
                "kind": MemoryCategory.EPISODIC.value,
                "content": user_message,
                "metadata": {
                    "memory_operation": {
                        "status": status,
                        "action": "remember",
                        "saved_keys": [memory.key for memory in saved],
                        "failures": failures,
                    }
                },
            }
        )
        if not saved:
            return MemoryCommandResult(
                handled=True,
                status="rejected",
                action="remember",
                message="记忆未保存：" + "；".join(failures),
                rejected_reason="write_rejected",
            )
        names = "、".join(memory.key for memory in saved)
        message = f"已写入长期记忆：{names}。"
        if failures:
            message += " 部分内容未保存：" + "；".join(failures)
        return MemoryCommandResult(
            handled=True,
            status=status,
            action="remember",
            message=message,
            records=saved,
        )

    def apply_response_policies(
        self, tenant_id: str, user_id: str, answer: str
    ) -> str:
        if not answer or not tenant_id.strip() or not user_id.strip():
            return answer
        policies = {
            memory.key: memory
            for memory in self.store.list_facts(
                tenant_id, user_id, include_inactive=False
            )
            if memory.category == MemoryCategory.USER_POLICY
        }
        prefix = policies.get("policy.response_prefix")
        if prefix is None or answer.startswith(prefix.value):
            return answer
        return f"{prefix.value}，{answer}"

    def process_turn(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        request_id: str,
        user_message: str,
        assistant_message: str,
    ) -> List[MemoryRecord]:
        self._require_owner(tenant_id, user_id)
        if self.extractor.is_sensitive(user_message):
            return []
        event_id = self.store.append_event(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "request_id": request_id,
                "kind": MemoryCategory.EPISODIC.value,
                "content": user_message,
                "metadata": {"assistant_message": assistant_message},
            }
        )
        saved: List[MemoryRecord] = []
        failures: List[str] = []
        for candidate in self.extractor.extract(user_message):
            if candidate.operation != "upsert":
                continue
            try:
                saved.append(
                    self.remember(
                        tenant_id,
                        user_id,
                        candidate.key,
                        candidate.value,
                        candidate.category,
                        importance=candidate.importance,
                        confidence=candidate.confidence,
                        explicit=candidate.explicit,
                        source_event_id=event_id,
                        metadata=candidate.metadata,
                    )
                )
            except ValueError as exc:
                if "conflicts with active memory" in str(exc):
                    pending = self._stage_conflict(
                        tenant_id,
                        user_id,
                        candidate,
                        event_id,
                    )
                    failures.append(
                        f"{candidate.key}: requires_confirmation:{pending.memory_id}"
                    )
                else:
                    failures.append(f"{candidate.key}: {exc}")
        needs_confirmation = any(
            "requires_confirmation:" in failure for failure in failures
        )
        operation_status = (
            "partial"
            if saved and failures
            else "saved"
            if saved
            else "needs_confirmation"
            if needs_confirmation
            else "no_candidate"
        )
        self.store.append_event(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "request_id": request_id,
                "kind": MemoryCategory.EPISODIC.value,
                "content": user_message,
                "metadata": {
                    "assistant_message": assistant_message,
                    "memory_operation": {
                        "status": operation_status,
                        "saved_keys": [memory.key for memory in saved],
                        "failures": failures,
                    },
                },
            }
        )
        metrics_registry.inc_counter(
            "agent_memory_extraction_total",
            {"status": operation_status},
        )
        return saved

    def _stage_conflict(
        self,
        tenant_id: str,
        user_id: str,
        candidate: MemoryCandidate,
        source_event_id: str,
    ) -> MemoryRecord:
        existing = self.store.get_active_fact(
            tenant_id, user_id, candidate.key
        )
        if existing is None:
            raise ValueError("conflicting active memory no longer exists")
        for record in self.store.list_facts(
            tenant_id, user_id, include_inactive=True
        ):
            if (
                record.key == candidate.key
                and record.value == candidate.value
                and record.status == "pending_confirmation"
            ):
                return record
        current_time = utc_now()
        pending = MemoryRecord(
            memory_id=str(uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            key=candidate.key,
            value=candidate.value,
            category=candidate.category,
            status="pending_confirmation",
            version=existing.version + 1,
            importance=candidate.importance,
            confidence=candidate.confidence,
            reinforcement=0.0,
            explicit=False,
            created_at=current_time,
            updated_at=current_time,
            last_confirmed_at=current_time,
            valid_from=current_time,
            supersedes_id=existing.memory_id,
            source_event_id=source_event_id,
            metadata={
                **candidate.metadata,
                "conflicts_with": existing.memory_id,
                "existing_value": existing.value,
                "review_required": True,
            },
        )
        self.store.save_fact(pending)
        metrics_registry.inc_counter(
            "agent_memory_conflict_total",
            {"status": "pending_confirmation"},
        )
        return pending

    def _resolve_pending_candidates(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        accepted_value: str,
        changed_at: datetime,
    ) -> None:
        for memory in self.store.list_facts(
            tenant_id, user_id, include_inactive=True
        ):
            if memory.key != key or memory.status != "pending_confirmation":
                continue
            status = "accepted" if memory.value == accepted_value else "rejected"
            self.store.mark_fact_status(memory.memory_id, status, changed_at)
            metrics_registry.inc_counter(
                "agent_memory_conflict_total", {"status": status}
            )

    def _resolve_forget_keys(
        self,
        tenant_id: str,
        user_id: str,
        selector: str,
        candidates: List[MemoryCandidate],
    ) -> tuple[set[str], bool]:
        memories = self.store.list_facts(
            tenant_id, user_id, include_inactive=False
        )
        active_by_key = {memory.key: memory for memory in memories}
        candidate_keys = {
            candidate.key for candidate in candidates if candidate.key in active_by_key
        }
        if candidate_keys:
            return candidate_keys, False

        normalized = selector.strip()
        exact = {
            memory.key
            for memory in memories
            if normalized in {memory.key, memory.value}
            or (len(memory.value) >= 3 and memory.value in normalized)
        }
        if exact:
            return exact, len(exact) > 1

        broad_category = None
        if "偏好" in normalized or "习惯" in normalized:
            broad_category = MemoryCategory.USER_PREFERENCE
        elif "规则" in normalized or "要求" in normalized or "指令" in normalized:
            broad_category = MemoryCategory.USER_POLICY
        if broad_category is not None:
            matches = {
                memory.key for memory in memories if memory.category == broad_category
            }
            if len(matches) != 1:
                return set(), len(matches) > 1
            return matches, False

        scored = sorted(
            (
                (
                    self._relevance(normalized, self._memory_search_text(memory)),
                    memory.key,
                )
                for memory in memories
            ),
            reverse=True,
        )
        if not scored or scored[0][0] < 0.35:
            return set(), False
        if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.08:
            return set(), True
        return {scored[0][1]}, False

    def run_retention(self) -> Dict[str, Any]:
        result = self.store.prune_retention(
            raw_message_days=90,
            episodic_days=180,
            superseded_fact_days=365,
            access_log_days=90,
            procedure_candidate_days=30,
        )
        deleted_ids = result.pop("deleted_memory_ids", [])
        if self.search_index is not None:
            self.search_index.delete(deleted_ids)
        return result

    def propose_procedure(
        self,
        title: str,
        content: str,
        *,
        agent_version: str,
        tenant_id: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> ProcedureMemory:
        if not title.strip() or not content.strip() or not agent_version.strip():
            raise ValueError("title, content and agent_version are required")
        procedure = ProcedureMemory(
            procedure_id=str(uuid4()),
            tenant_id=tenant_id,
            agent_version=agent_version,
            status="candidate",
            title=title.strip(),
            content=content.strip(),
            evidence=dict(evidence or {}),
            created_at=utc_now(),
        )
        self.store.save_procedure(procedure)
        return procedure

    def approve_procedure(self, procedure_id: str) -> ProcedureMemory:
        return self.store.approve_procedure(procedure_id, utc_now())

    def list_procedures(
        self, tenant_id: Optional[str], status: str = "approved"
    ) -> List[ProcedureMemory]:
        return self.store.list_procedures(tenant_id, status)

    @staticmethod
    def _require_owner(tenant_id: str, user_id: str) -> None:
        if not tenant_id.strip() or not user_id.strip():
            raise ValueError("tenant_id and user_id are required for long-term memory")

    @staticmethod
    def _bounded(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _relevance(query: str, text: str) -> float:
        def terms(value: str) -> set[str]:
            normalized = re.sub(r"\s+", "", value.lower())
            chars = {char for char in normalized if char.isalnum() or "\u4e00" <= char <= "\u9fff"}
            bigrams = {normalized[i : i + 2] for i in range(max(0, len(normalized) - 1))}
            return chars | bigrams

        query_terms = terms(query)
        if not query_terms:
            return 0.0
        overlap = query_terms & terms(text)
        return round(len(overlap) / len(query_terms), 6)

    @staticmethod
    def _memory_search_text(memory: MemoryRecord) -> str:
        aliases = {
            "device.model": "设备型号 扫地机器人型号",
            "profile.city": "居住城市 地址 城市",
            "preference.general": "用户偏好 喜欢",
            "profile.home_area": "家庭面积 房屋面积",
            "profile.pets": "宠物 家庭宠物",
            "device.current_state": "设备状态 故障",
            "open_item.current": "待处理事项 未解决问题",
            "safety.confirmation_required": "确认要求 安全限制",
        }
        return f"{memory.key} {aliases.get(memory.key, '')} {memory.value}"


def stable_value_hash(key: str, value: str) -> str:
    return hashlib.sha256(f"{key}\0{value}".encode("utf-8")).hexdigest()

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4


class MemoryCategory(str, Enum):
    TRANSIENT = "transient"
    EPISODIC = "episodic"
    DEVICE_STATE = "device_state"
    DEVICE_IDENTITY = "device_identity"
    USER_PREFERENCE = "user_preference"
    STABLE_PROFILE = "stable_profile"
    OPEN_ITEM = "open_item"
    SAFETY_CONSTRAINT = "safety_constraint"


HALF_LIFE_DAYS: Dict[MemoryCategory, Optional[float]] = {
    MemoryCategory.TRANSIENT: 1.0,
    MemoryCategory.EPISODIC: 30.0,
    MemoryCategory.DEVICE_STATE: 30.0,
    MemoryCategory.DEVICE_IDENTITY: 180.0,
    MemoryCategory.USER_PREFERENCE: 180.0,
    MemoryCategory.STABLE_PROFILE: 365.0,
    MemoryCategory.OPEN_ITEM: None,
    MemoryCategory.SAFETY_CONSTRAINT: None,
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


class MemoryStore(Protocol):
    def get_active_fact(self, tenant_id: str, user_id: str, key: str) -> Optional[MemoryRecord]: ...
    def save_fact(self, memory: MemoryRecord, supersede_id: Optional[str] = None) -> None: ...
    def confirm_fact(self, memory_id: str, confirmed_at: datetime) -> MemoryRecord: ...
    def list_facts(
        self, tenant_id: str, user_id: str, include_inactive: bool = False
    ) -> List[MemoryRecord]: ...
    def forget_facts(self, tenant_id: str, user_id: str, key: Optional[str] = None) -> int: ...
    def has_tombstone(self, tenant_id: str, user_id: str, key: str, value: str) -> bool: ...
    def clear_tombstone(self, tenant_id: str, user_id: str, key: str, value: str) -> None: ...
    def append_event(self, event: Dict[str, Any]) -> str: ...
    def list_events(self, tenant_id: str, user_id: str, limit: int = 100) -> List[MemoryRecord]: ...
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
    _EXPLICIT_PREFIX = re.compile(r"^(?:请)?记住(?:一下)?[：,:，\s]*")
    _RULES = (
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
    )

    @classmethod
    def is_sensitive(cls, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in cls._SENSITIVE_TERMS)

    def extract(self, user_message: str) -> List[MemoryCandidate]:
        text = user_message.strip().rstrip("。！!")
        if self.is_sensitive(text):
            return []
        explicit = bool(self._EXPLICIT_PREFIX.match(text))
        if explicit:
            text = self._EXPLICIT_PREFIX.sub("", text, count=1).strip()
        for pattern, key, category in self._RULES:
            match = pattern.match(text)
            if not match:
                continue
            value = match.group(1).strip()
            if not value:
                return []
            return [
                MemoryCandidate(
                    key=key,
                    value=value,
                    category=category,
                    explicit=explicit,
                    confidence=1.0 if explicit else 0.9,
                )
            ]
        return []


class LongTermMemoryService:
    def __init__(
        self,
        store: MemoryStore,
        extractor: Optional[RuleBasedMemoryExtractor] = None,
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
        if self.store.has_tombstone(tenant_id, user_id, normalized_key, normalized_value):
            if not explicit:
                raise ValueError("automatically extracted memory was previously forgotten")
            self.store.clear_tombstone(tenant_id, user_id, normalized_key, normalized_value)

        current_time = now or utc_now()
        existing = self.store.get_active_fact(tenant_id, user_id, normalized_key)
        if existing and existing.value == normalized_value:
            return self.store.confirm_fact(existing.memory_id, current_time)
        if existing and not explicit:
            raise ValueError("automatically extracted fact conflicts with active memory")

        record = MemoryRecord(
            memory_id=str(uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            key=normalized_key,
            value=normalized_value,
            category=MemoryCategory(category),
            status="active",
            version=(existing.version + 1) if existing else 1,
            importance=self._bounded(importance),
            confidence=self._bounded(confidence),
            reinforcement=1.0,
            explicit=explicit,
            created_at=current_time,
            updated_at=current_time,
            last_confirmed_at=current_time,
            valid_from=current_time,
            supersedes_id=existing.memory_id if existing else None,
            source_event_id=source_event_id,
            metadata=dict(metadata or {}),
        )
        self.store.save_fact(record, supersede_id=record.supersedes_id)
        if self.search_index is not None:
            if record.supersedes_id:
                self.search_index.delete([record.supersedes_id])
            self.search_index.upsert(record)
        return record

    def list_memories(
        self,
        tenant_id: str,
        user_id: str,
        *,
        include_inactive: bool = False,
    ) -> List[MemoryRecord]:
        self._require_owner(tenant_id, user_id)
        return self.store.list_facts(tenant_id, user_id, include_inactive)

    def forget(
        self,
        tenant_id: str,
        user_id: str,
        *,
        key: Optional[str] = None,
    ) -> int:
        self._require_owner(tenant_id, user_id)
        active = self.store.list_facts(tenant_id, user_id, include_inactive=False)
        targets = [memory for memory in active if key is None or memory.key == key]
        deleted = self.store.forget_facts(tenant_id, user_id, key)
        if self.search_index is not None:
            self.search_index.delete([memory.memory_id for memory in targets])
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
        now: Optional[datetime] = None,
    ) -> List[ScoredMemory]:
        self._require_owner(tenant_id, user_id)
        current_time = now or utc_now()
        memories = self.store.list_facts(tenant_id, user_id, include_inactive=False)
        event_loader = getattr(self.store, "list_events", None)
        if callable(event_loader):
            memories.extend(event_loader(tenant_id, user_id, max(limit * 10, 50)))
        if self.search_index is not None and query.strip():
            try:
                candidate_ids = self.search_index.query(
                    tenant_id, user_id, query, max(limit * 5, 20)
                )
                indexed = {memory.memory_id: memory for memory in memories}
                indexed_facts = [
                    indexed[memory_id] for memory_id in candidate_ids if memory_id in indexed
                ]
                episodes = [
                    memory for memory in memories if memory.category == MemoryCategory.EPISODIC
                ]
                memories = indexed_facts + episodes
            except Exception:
                pass
        scored: List[ScoredMemory] = []
        for memory in memories:
            if memory.valid_to and current_time >= memory.valid_to:
                continue
            relevance = self._relevance(query, f"{memory.key} {memory.value}")
            if memory.category in {
                MemoryCategory.OPEN_ITEM,
                MemoryCategory.SAFETY_CONSTRAINT,
            }:
                relevance = 1.0
            recency = calculate_time_decay(
                memory.category, memory.last_confirmed_at, current_time
            )
            score = (
                0.45 * relevance
                + 0.20 * recency
                + 0.20 * memory.importance
                + 0.10 * memory.confidence
                + 0.05 * min(memory.reinforcement, 2.0) / 2.0
            )
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
        return selected

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
        for candidate in self.extractor.extract(user_message):
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
            except ValueError:
                continue
        return saved

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


def stable_value_hash(key: str, value: str) -> str:
    return hashlib.sha256(f"{key}\0{value}".encode("utf-8")).hexdigest()

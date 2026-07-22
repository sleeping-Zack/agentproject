import sqlite3
from datetime import datetime, timedelta, timezone

from agent.long_term_memory import (
    MemoryCategory,
    MemoryCandidate,
    LongTermMemoryService,
    RuleBasedMemoryExtractor,
    calculate_time_decay,
)
from services.memory_store import SQLiteMemoryStore


def _service(tmp_path):
    return LongTermMemoryService(SQLiteMemoryStore(str(tmp_path / "memory.db")))


def test_memory_is_isolated_by_tenant_and_user(tmp_path):
    service = _service(tmp_path)
    service.remember(
        "tenant-a", "user-1", "device.model", "S10", MemoryCategory.DEVICE_IDENTITY
    )
    service.remember(
        "tenant-a", "user-2", "device.model", "X20", MemoryCategory.DEVICE_IDENTITY
    )
    service.remember(
        "tenant-b", "user-1", "device.model", "Q5", MemoryCategory.DEVICE_IDENTITY
    )

    assert [m.value for m in service.list_memories("tenant-a", "user-1")] == ["S10"]
    assert [m.value for m in service.list_memories("tenant-a", "user-2")] == ["X20"]
    assert [m.value for m in service.list_memories("tenant-b", "user-1")] == ["Q5"]


def test_correcting_fact_versions_the_old_value(tmp_path):
    service = _service(tmp_path)
    old = service.remember(
        "tenant-a", "user-1", "profile.city", "深圳", MemoryCategory.STABLE_PROFILE
    )
    current = service.remember(
        "tenant-a", "user-1", "profile.city", "上海", MemoryCategory.STABLE_PROFILE
    )

    assert current.version == 2
    assert current.supersedes_id == old.memory_id
    assert [m.value for m in service.list_memories("tenant-a", "user-1")] == ["上海"]
    historical = service.list_memories("tenant-a", "user-1", include_inactive=True)
    assert [(m.value, m.status) for m in historical] == [
        ("深圳", "superseded"),
        ("上海", "active"),
    ]


def test_automatic_conflict_does_not_replace_confirmed_fact(tmp_path):
    service = _service(tmp_path)
    service.remember(
        "tenant-a", "user-1", "profile.city", "深圳", MemoryCategory.STABLE_PROFILE
    )

    try:
        service.remember(
            "tenant-a",
            "user-1",
            "profile.city",
            "上海",
            MemoryCategory.STABLE_PROFILE,
            explicit=False,
        )
    except ValueError as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("automatic conflict must require confirmation")

    assert service.list_memories("tenant-a", "user-1")[0].value == "深圳"


def test_half_life_decay_uses_last_confirmation_time():
    now = datetime(2026, 7, 22, tzinfo=timezone.utc)
    confirmed_at = now - timedelta(days=180)

    score = calculate_time_decay(
        MemoryCategory.USER_PREFERENCE,
        last_confirmed_at=confirmed_at,
        now=now,
    )

    assert score == 0.5


def test_recall_applies_relevance_decay_and_category_quota(tmp_path):
    service = _service(tmp_path)
    service.remember(
        "tenant-a",
        "user-1",
        "device.model",
        "云鲸 S10 扫地机器人",
        MemoryCategory.DEVICE_IDENTITY,
        importance=0.9,
    )
    for index in range(4):
        service.remember(
            "tenant-a",
            "user-1",
            f"episode.{index}",
            f"第 {index} 次清理滚刷",
            MemoryCategory.EPISODIC,
        )

    recalled = service.recall(
        "tenant-a",
        "user-1",
        "我的扫地机器人是什么型号",
        limit=4,
        per_category_limit=2,
    )

    assert recalled[0].memory.key == "device.model"
    assert sum(m.memory.category == MemoryCategory.EPISODIC for m in recalled) == 2


def test_forget_removes_active_fact_and_creates_tombstone(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    service = LongTermMemoryService(store)
    service.remember(
        "tenant-a", "user-1", "profile.city", "深圳", MemoryCategory.STABLE_PROFILE
    )

    assert service.forget("tenant-a", "user-1", key="profile.city") == 1
    assert service.list_memories("tenant-a", "user-1") == []
    assert store.has_tombstone("tenant-a", "user-1", "profile.city", "深圳")


def test_extractor_prefers_explicit_memory_and_blocks_sensitive_data():
    extractor = RuleBasedMemoryExtractor()

    explicit = extractor.extract("请记住我的扫地机器人型号是 S10")
    sensitive = extractor.extract("请记住我的密码是 123456")
    automatic = extractor.extract("我住在深圳")

    assert explicit == [
        MemoryCandidate(
            key="device.model",
            value="S10",
            category=MemoryCategory.DEVICE_IDENTITY,
            explicit=True,
            confidence=1.0,
        )
    ]
    assert sensitive == []
    assert automatic[0].key == "profile.city"
    assert automatic[0].explicit is False


def test_sensitive_data_is_rejected_even_through_direct_memory_api(tmp_path):
    service = _service(tmp_path)

    try:
        service.remember(
            "tenant-a", "user-1", "account.password", "123456", MemoryCategory.STABLE_PROFILE
        )
    except ValueError as exc:
        assert "sensitive data" in str(exc)
    else:
        raise AssertionError("sensitive memory must be rejected")

    assert service.process_turn(
        "tenant-a", "user-1", "session-1", "request-1", "我的密码是 123456", "收到"
    ) == []
    assert service.recall("tenant-a", "user-1", "密码") == []


def test_summary_is_persisted_with_covered_message_count(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    store.save_summary("tenant-a", "session-1", "用户关注滚刷清理", 12, "summary-v1")

    fresh_store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    summary = fresh_store.load_summary("tenant-a", "session-1")

    assert summary == {
        "summary": "用户关注滚刷清理",
        "covered_message_count": 12,
        "version": "summary-v1",
    }


def test_vector_index_is_only_a_candidate_layer_and_is_updated(tmp_path):
    class FakeIndex:
        def __init__(self):
            self.memories = []
            self.deleted = []

        def upsert(self, memory):
            self.memories.append(memory)

        def delete(self, memory_ids):
            self.deleted.extend(memory_ids)

        def query(self, tenant_id, user_id, text, limit):
            return ["foreign-id", *[memory.memory_id for memory in self.memories]]

    index = FakeIndex()
    service = LongTermMemoryService(
        SQLiteMemoryStore(str(tmp_path / "memory.db")), search_index=index
    )
    first = service.remember(
        "tenant-a", "user-1", "device.model", "S10", MemoryCategory.DEVICE_IDENTITY
    )
    current = service.remember(
        "tenant-a", "user-1", "device.model", "S20", MemoryCategory.DEVICE_IDENTITY
    )

    recalled = service.recall("tenant-a", "user-1", "设备型号")

    assert first.memory_id in index.deleted
    assert [item.memory.memory_id for item in recalled] == [current.memory_id]


def test_retention_prunes_superseded_facts_but_never_active_facts(tmp_path):
    db_path = str(tmp_path / "memory.db")
    service = LongTermMemoryService(SQLiteMemoryStore(db_path))
    old = service.remember(
        "tenant-a", "user-1", "device.model", "S10", MemoryCategory.DEVICE_IDENTITY
    )
    current = service.remember(
        "tenant-a", "user-1", "device.model", "S20", MemoryCategory.DEVICE_IDENTITY
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE memory_facts SET updated_at = '2020-01-01T00:00:00+00:00' "
            "WHERE memory_id IN (?, ?)",
            (old.memory_id, current.memory_id),
        )

    result = service.run_retention()

    assert result["superseded_facts"] == 1
    remaining = service.list_memories("tenant-a", "user-1", include_inactive=True)
    assert [memory.memory_id for memory in remaining] == [current.memory_id]


def test_procedural_memory_requires_approval_before_use(tmp_path):
    service = _service(tmp_path)
    candidate = service.propose_procedure(
        "清理滚刷",
        "断电后拆下滚刷并清理缠绕物",
        agent_version="agent-v1",
        tenant_id="tenant-a",
        evidence={"request_id": "request-1"},
    )

    assert service.list_procedures("tenant-a") == []
    approved = service.approve_procedure(candidate.procedure_id)

    assert approved.status == "approved"
    assert service.list_procedures("tenant-a") == [approved]

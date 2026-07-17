import os
from uuid import uuid4

import pytest

from observability.event_bus import EventStreamConflictError, RedisEventBus
from services.postgres import (
    PostgresApprovalStore,
    PostgresArtifactStore,
    PostgresStore,
)
from services.rate_limit import RedisRateLimiter


REDIS_URL = os.getenv("AGENT_TEST_REDIS_URL")
POSTGRES_URL = os.getenv("AGENT_TEST_POSTGRES_URL")


@pytest.mark.skipif(not REDIS_URL, reason="AGENT_TEST_REDIS_URL is not configured")
def test_redis_event_bus_shares_stream_state_across_instances():
    request_id = f"integration-{uuid4()}"
    prefix = f"test:agent:events:{uuid4()}"
    first = RedisEventBus(REDIS_URL, key_prefix=prefix)
    second = RedisEventBus(REDIS_URL, key_prefix=prefix)
    identity = {"tenant_id": "tenant-a", "session_id": "session-a"}
    try:
        assert first.open(request_id, identity) is True
        assert second.open(request_id, identity) is False
        with pytest.raises(EventStreamConflictError):
            second.open(request_id, {"tenant_id": "tenant-b"})

        published = first.publish(request_id, "token_delta", {"delta": "A"})
        replayed = second.replay(request_id)
        assert replayed == [published]

        second.cancel(request_id)
        assert first.is_cancelled(request_id)
        first.close(request_id)
        assert second.is_closed(request_id)
    finally:
        first.discard(request_id)


@pytest.mark.skipif(not REDIS_URL, reason="AGENT_TEST_REDIS_URL is not configured")
def test_redis_rate_limiter_is_shared_across_instances():
    prefix = f"test:agent:rate:{uuid4()}"
    first = RedisRateLimiter(2, 60, REDIS_URL, key_prefix=prefix)
    second = RedisRateLimiter(2, 60, REDIS_URL, key_prefix=prefix)
    key = str(uuid4())

    assert first.allow(key)
    assert second.allow(key)
    assert not first.allow(key)


@pytest.mark.skipif(not POSTGRES_URL, reason="AGENT_TEST_POSTGRES_URL is not configured")
def test_postgres_backends_share_and_deduplicate_state():
    suffix = str(uuid4())
    session_store = PostgresStore(POSTGRES_URL)
    other_session_store = PostgresStore(POSTGRES_URL)
    request_id = f"request-{suffix}"
    session_id = f"session-{suffix}"

    assert session_store.save_session_message(
        session_id,
        "user",
        "hello",
        tenant_id="tenant-a",
        request_id=request_id,
    )
    assert not other_session_store.save_session_message(
        session_id,
        "user",
        "retry",
        tenant_id="tenant-a",
        request_id=request_id,
    )
    assert other_session_store.get_session_messages(session_id, "tenant-a") == [
        {"role": "user", "content": "hello"}
    ]

    approval_store = PostgresApprovalStore(POSTGRES_URL)
    first_approval = approval_store.create_pending(
        request_id=request_id,
        tenant_id="tenant-a",
        user_role="user",
        tool_name="fetch_external_data",
        args={"month": "2026-07"},
        reason="sensitive data",
    )
    duplicate_approval = PostgresApprovalStore(POSTGRES_URL).create_pending(
        request_id=request_id,
        tenant_id="tenant-a",
        user_role="user",
        tool_name="fetch_external_data",
        args={"month": "2026-07"},
        reason="sensitive data",
    )
    assert duplicate_approval.approval_id == first_approval.approval_id

    artifact_store = PostgresArtifactStore(POSTGRES_URL)
    first_artifact = artifact_store.save_artifact(
        request_id=request_id,
        tenant_id="tenant-a",
        artifact_type="answer",
        name="final-answer",
        payload={"answer": "ok"},
    )
    duplicate_artifact = PostgresArtifactStore(POSTGRES_URL).save_artifact(
        request_id=request_id,
        tenant_id="tenant-a",
        artifact_type="answer",
        name="final-answer",
        payload={"answer": "retry"},
    )
    assert duplicate_artifact.artifact_id == first_artifact.artifact_id

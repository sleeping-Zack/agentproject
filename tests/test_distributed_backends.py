import os
from uuid import uuid4

import pytest

from observability.event_bus import EventStreamConflictError, RedisEventBus
from services.postgres import (
    PostgresApprovalStore,
    PostgresArtifactStore,
    PostgresStore,
)
from services.postgres_evaluation import (
    PostgresEvaluationAnalysisStore,
    PostgresHumanEvalStore,
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
    import psycopg

    suffix = str(uuid4())
    session_store = PostgresStore(POSTGRES_URL)
    other_session_store = PostgresStore(POSTGRES_URL)
    request_id = f"request-{suffix}"
    session_id = f"session-{suffix}"

    try:
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
    finally:
        with psycopg.connect(POSTGRES_URL) as connection:
            for table in ("artifacts", "approvals", "session_messages"):
                connection.execute(
                    f"DELETE FROM {table} WHERE tenant_id = %s AND request_id = %s",  # noqa: S608
                    ("tenant-a", request_id),
                )


@pytest.mark.skipif(not POSTGRES_URL, reason="AGENT_TEST_POSTGRES_URL is not configured")
def test_postgres_evaluation_backends_round_trip():
    import psycopg

    suffix = str(uuid4())
    tenant_id = f"tenant-{suffix}"
    batch_id = None
    report_id = f"report-{suffix}"
    human_store = PostgresHumanEvalStore(POSTGRES_URL)
    analysis_store = PostgresEvaluationAnalysisStore(POSTGRES_URL)
    try:
        batch_id = human_store.create_batch(
            tenant_id=tenant_id,
            name="postgres-integration",
            dataset_version="v1",
            rubric_version="v1",
            assignments_per_item=1,
            qc_rate=0.0,
            seed=1,
            created_by="operator",
            items=[
                {
                    "case_id": "case-1",
                    "blind_payload": {"query": "hello"},
                    "oracle_payload": {},
                    "ordinal": 0,
                    "qc_selected": False,
                }
            ],
            assignments=[
                {
                    "case_id": "case-1",
                    "reviewer_id": "reviewer",
                    "reviewer_slot": 0,
                }
            ],
        )
        task = PostgresHumanEvalStore(POSTGRES_URL).claim_next(
            batch_id=batch_id,
            tenant_id=tenant_id,
            reviewer_id="reviewer",
        )
        assert task is not None
        human_store.submit_annotation(
            assignment_id=task["assignment_id"],
            tenant_id=tenant_id,
            reviewer_id="reviewer",
            payload={"valid": True},
            overall_score=3.0,
            passed=True,
        )
        assert human_store.batch_bundle(batch_id, tenant_id)["batch"]["name"] == (
            "postgres-integration"
        )

        report = {
            "report_id": report_id,
            "experiment": {"experiment_id": f"experiment-{suffix}"},
            "generated_at": "2026-08-23T00:00:00Z",
            "release_decision": {"status": "eligible_for_human_approval"},
            "quality_comparison": {
                "pass_rate": {"delta": 0.1},
                "overall_score": {"delta": 0.2},
            },
            "safety": {"new_failure_count": 0},
        }
        analysis_store.save_report(tenant_id, report)
        assert (
            PostgresEvaluationAnalysisStore(POSTGRES_URL).get_report(tenant_id, report_id) == report
        )
    finally:
        with psycopg.connect(POSTGRES_URL) as connection:
            if batch_id is not None:
                connection.execute(
                    "DELETE FROM human_eval_annotations WHERE assignment_id IN ("
                    "SELECT assignment_id FROM human_eval_assignments WHERE batch_id = %s)",
                    (batch_id,),
                )
                for table in (
                    "human_eval_qc_reviews",
                    "human_eval_adjudications",
                    "human_eval_audit_events",
                    "human_eval_assignments",
                    "human_eval_items",
                    "human_eval_batches",
                ):
                    connection.execute(
                        f"DELETE FROM {table} WHERE batch_id = %s",  # noqa: S608
                        (batch_id,),
                    )
            connection.execute(
                "DELETE FROM evaluation_analysis_reports WHERE tenant_id = %s AND report_id = %s",
                (tenant_id, report_id),
            )

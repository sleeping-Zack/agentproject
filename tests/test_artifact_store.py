from services.artifact_store import SQLiteArtifactStore


def test_artifact_store_saves_and_lists_request_artifacts(tmp_path):
    store = SQLiteArtifactStore(str(tmp_path / "artifacts.db"))

    artifact = store.save_artifact(
        request_id="req-1",
        tenant_id="tenant-a",
        artifact_type="answer",
        name="final-answer",
        payload={"answer": "报告内容"},
        metadata={"source": "runner"},
    )

    assert artifact.artifact_id

    loaded = store.get_artifact(artifact.artifact_id)
    assert loaded.payload["answer"] == "报告内容"
    assert loaded.metadata["source"] == "runner"

    listed = store.list_artifacts("req-1", tenant_id="tenant-a")
    assert [item.artifact_id for item in listed] == [artifact.artifact_id]


def test_artifact_store_is_idempotent_per_request_type_and_name(tmp_path):
    store = SQLiteArtifactStore(str(tmp_path / "artifacts.db"))
    first = store.save_artifact(
        request_id="req-idempotent",
        tenant_id="tenant-a",
        artifact_type="answer",
        name="final-answer",
        payload={"answer": "first"},
    )
    duplicate = store.save_artifact(
        request_id="req-idempotent",
        tenant_id="tenant-a",
        artifact_type="answer",
        name="final-answer",
        payload={"answer": "retry"},
    )

    assert duplicate.artifact_id == first.artifact_id
    assert duplicate.payload == {"answer": "first"}
    assert len(store.list_artifacts("req-idempotent", tenant_id="tenant-a")) == 1

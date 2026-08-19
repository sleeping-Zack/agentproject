from services.frontend_api import AgentApiClient
from tests.test_frontend_api import FakeResponse, FakeSession


def test_client_supports_reviewer_human_eval_workflow():
    session = FakeSession(
        [
            FakeResponse({"task": {"assignment_id": "assignment-1"}}),
            FakeResponse({"status": "submitted"}),
            FakeResponse({"submitted_assignments": 1}),
        ]
    )
    client = AgentApiClient(
        "http://api", "secret", "tenant-a", "reviewer-a", session=session
    )
    submission = {
        "valid": True,
        "scores": {"task_completion": 3},
        "vetoes": [],
        "rationales": {},
        "confidence": "high",
        "duration_seconds": 10,
    }

    assert client.next_human_eval_task("batch-1") == {
        "assignment_id": "assignment-1"
    }
    assert client.submit_human_eval_annotation("assignment-1", submission) == {
        "status": "submitted"
    }
    assert client.human_eval_progress("batch-1") == {"submitted_assignments": 1}

    assert [(call[0], call[1]) for call in session.calls] == [
        ("GET", "http://api/human-eval/batches/batch-1/tasks/next"),
        ("POST", "http://api/human-eval/tasks/assignment-1/submit"),
        ("GET", "http://api/human-eval/batches/batch-1"),
    ]
    assert session.calls[1][2]["json"] is submission


def test_client_supports_operator_human_eval_workflow():
    session = FakeSession(
        [
            FakeResponse([{"item_id": "item-1"}]),
            FakeResponse({"status": "adjudicated"}),
            FakeResponse([{"item_id": "item-2"}]),
            FakeResponse({"status": "returned"}),
            FakeResponse({"pair_count": 4}),
            FakeResponse({"labels": []}),
            FakeResponse({"status": "closed"}),
        ]
    )
    client = AgentApiClient(
        "http://api", "secret", "tenant-a", "operator-a", session=session
    )
    submission = {"valid": False}

    assert client.human_eval_disagreements("batch-1") == [{"item_id": "item-1"}]
    assert client.adjudicate_human_eval_item(
        "batch-1", "item-1", submission
    ) == {"status": "adjudicated"}
    assert client.human_eval_qc_queue("batch-1") == [{"item_id": "item-2"}]
    assert client.review_human_eval_qc(
        "batch-1",
        "item-2",
        decision="returned",
        note="Please rescore.",
        returned_assignment_ids=["assignment-2"],
    ) == {"status": "returned"}
    assert client.human_eval_report("batch-1") == {"pair_count": 4}
    assert client.export_human_eval_labels("batch-1") == {"labels": []}
    assert client.close_human_eval_batch("batch-1") == {"status": "closed"}

    assert [(call[0], call[1]) for call in session.calls] == [
        ("GET", "http://api/human-eval/batches/batch-1/disagreements"),
        (
            "POST",
            "http://api/human-eval/batches/batch-1/items/item-1/adjudicate",
        ),
        ("GET", "http://api/human-eval/batches/batch-1/qc"),
        ("POST", "http://api/human-eval/batches/batch-1/items/item-2/qc"),
        ("GET", "http://api/human-eval/batches/batch-1/report"),
        ("GET", "http://api/human-eval/batches/batch-1/export"),
        ("POST", "http://api/human-eval/batches/batch-1/close"),
    ]
    assert session.calls[1][2]["json"] is submission
    assert session.calls[3][2]["json"] == {
        "decision": "returned",
        "note": "Please rescore.",
        "returned_assignment_ids": ["assignment-2"],
    }

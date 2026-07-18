from __future__ import annotations

from fastapi.testclient import TestClient

from api.reranker_server import create_app


class FakeRuntime:
    model_name = "test-model"

    def __init__(self) -> None:
        self.is_loaded = False
        self.calls = []

    def load(self) -> None:
        self.is_loaded = True

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return [0.1, 0.9]


def test_reranker_service_preloads_and_returns_aligned_scores():
    runtime = FakeRuntime()
    app = create_app(runtime)

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        response = client.post(
            "/rerank",
            json={
                "model": "test-model",
                "query": "怎么清理滚刷",
                "documents": ["购买指南", "清理缠绕毛发"],
                "top_n": 2,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"model": "test-model", "scores": [0.1, 0.9]}
    assert runtime.calls == [("怎么清理滚刷", ["购买指南", "清理缠绕毛发"])]


def test_reranker_service_rejects_model_switch_and_partial_score_contract():
    runtime = FakeRuntime()
    app = create_app(runtime, preload=False)

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
        wrong_model = client.post(
            "/rerank",
            json={
                "model": "another-model",
                "query": "query",
                "documents": ["one"],
                "top_n": 1,
            },
        )
        partial = client.post(
            "/rerank",
            json={
                "model": "test-model",
                "query": "query",
                "documents": ["one", "two"],
                "top_n": 1,
            },
        )

    assert wrong_model.status_code == 400
    assert partial.status_code == 400


def test_reranker_service_rejects_oversized_documents(monkeypatch):
    monkeypatch.setenv("AGENT_RERANK_SERVICE_MAX_DOCUMENT_CHARS", "4")
    app = create_app(FakeRuntime(), preload=False)

    with TestClient(app) as client:
        response = client.post(
            "/rerank",
            json={
                "model": "test-model",
                "query": "query",
                "documents": ["12345"],
                "top_n": 1,
            },
        )

    assert response.status_code == 413

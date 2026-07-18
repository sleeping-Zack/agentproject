import pytest

from utils.config_handler import _apply_chroma_env_overrides, _resolve_config_path


def test_rerank_environment_overrides_are_typed(monkeypatch):
    monkeypatch.setenv("AGENT_RERANK_ENABLED", "true")
    monkeypatch.setenv("AGENT_RERANK_BACKEND", "remote")
    monkeypatch.setenv("AGENT_RERANK_URL", "http://reranker.internal/rerank")
    monkeypatch.setenv("AGENT_RERANK_VERSION", "experiment-v3")
    monkeypatch.setenv("AGENT_RERANK_STRATEGY", "weighted_rrf")
    monkeypatch.setenv("AGENT_RERANK_HYBRID_WEIGHT", "0.8")
    monkeypatch.setenv("AGENT_RERANK_MODEL_WEIGHT", "0.2")
    monkeypatch.setenv("AGENT_RERANK_FUSION_K", "12")
    monkeypatch.setenv("AGENT_RERANK_MAX_DOCUMENT_CHARS", "900")
    monkeypatch.setenv("AGENT_RERANK_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("AGENT_RERANK_FAILURE_THRESHOLD", "4")
    monkeypatch.setenv("AGENT_RERANK_RECOVERY_SECONDS", "20")

    config = _apply_chroma_env_overrides({"retrieval": {}})

    retrieval = config["retrieval"]
    assert retrieval["enable_reranker"] is True
    assert retrieval["reranker_backend"] == "remote"
    assert retrieval["reranker_url"] == "http://reranker.internal/rerank"
    assert retrieval["rerank_version"] == "experiment-v3"
    assert retrieval["rerank_hybrid_weight"] == 0.8
    assert retrieval["rerank_model_weight"] == 0.2
    assert retrieval["rerank_fusion_k"] == 12
    assert retrieval["rerank_max_document_chars"] == 900
    assert retrieval["reranker_timeout_seconds"] == 1.5
    assert retrieval["reranker_failure_threshold"] == 4
    assert retrieval["reranker_recovery_seconds"] == 20.0


def test_rerank_environment_overrides_reject_invalid_boolean(monkeypatch):
    monkeypatch.setenv("AGENT_RERANK_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="enable_reranker"):
        _apply_chroma_env_overrides({"retrieval": {}})


def test_chroma_config_path_can_target_an_isolated_experiment(monkeypatch):
    monkeypatch.setenv(
        "AGENT_CHROMA_CONFIG_PATH",
        "reports/chunk-experiments/chunk-350-50/chroma.yml",
    )

    path = _resolve_config_path("AGENT_CHROMA_CONFIG_PATH", "config/chroma.yml")

    assert path.endswith("reports/chunk-experiments/chunk-350-50/chroma.yml")

from __future__ import annotations

import os

from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore
from services.persistence import SQLiteStore
from services.memory_store import PostgresMemoryStore, SQLiteMemoryStore
from services.postgres import (
    PostgresApprovalStore,
    PostgresArtifactStore,
    PostgresStore,
)


def _backend(name: str, default: str = "sqlite") -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in {"sqlite", "postgres"}:
        raise ValueError(f"unsupported backend for {name}: {value}")
    return value


def _database_url() -> str:
    return os.getenv(
        "AGENT_DATABASE_URL",
        "postgresql://agent:agent@127.0.0.1:5432/agent",
    )


def create_session_store():
    if _backend("AGENT_STORAGE_BACKEND") == "postgres":
        return PostgresStore(_database_url())
    return SQLiteStore(os.getenv("AGENT_DB_PATH", "storage/agent.db"))


def create_memory_store():
    if _backend("AGENT_STORAGE_BACKEND") == "postgres":
        return PostgresMemoryStore(_database_url())
    return SQLiteMemoryStore(os.getenv("AGENT_DB_PATH", "storage/agent.db"))


def create_memory_index():
    if os.getenv("AGENT_MEMORY_VECTOR_ENABLED", "false").strip().lower() != "true":
        return None
    from agent.memory_index import ChromaMemoryIndex
    from model.factory import embed_model

    return ChromaMemoryIndex(
        os.getenv("AGENT_MEMORY_VECTOR_PATH", "storage/memory_chroma"),
        embed_model,
    )


def create_approval_store():
    default = os.getenv("AGENT_STORAGE_BACKEND", "sqlite")
    if _backend("AGENT_APPROVAL_BACKEND", default) == "postgres":
        return PostgresApprovalStore(_database_url())
    return SQLiteApprovalStore(
        os.getenv("AGENT_APPROVAL_DB_PATH", "storage/approvals.db")
    )


def create_artifact_store():
    default = os.getenv("AGENT_STORAGE_BACKEND", "sqlite")
    if _backend("AGENT_ARTIFACT_BACKEND", default) == "postgres":
        return PostgresArtifactStore(_database_url())
    return SQLiteArtifactStore(
        os.getenv("AGENT_ARTIFACT_DB_PATH", "storage/artifacts.db")
    )

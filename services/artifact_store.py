from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

from safety.approval import utc_now_iso


@dataclass
class ArtifactRecord:
    artifact_id: str
    request_id: str
    tenant_id: str
    artifact_type: str
    name: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


class ArtifactStore(Protocol):
    def save_artifact(
        self,
        request_id: str,
        tenant_id: str,
        artifact_type: str,
        name: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord: ...

    def get_artifact(self, artifact_id: str) -> ArtifactRecord: ...

    def list_artifacts(
        self,
        request_id: str,
        tenant_id: Optional[str] = None,
    ) -> List[ArtifactRecord]: ...


class SQLiteArtifactStore:
    def __init__(self, db_path: str = "storage/artifacts.db") -> None:
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS artifacts ("
                "artifact_id TEXT PRIMARY KEY,"
                "request_id TEXT NOT NULL,"
                "tenant_id TEXT NOT NULL,"
                "artifact_type TEXT NOT NULL,"
                "name TEXT NOT NULL,"
                "payload TEXT NOT NULL,"
                "metadata TEXT NOT NULL,"
                "created_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_request "
                "ON artifacts(request_id, tenant_id)"
            )
            conn.execute(
                "DELETE FROM artifacts WHERE rowid NOT IN ("
                "SELECT MIN(rowid) FROM artifacts "
                "GROUP BY tenant_id, request_id, artifact_type, name)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_idempotency "
                "ON artifacts(tenant_id, request_id, artifact_type, name)"
            )

    def save_artifact(
        self,
        request_id: str,
        tenant_id: str,
        artifact_type: str,
        name: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            artifact_id=str(uuid4()),
            request_id=request_id,
            tenant_id=tenant_id,
            artifact_type=artifact_type,
            name=name,
            payload=payload,
            metadata=metadata or {},
            created_at=utc_now_iso(),
        )
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO artifacts("
                "artifact_id, request_id, tenant_id, artifact_type, name, "
                "payload, metadata, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.artifact_id,
                    record.request_id,
                    record.tenant_id,
                    record.artifact_type,
                    record.name,
                    json.dumps(record.payload, ensure_ascii=False),
                    json.dumps(record.metadata, ensure_ascii=False),
                    record.created_at,
                ),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT artifact_id, request_id, tenant_id, artifact_type, name, "
                    "payload, metadata, created_at FROM artifacts "
                    "WHERE tenant_id = ? AND request_id = ? "
                    "AND artifact_type = ? AND name = ?",
                    (tenant_id, request_id, artifact_type, name),
                ).fetchone()
                if row is not None:
                    return self._row_to_record(row)
        return record

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT artifact_id, request_id, tenant_id, artifact_type, name, "
                "payload, metadata, created_at FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return self._row_to_record(row)

    def list_artifacts(
        self,
        request_id: str,
        tenant_id: Optional[str] = None,
    ) -> List[ArtifactRecord]:
        if tenant_id:
            query = (
                "SELECT artifact_id, request_id, tenant_id, artifact_type, name, "
                "payload, metadata, created_at FROM artifacts "
                "WHERE request_id = ? AND tenant_id = ? ORDER BY created_at"
            )
            params = (request_id, tenant_id)
        else:
            query = (
                "SELECT artifact_id, request_id, tenant_id, artifact_type, name, "
                "payload, metadata, created_at FROM artifacts "
                "WHERE request_id = ? ORDER BY created_at"
            )
            params = (request_id,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row[0],
            request_id=row[1],
            tenant_id=row[2],
            artifact_type=row[3],
            name=row[4],
            payload=json.loads(row[5]),
            metadata=json.loads(row[6]),
            created_at=row[7],
        )

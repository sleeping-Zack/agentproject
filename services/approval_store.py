from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Protocol
from uuid import uuid4

from safety.approval import ApprovalRecord, utc_now_iso


class ApprovalStore(Protocol):
    def create_pending(
        self,
        request_id: str,
        tenant_id: str,
        user_role: str,
        tool_name: str,
        args: Dict[str, Any],
        reason: str,
    ) -> ApprovalRecord: ...

    def get(self, approval_id: str) -> ApprovalRecord: ...

    def approve(self, approval_id: str, decided_by: str) -> ApprovalRecord: ...

    def deny(self, approval_id: str, decided_by: str) -> ApprovalRecord: ...


class SQLiteApprovalStore:
    def __init__(self, db_path: str = "storage/approvals.db") -> None:
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
                "CREATE TABLE IF NOT EXISTS approvals ("
                "approval_id TEXT PRIMARY KEY,"
                "request_id TEXT NOT NULL,"
                "tenant_id TEXT NOT NULL,"
                "user_role TEXT NOT NULL,"
                "tool_name TEXT NOT NULL,"
                "args TEXT NOT NULL,"
                "reason TEXT NOT NULL,"
                "status TEXT NOT NULL,"
                "created_at TEXT NOT NULL,"
                "decided_at TEXT,"
                "decided_by TEXT)"
            )
            conn.execute(
                "DELETE FROM approvals WHERE rowid NOT IN ("
                "SELECT MIN(rowid) FROM approvals "
                "GROUP BY tenant_id, request_id, tool_name)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_approvals_request_tool "
                "ON approvals(tenant_id, request_id, tool_name)"
            )

    def create_pending(
        self,
        request_id: str,
        tenant_id: str,
        user_role: str,
        tool_name: str,
        args: Dict[str, Any],
        reason: str,
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=str(uuid4()),
            request_id=request_id,
            tenant_id=tenant_id,
            user_role=user_role,
            tool_name=tool_name,
            args=args,
            reason=reason,
            created_at=utc_now_iso(),
        )
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO approvals("
                "approval_id, request_id, tenant_id, user_role, tool_name, args, "
                "reason, status, created_at, decided_at, decided_by"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.approval_id,
                    record.request_id,
                    record.tenant_id,
                    record.user_role,
                    record.tool_name,
                    json.dumps(record.args, ensure_ascii=False),
                    record.reason,
                    record.status,
                    record.created_at,
                    record.decided_at,
                    record.decided_by,
                ),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT approval_id, request_id, tenant_id, user_role, tool_name, args, "
                    "reason, status, created_at, decided_at, decided_by "
                    "FROM approvals WHERE tenant_id = ? AND request_id = ? AND tool_name = ?",
                    (tenant_id, request_id, tool_name),
                ).fetchone()
                if row is not None:
                    return self._row_to_record(row)
        return record

    def get(self, approval_id: str) -> ApprovalRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT approval_id, request_id, tenant_id, user_role, tool_name, args, "
                "reason, status, created_at, decided_at, decided_by "
                "FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            raise KeyError(approval_id)
        return self._row_to_record(row)

    def approve(self, approval_id: str, decided_by: str) -> ApprovalRecord:
        return self._decide(approval_id, "approved", decided_by)

    def deny(self, approval_id: str, decided_by: str) -> ApprovalRecord:
        return self._decide(approval_id, "denied", decided_by)

    def _decide(self, approval_id: str, status: str, decided_by: str) -> ApprovalRecord:
        decided_at = utc_now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ? "
                "WHERE approval_id = ? AND status = 'pending'",
                (status, decided_at, decided_by, approval_id),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT 1 FROM approvals WHERE approval_id = ?",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(approval_id)
        return self.get(approval_id)

    @staticmethod
    def _row_to_record(row) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=row[0],
            request_id=row[1],
            tenant_id=row[2],
            user_role=row[3],
            tool_name=row[4],
            args=json.loads(row[5]),
            reason=row[6],
            status=row[7],
            created_at=row[8],
            decided_at=row[9],
            decided_by=row[10],
        )

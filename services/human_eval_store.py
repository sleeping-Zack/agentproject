from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteHumanEvalStore:
    def __init__(self, db_path: str | Path = "storage/human_eval.db") -> None:
        self.db_path = str(db_path)
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS human_eval_batches (
                    batch_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    rubric_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    assignments_per_item INTEGER NOT NULL,
                    qc_rate REAL NOT NULL,
                    seed INTEGER NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    closed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS human_eval_items (
                    item_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    blind_payload TEXT NOT NULL,
                    oracle_payload TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    qc_selected INTEGER NOT NULL,
                    qc_status TEXT NOT NULL,
                    FOREIGN KEY(batch_id) REFERENCES human_eval_batches(batch_id),
                    UNIQUE(batch_id, case_id)
                );

                CREATE TABLE IF NOT EXISTS human_eval_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    reviewer_slot INTEGER NOT NULL,
                    blind_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    assigned_at TEXT NOT NULL,
                    claimed_at TEXT,
                    submitted_at TEXT,
                    FOREIGN KEY(batch_id) REFERENCES human_eval_batches(batch_id),
                    FOREIGN KEY(item_id) REFERENCES human_eval_items(item_id),
                    UNIQUE(item_id, reviewer_id)
                );

                CREATE TABLE IF NOT EXISTS human_eval_annotations (
                    annotation_id TEXT PRIMARY KEY,
                    assignment_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    overall_score REAL,
                    passed INTEGER,
                    is_current INTEGER NOT NULL,
                    submitted_at TEXT NOT NULL,
                    FOREIGN KEY(assignment_id) REFERENCES human_eval_assignments(assignment_id),
                    UNIQUE(assignment_id, revision)
                );

                CREATE TABLE IF NOT EXISTS human_eval_qc_reviews (
                    qc_review_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL,
                    returned_assignments TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(batch_id) REFERENCES human_eval_batches(batch_id),
                    FOREIGN KEY(item_id) REFERENCES human_eval_items(item_id)
                );

                CREATE TABLE IF NOT EXISTS human_eval_adjudications (
                    adjudication_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    item_id TEXT NOT NULL UNIQUE,
                    adjudicator_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    overall_score REAL,
                    passed INTEGER,
                    triggers TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(batch_id) REFERENCES human_eval_batches(batch_id),
                    FOREIGN KEY(item_id) REFERENCES human_eval_items(item_id)
                );

                CREATE TABLE IF NOT EXISTS human_eval_audit_events (
                    event_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    item_id TEXT,
                    actor_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(batch_id) REFERENCES human_eval_batches(batch_id)
                );

                CREATE INDEX IF NOT EXISTS idx_human_eval_assignment_queue
                    ON human_eval_assignments(batch_id, reviewer_id, status, assigned_at);
                CREATE INDEX IF NOT EXISTS idx_human_eval_annotation_current
                    ON human_eval_annotations(assignment_id, is_current);
                CREATE INDEX IF NOT EXISTS idx_human_eval_items_qc
                    ON human_eval_items(batch_id, qc_selected, qc_status);
                """
            )

    def create_batch(
        self,
        *,
        tenant_id: str,
        name: str,
        dataset_version: str,
        rubric_version: str,
        assignments_per_item: int,
        qc_rate: float,
        seed: int,
        created_by: str,
        items: Sequence[Mapping[str, Any]],
        assignments: Sequence[Mapping[str, Any]],
    ) -> str:
        batch_id = str(uuid4())
        created_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO human_eval_batches("
                "batch_id, tenant_id, name, dataset_version, rubric_version, status, "
                "assignments_per_item, qc_rate, seed, created_by, created_at, closed_at"
                ") VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, NULL)",
                (
                    batch_id,
                    tenant_id,
                    name,
                    dataset_version,
                    rubric_version,
                    assignments_per_item,
                    qc_rate,
                    seed,
                    created_by,
                    created_at,
                ),
            )
            item_ids: Dict[str, str] = {}
            for item in items:
                item_id = str(uuid4())
                item_ids[str(item["case_id"])] = item_id
                conn.execute(
                    "INSERT INTO human_eval_items("
                    "item_id, batch_id, case_id, blind_payload, oracle_payload, ordinal, "
                    "qc_selected, qc_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        item_id,
                        batch_id,
                        str(item["case_id"]),
                        self._json(item["blind_payload"]),
                        self._json(item.get("oracle_payload") or {}),
                        int(item["ordinal"]),
                        1 if item.get("qc_selected") else 0,
                        "waiting" if item.get("qc_selected") else "not_selected",
                    ),
                )
            for assignment in assignments:
                case_id = str(assignment["case_id"])
                conn.execute(
                    "INSERT INTO human_eval_assignments("
                    "assignment_id, batch_id, item_id, reviewer_id, reviewer_slot, blind_key, "
                    "status, revision, assigned_at, claimed_at, submitted_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, 'assigned', 0, ?, NULL, NULL)",
                    (
                        str(uuid4()),
                        batch_id,
                        item_ids[case_id],
                        str(assignment["reviewer_id"]),
                        int(assignment["reviewer_slot"]),
                        str(uuid4()),
                        created_at,
                    ),
                )
            self._audit(
                conn,
                batch_id=batch_id,
                actor_id=created_by,
                event_type="batch_created",
                payload={
                    "item_count": len(items),
                    "assignment_count": len(assignments),
                    "qc_rate": qc_rate,
                },
            )
        return batch_id

    def get_batch(self, batch_id: str, tenant_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM human_eval_batches WHERE batch_id = ? AND tenant_id = ?",
                (batch_id, tenant_id),
            ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        return dict(row)

    def claim_next(
        self,
        *,
        batch_id: str,
        tenant_id: str,
        reviewer_id: str,
    ) -> Optional[Dict[str, Any]]:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            batch = conn.execute(
                "SELECT status FROM human_eval_batches WHERE batch_id = ? AND tenant_id = ?",
                (batch_id, tenant_id),
            ).fetchone()
            if batch is None:
                raise KeyError(batch_id)
            if batch["status"] != "active":
                raise ValueError("batch is not active")
            row = conn.execute(
                "SELECT a.*, i.blind_payload, i.qc_selected, i.qc_status "
                "FROM human_eval_assignments a "
                "JOIN human_eval_items i ON i.item_id = a.item_id "
                "WHERE a.batch_id = ? AND a.reviewer_id = ? AND a.status = 'in_progress' "
                "ORDER BY a.claimed_at, i.ordinal LIMIT 1",
                (batch_id, reviewer_id),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT a.*, i.blind_payload, i.qc_selected, i.qc_status "
                    "FROM human_eval_assignments a "
                    "JOIN human_eval_items i ON i.item_id = a.item_id "
                    "WHERE a.batch_id = ? AND a.reviewer_id = ? "
                    "AND a.status IN ('returned', 'assigned') "
                    "ORDER BY CASE a.status WHEN 'returned' THEN 0 ELSE 1 END, i.ordinal LIMIT 1",
                    (batch_id, reviewer_id),
                ).fetchone()
                if row is not None:
                    conn.execute(
                        "UPDATE human_eval_assignments SET status = 'in_progress', claimed_at = ? "
                        "WHERE assignment_id = ?",
                        (now, row["assignment_id"]),
                    )
                    self._audit(
                        conn,
                        batch_id=batch_id,
                        item_id=row["item_id"],
                        actor_id=reviewer_id,
                        event_type="assignment_claimed",
                        payload={"assignment_id": row["assignment_id"]},
                    )
            if row is None:
                return None
            return {
                "assignment_id": row["assignment_id"],
                "blind_key": row["blind_key"],
                "revision": int(row["revision"]),
                "returned_for_revision": row["status"] == "returned",
                "payload": json.loads(row["blind_payload"]),
            }

    def submit_annotation(
        self,
        *,
        assignment_id: str,
        tenant_id: str,
        reviewer_id: str,
        payload: Mapping[str, Any],
        overall_score: Optional[float],
        passed: Optional[bool],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT a.*, i.qc_selected FROM human_eval_assignments a "
                "JOIN human_eval_items i ON i.item_id = a.item_id "
                "JOIN human_eval_batches b ON b.batch_id = a.batch_id "
                "WHERE a.assignment_id = ? AND a.reviewer_id = ? AND b.tenant_id = ?",
                (assignment_id, reviewer_id, tenant_id),
            ).fetchone()
            if row is None:
                raise KeyError(assignment_id)
            if row["status"] not in {"in_progress", "returned"}:
                raise ValueError("assignment must be claimed before submission")
            revision = int(row["revision"]) + 1
            conn.execute(
                "UPDATE human_eval_annotations SET is_current = 0 WHERE assignment_id = ?",
                (assignment_id,),
            )
            annotation_id = str(uuid4())
            conn.execute(
                "INSERT INTO human_eval_annotations("
                "annotation_id, assignment_id, revision, payload, overall_score, passed, "
                "is_current, submitted_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    annotation_id,
                    assignment_id,
                    revision,
                    self._json(payload),
                    overall_score,
                    None if passed is None else int(passed),
                    now,
                ),
            )
            conn.execute(
                "UPDATE human_eval_assignments SET status = 'submitted', revision = ?, "
                "submitted_at = ? WHERE assignment_id = ?",
                (revision, now, assignment_id),
            )
            if row["qc_selected"]:
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM human_eval_assignments "
                    "WHERE item_id = ? AND status != 'submitted'",
                    (row["item_id"],),
                ).fetchone()[0]
                if remaining == 0:
                    conn.execute(
                        "UPDATE human_eval_items SET qc_status = 'pending' WHERE item_id = ?",
                        (row["item_id"],),
                    )
            self._audit(
                conn,
                batch_id=row["batch_id"],
                item_id=row["item_id"],
                actor_id=reviewer_id,
                event_type="annotation_submitted",
                payload={
                    "assignment_id": assignment_id,
                    "revision": revision,
                    "valid": payload["valid"],
                },
            )
        return {
            "annotation_id": annotation_id,
            "assignment_id": assignment_id,
            "revision": revision,
            "overall_score": overall_score,
            "passed": passed,
            "submitted_at": now,
        }

    def batch_bundle(self, batch_id: str, tenant_id: str) -> Dict[str, Any]:
        batch = self.get_batch(batch_id, tenant_id)
        with self._connect() as conn:
            item_rows = conn.execute(
                "SELECT * FROM human_eval_items WHERE batch_id = ? ORDER BY ordinal",
                (batch_id,),
            ).fetchall()
            assignment_rows = conn.execute(
                "SELECT * FROM human_eval_assignments WHERE batch_id = ? "
                "ORDER BY item_id, reviewer_slot",
                (batch_id,),
            ).fetchall()
            annotation_rows = conn.execute(
                "SELECT n.*, a.item_id, a.reviewer_id, a.reviewer_slot "
                "FROM human_eval_annotations n "
                "JOIN human_eval_assignments a ON a.assignment_id = n.assignment_id "
                "WHERE a.batch_id = ? AND n.is_current = 1",
                (batch_id,),
            ).fetchall()
            adjudication_rows = conn.execute(
                "SELECT * FROM human_eval_adjudications WHERE batch_id = ?",
                (batch_id,),
            ).fetchall()
            qc_rows = conn.execute(
                "SELECT * FROM human_eval_qc_reviews WHERE batch_id = ? ORDER BY created_at",
                (batch_id,),
            ).fetchall()
        items = {
            row["item_id"]: {
                **dict(row),
                "blind_payload": json.loads(row["blind_payload"]),
                "oracle_payload": json.loads(row["oracle_payload"]),
                "assignments": [],
                "adjudication": None,
            }
            for row in item_rows
        }
        annotations = {row["assignment_id"]: self._annotation_row(row) for row in annotation_rows}
        for row in assignment_rows:
            assignment = dict(row)
            assignment["annotation"] = annotations.get(row["assignment_id"])
            items[row["item_id"]]["assignments"].append(assignment)
        for row in adjudication_rows:
            items[row["item_id"]]["adjudication"] = self._adjudication_row(row)
        return {
            "batch": batch,
            "items": list(items.values()),
            "qc_reviews": [self._qc_row(row) for row in qc_rows],
        }

    def qc_queue(self, batch_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        self.get_batch(batch_id, tenant_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT item_id, case_id, blind_payload, qc_status FROM human_eval_items "
                "WHERE batch_id = ? AND qc_selected = 1 AND qc_status IN ('pending', 'returned') "
                "ORDER BY ordinal",
                (batch_id,),
            ).fetchall()
        return [
            {
                "item_id": row["item_id"],
                "case_id": row["case_id"],
                "payload": json.loads(row["blind_payload"]),
                "qc_status": row["qc_status"],
            }
            for row in rows
        ]

    def record_qc(
        self,
        *,
        batch_id: str,
        tenant_id: str,
        item_id: str,
        reviewer_id: str,
        decision: str,
        note: str,
        returned_assignments: Sequence[str],
    ) -> Dict[str, Any]:
        if decision not in {"accepted", "returned"}:
            raise ValueError("qc decision must be accepted or returned")
        if decision == "returned" and not returned_assignments:
            raise ValueError("returned QC requires at least one assignment")
        if decision == "accepted" and returned_assignments:
            raise ValueError("accepted QC cannot return assignments")
        self.get_batch(batch_id, tenant_id)
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            item = conn.execute(
                "SELECT * FROM human_eval_items WHERE item_id = ? AND batch_id = ?",
                (item_id, batch_id),
            ).fetchone()
            if item is None:
                raise KeyError(item_id)
            if not item["qc_selected"] or item["qc_status"] not in {"pending", "returned"}:
                raise ValueError("item is not ready for QC")
            assignment_rows = conn.execute(
                "SELECT assignment_id, reviewer_id, status FROM human_eval_assignments "
                "WHERE item_id = ?",
                (item_id,),
            ).fetchall()
            if reviewer_id in {row["reviewer_id"] for row in assignment_rows}:
                raise ValueError("QC reviewer must be independent from both reviewers")
            valid_assignments = {
                row["assignment_id"]
                for row in assignment_rows
                if row["status"] == "submitted"
            }
            if not set(returned_assignments).issubset(valid_assignments):
                raise ValueError("returned assignment does not belong to the submitted item")
            qc_review_id = str(uuid4())
            conn.execute(
                "INSERT INTO human_eval_qc_reviews("
                "qc_review_id, batch_id, item_id, reviewer_id, decision, note, "
                "returned_assignments, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    qc_review_id,
                    batch_id,
                    item_id,
                    reviewer_id,
                    decision,
                    note.strip(),
                    self._json(list(returned_assignments)),
                    now,
                ),
            )
            conn.execute(
                "UPDATE human_eval_items SET qc_status = ? WHERE item_id = ?",
                (decision, item_id),
            )
            if decision == "returned":
                placeholders = ",".join("?" for _ in returned_assignments)
                conn.execute(
                    f"UPDATE human_eval_assignments SET status = 'returned' "
                    f"WHERE assignment_id IN ({placeholders})",
                    tuple(returned_assignments),
                )
            self._audit(
                conn,
                batch_id=batch_id,
                item_id=item_id,
                actor_id=reviewer_id,
                event_type=f"qc_{decision}",
                payload={
                    "qc_review_id": qc_review_id,
                    "returned_assignments": list(returned_assignments),
                },
            )
        return {
            "qc_review_id": qc_review_id,
            "item_id": item_id,
            "decision": decision,
            "returned_assignments": list(returned_assignments),
            "created_at": now,
        }

    def save_adjudication(
        self,
        *,
        batch_id: str,
        tenant_id: str,
        item_id: str,
        adjudicator_id: str,
        payload: Mapping[str, Any],
        overall_score: Optional[float],
        passed: Optional[bool],
        triggers: Sequence[str],
    ) -> Dict[str, Any]:
        self.get_batch(batch_id, tenant_id)
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            item = conn.execute(
                "SELECT 1 FROM human_eval_items WHERE item_id = ? AND batch_id = ?",
                (item_id, batch_id),
            ).fetchone()
            if item is None:
                raise KeyError(item_id)
            existing = conn.execute(
                "SELECT 1 FROM human_eval_adjudications WHERE item_id = ?",
                (item_id,),
            ).fetchone()
            if existing is not None:
                raise ValueError("item is already adjudicated")
            adjudication_id = str(uuid4())
            conn.execute(
                "INSERT INTO human_eval_adjudications("
                "adjudication_id, batch_id, item_id, adjudicator_id, payload, overall_score, "
                "passed, triggers, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    adjudication_id,
                    batch_id,
                    item_id,
                    adjudicator_id,
                    self._json(payload),
                    overall_score,
                    None if passed is None else int(passed),
                    self._json(list(triggers)),
                    now,
                ),
            )
            self._audit(
                conn,
                batch_id=batch_id,
                item_id=item_id,
                actor_id=adjudicator_id,
                event_type="item_adjudicated",
                payload={"adjudication_id": adjudication_id, "triggers": list(triggers)},
            )
        return {
            "adjudication_id": adjudication_id,
            "item_id": item_id,
            "triggers": list(triggers),
            "overall_score": overall_score,
            "passed": passed,
            "created_at": now,
        }

    def close_batch(self, batch_id: str, tenant_id: str, actor_id: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            batch = conn.execute(
                "SELECT status FROM human_eval_batches WHERE batch_id = ? AND tenant_id = ?",
                (batch_id, tenant_id),
            ).fetchone()
            if batch is None:
                raise KeyError(batch_id)
            if batch["status"] != "active":
                raise ValueError("batch is not active")
            conn.execute(
                "UPDATE human_eval_batches SET status = 'closed', closed_at = ? "
                "WHERE batch_id = ?",
                (now, batch_id),
            )
            self._audit(
                conn,
                batch_id=batch_id,
                actor_id=actor_id,
                event_type="batch_closed",
                payload={},
            )

    def audit_events(self, batch_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        self.get_batch(batch_id, tenant_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM human_eval_audit_events WHERE batch_id = ? ORDER BY created_at",
                (batch_id,),
            ).fetchall()
        return [
            {**dict(row), "payload": json.loads(row["payload"])} for row in rows
        ]

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _annotation_row(cls, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "annotation_id": row["annotation_id"],
            "assignment_id": row["assignment_id"],
            "reviewer_id": row["reviewer_id"],
            "reviewer_slot": row["reviewer_slot"],
            "revision": row["revision"],
            **json.loads(row["payload"]),
            "overall_score": row["overall_score"],
            "passed": None if row["passed"] is None else bool(row["passed"]),
            "submitted_at": row["submitted_at"],
        }

    @staticmethod
    def _adjudication_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "adjudication_id": row["adjudication_id"],
            "adjudicator_id": row["adjudicator_id"],
            **json.loads(row["payload"]),
            "overall_score": row["overall_score"],
            "passed": None if row["passed"] is None else bool(row["passed"]),
            "triggers": json.loads(row["triggers"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _qc_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            **dict(row),
            "returned_assignments": json.loads(row["returned_assignments"]),
        }

    def _audit(
        self,
        conn: sqlite3.Connection,
        *,
        batch_id: str,
        actor_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        item_id: Optional[str] = None,
    ) -> None:
        conn.execute(
            "INSERT INTO human_eval_audit_events("
            "event_id, batch_id, item_id, actor_id, event_type, payload, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                batch_id,
                item_id,
                actor_id,
                event_type,
                self._json(payload),
                utc_now_iso(),
            ),
        )

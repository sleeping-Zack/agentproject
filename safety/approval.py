from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ApprovalRecord:
    approval_id: str
    request_id: str
    tenant_id: str
    user_role: str
    tool_name: str
    args: Dict[str, Any]
    reason: str
    status: str = "pending"
    created_at: str = ""
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"

    @property
    def is_approved(self) -> bool:
        return self.status == "approved"

    @property
    def is_denied(self) -> bool:
        return self.status == "denied"

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


VALID_ROLES = {"user", "operator", "admin"}
ADMIN_ROLES = {"operator", "admin"}


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str = "default"
    user_role: str = "user"
    principal_id: str = "anonymous"

    @property
    def can_approve(self) -> bool:
        return self.user_role in ADMIN_ROLES


def resolve_auth_context(
    api_key: str,
    header_tenant_id: Optional[str] = None,
    header_user_role: Optional[str] = None,
    header_principal_id: Optional[str] = None,
    body_tenant_id: Optional[str] = None,
) -> AuthContext:
    """Build the trusted request auth context.

    This intentionally does not accept `user_role` from request bodies. In this
    demo deployment the role may come from an auth header; production callers
    should replace this with API-key/JWT claims.
    """
    del api_key  # The caller validates the key before resolving the context.
    tenant_id = header_tenant_id or body_tenant_id or "default"
    user_role = header_user_role or "user"
    if user_role not in VALID_ROLES:
        raise ValueError(f"invalid user role: {user_role}")
    principal_id = header_principal_id or f"{user_role}:{tenant_id}"
    return AuthContext(
        tenant_id=tenant_id,
        user_role=user_role,
        principal_id=principal_id,
    )

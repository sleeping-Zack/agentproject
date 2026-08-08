from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


VALID_ROLES = {"user", "operator", "admin"}
ADMIN_ROLES = {"operator", "admin"}


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str = "default"
    user_role: str = "user"
    principal_id: str = "anonymous"
    data_user_id: Optional[str] = None

    @property
    def can_approve(self) -> bool:
        return self.user_role in ADMIN_ROLES


def resolve_auth_context(
    api_key: str,
    authorization: Optional[str] = None,
    header_tenant_id: Optional[str] = None,
    header_user_role: Optional[str] = None,
    header_principal_id: Optional[str] = None,
    body_tenant_id: Optional[str] = None,
) -> AuthContext:
    """Build an authenticated identity without trusting caller-supplied claims."""
    mode = os.getenv("AGENT_AUTH_MODE", "principal_api_key").strip().lower()
    if mode == "jwt":
        context = _jwt_context(authorization)
        _reject_claim_mismatch(
            context,
            header_tenant_id=header_tenant_id,
            header_user_role=header_user_role,
            header_principal_id=header_principal_id,
            body_tenant_id=body_tenant_id,
        )
        return context
    if mode == "principal_api_key":
        context = _api_key_context(api_key)
        _reject_claim_mismatch(
            context,
            header_tenant_id=header_tenant_id,
            header_user_role=header_user_role,
            header_principal_id=header_principal_id,
            body_tenant_id=body_tenant_id,
        )
        return context
    if mode != "legacy_headers":
        raise ValueError(f"unsupported authentication mode: {mode}")

    # Explicit migration/test mode only. Production modes derive these values
    # from a signed token or a server-side API-key principal mapping.
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


def _api_key_context(api_key: str) -> AuthContext:
    configured = os.getenv("AGENT_API_PRINCIPALS_JSON", "").strip()
    principals: Dict[str, Dict[str, Any]] = {}
    if configured:
        try:
            parsed = json.loads(configured)
        except json.JSONDecodeError as exc:
            raise ValueError("AGENT_API_PRINCIPALS_JSON is invalid") from exc
        if not isinstance(parsed, dict):
            raise ValueError("AGENT_API_PRINCIPALS_JSON must be an object")
        principals = {
            str(key): value for key, value in parsed.items() if isinstance(value, dict)
        }
    else:
        expected = os.getenv("AGENT_API_KEY", "dev-api-key")
        principals = {
            expected: {
                "tenant_id": os.getenv(
                    "AGENT_API_TENANT_ID",
                    os.getenv("AGENT_UI_TENANT_ID", "tenant-a"),
                ),
                "user_id": os.getenv(
                    "AGENT_API_USER_ID",
                    os.getenv("AGENT_UI_USER_ID", "user-1005"),
                ),
                "role": os.getenv(
                    "AGENT_API_USER_ROLE",
                    os.getenv("AGENT_UI_USER_ROLE", "user"),
                ),
                "data_user_id": os.getenv(
                    "AGENT_API_DATA_USER_ID",
                    os.getenv("AGENT_DEFAULT_USER_ID", ""),
                ),
            }
        }
        operator_key = os.getenv("AGENT_OPERATOR_API_KEY", "").strip()
        if operator_key:
            principals[operator_key] = {
                "tenant_id": os.getenv(
                    "AGENT_OPERATOR_TENANT_ID",
                    os.getenv("AGENT_API_TENANT_ID", "tenant-a"),
                ),
                "user_id": os.getenv("AGENT_OPERATOR_USER_ID", "operator-local"),
                "role": "operator",
            }
        admin_key = os.getenv("AGENT_ADMIN_API_KEY", "").strip()
        if admin_key:
            principals[admin_key] = {
                "tenant_id": os.getenv(
                    "AGENT_ADMIN_TENANT_ID",
                    os.getenv("AGENT_API_TENANT_ID", "tenant-a"),
                ),
                "user_id": os.getenv("AGENT_ADMIN_USER_ID", "admin-local"),
                "role": "admin",
            }

    claims = next(
        (
            value
            for candidate, value in principals.items()
            if hmac.compare_digest(str(candidate), str(api_key or ""))
        ),
        None,
    )
    if claims is None:
        raise ValueError("invalid api key")
    return _claims_context(claims)


def _jwt_context(authorization: Optional[str]) -> AuthContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("Bearer token is required")
    token = authorization[7:].strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid bearer token")
    header = _decode_segment(parts[0])
    claims = _decode_segment(parts[1])
    if header.get("alg") != "HS256":
        raise ValueError("unsupported JWT algorithm")
    secret = os.getenv("AGENT_JWT_SECRET", "")
    if not secret:
        raise ValueError("AGENT_JWT_SECRET is required")
    signed = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).digest()
    supplied = _decode_bytes(parts[2])
    if not hmac.compare_digest(expected, supplied):
        raise ValueError("invalid bearer token signature")

    now = int(time.time())
    if os.getenv("AGENT_JWT_REQUIRE_EXP", "true").strip().lower() == "true":
        if "exp" not in claims:
            raise ValueError("JWT exp claim is required")
    if "exp" in claims and now >= int(claims["exp"]):
        raise ValueError("bearer token has expired")
    if "nbf" in claims and now < int(claims["nbf"]):
        raise ValueError("bearer token is not active")
    issuer = os.getenv("AGENT_JWT_ISSUER", "").strip()
    if issuer and claims.get("iss") != issuer:
        raise ValueError("invalid JWT issuer")
    audience = os.getenv("AGENT_JWT_AUDIENCE", "").strip()
    token_audience = claims.get("aud")
    if audience and not (
        token_audience == audience
        or isinstance(token_audience, list) and audience in token_audience
    ):
        raise ValueError("invalid JWT audience")
    return _claims_context(claims)


def _decode_segment(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(_decode_bytes(value).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid bearer token") from exc
    if not isinstance(parsed, dict):
        raise ValueError("invalid bearer token")
    return parsed


def _decode_bytes(value: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid bearer token") from exc


def _claims_context(claims: Dict[str, Any]) -> AuthContext:
    tenant_id = str(claims.get("tenant_id") or claims.get("tid") or "").strip()
    principal_id = str(claims.get("user_id") or claims.get("sub") or "").strip()
    user_role = str(claims.get("role") or "user").strip()
    data_user_id = str(
        claims.get("data_user_id") or claims.get("usage_user_id") or ""
    ).strip() or None
    if not tenant_id or not principal_id:
        raise ValueError("authenticated tenant_id and user_id are required")
    if user_role not in VALID_ROLES:
        raise ValueError(f"invalid user role: {user_role}")
    return AuthContext(
        tenant_id=tenant_id,
        user_role=user_role,
        principal_id=principal_id,
        data_user_id=data_user_id,
    )


def _reject_claim_mismatch(
    context: AuthContext,
    *,
    header_tenant_id: Optional[str],
    header_user_role: Optional[str],
    header_principal_id: Optional[str],
    body_tenant_id: Optional[str],
) -> None:
    supplied = {
        "tenant_id": header_tenant_id or body_tenant_id,
        "user_role": header_user_role,
        "principal_id": header_principal_id,
    }
    for name, value in supplied.items():
        if value is not None and value.strip() and value.strip() != getattr(context, name):
            raise ValueError(f"{name} does not match authenticated identity")

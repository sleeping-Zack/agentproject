import base64
import hashlib
import hmac
import json
import time

from safety.auth import AuthContext, resolve_auth_context


def test_auth_context_uses_header_role_not_request_body():
    context = resolve_auth_context(
        api_key="dev-api-key",
        header_tenant_id="tenant-a",
        header_user_role="user",
        body_tenant_id="tenant-b",
    )

    assert context == AuthContext(
        tenant_id="tenant-a",
        user_role="user",
        principal_id="user:tenant-a",
    )


def test_auth_context_rejects_invalid_role_header():
    try:
        resolve_auth_context(
            api_key="dev-api-key",
            header_tenant_id="tenant-a",
            header_user_role="superadmin",
        )
    except ValueError as exc:
        assert "invalid user role" in str(exc)
    else:
        raise AssertionError("invalid user role was accepted")


def test_principal_api_key_ignores_forged_identity_headers(monkeypatch):
    monkeypatch.setenv("AGENT_AUTH_MODE", "principal_api_key")
    monkeypatch.setenv(
        "AGENT_API_PRINCIPALS_JSON",
        json.dumps(
            {
                "bound-key": {
                    "tenant_id": "tenant-a",
                    "user_id": "user-1",
                    "data_user_id": "1001",
                    "role": "user",
                }
            }
        ),
    )

    try:
        resolve_auth_context(
            api_key="bound-key",
            header_tenant_id="tenant-a",
            header_principal_id="user-2",
        )
    except ValueError as exc:
        assert "does not match authenticated identity" in str(exc)
    else:
        raise AssertionError("forged principal header was accepted")

    context = resolve_auth_context(api_key="bound-key")
    assert context.data_user_id == "1001"


def test_principal_api_key_supports_separate_operator_credential(monkeypatch):
    monkeypatch.setenv("AGENT_AUTH_MODE", "principal_api_key")
    monkeypatch.setenv("AGENT_API_PRINCIPALS_JSON", "")
    monkeypatch.setenv("AGENT_API_KEY", "user-key")
    monkeypatch.setenv("AGENT_API_TENANT_ID", "tenant-a")
    monkeypatch.setenv("AGENT_API_USER_ID", "user-1005")
    monkeypatch.setenv("AGENT_API_DATA_USER_ID", "1005")
    monkeypatch.setenv("AGENT_OPERATOR_API_KEY", "operator-key")
    monkeypatch.setenv("AGENT_OPERATOR_USER_ID", "operator-1")

    user = resolve_auth_context(api_key="user-key")
    operator = resolve_auth_context(api_key="operator-key")

    assert user == AuthContext("tenant-a", "user", "user-1005", "1005")
    assert operator == AuthContext("tenant-a", "operator", "operator-1", None)


def test_hs256_jwt_claims_are_verified_and_bound(monkeypatch):
    monkeypatch.setenv("AGENT_AUTH_MODE", "jwt")
    monkeypatch.setenv("AGENT_JWT_SECRET", "test-secret")
    monkeypatch.setenv("AGENT_JWT_ISSUER", "issuer-a")
    monkeypatch.setenv("AGENT_JWT_AUDIENCE", "sweeper-api")
    token = _jwt(
        {
            "sub": "user-1",
            "tenant_id": "tenant-a",
            "role": "operator",
            "iss": "issuer-a",
            "aud": "sweeper-api",
            "exp": int(time.time()) + 60,
        },
        "test-secret",
    )

    context = resolve_auth_context(api_key="", authorization=f"Bearer {token}")

    assert context == AuthContext("tenant-a", "operator", "user-1")


def test_expired_jwt_is_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_AUTH_MODE", "jwt")
    monkeypatch.setenv("AGENT_JWT_SECRET", "test-secret")
    token = _jwt(
        {
            "sub": "user-1",
            "tenant_id": "tenant-a",
            "role": "user",
            "exp": int(time.time()) - 1,
        },
        "test-secret",
    )

    try:
        resolve_auth_context(api_key="", authorization=f"Bearer {token}")
    except ValueError as exc:
        assert "expired" in str(exc)
    else:
        raise AssertionError("expired JWT was accepted")


def _jwt(claims, secret):
    def encode(value):
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    header = encode({"alg": "HS256", "typ": "JWT"})
    payload = encode(claims)
    signature = hmac.new(
        secret.encode("utf-8"),
        f"{header}.{payload}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{header}.{payload}.{encoded_signature}"

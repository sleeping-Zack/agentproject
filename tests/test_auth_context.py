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

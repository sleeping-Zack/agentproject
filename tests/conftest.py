import pytest


@pytest.fixture(autouse=True)
def _legacy_header_auth_for_existing_endpoint_tests(monkeypatch):
    """Existing API tests use explicit identities; production defaults stay claim-bound."""
    monkeypatch.setenv("AGENT_AUTH_MODE", "legacy_headers")
    monkeypatch.setenv("AGENT_SEMANTIC_ROUTER_ENABLED", "false")

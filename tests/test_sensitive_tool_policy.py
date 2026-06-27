import pytest

from agent.tools.agent_tools import fetch_external_data


def test_fetch_external_data_rejects_direct_unapproved_invocation():
    with pytest.raises(PermissionError):
        fetch_external_data.invoke({"user_id": "1001", "month": "2025-09"})

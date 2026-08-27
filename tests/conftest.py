"""
Shared pytest fixtures. auth_headers provides a valid Authorization
header for a fixed test user - get_current_user_id only verifies the
JWT signature (no database lookup), so tests don't need to actually
sign up a real user, just a validly-signed token.
"""

import pytest

from src.auth.auth import create_access_token

TEST_USER_ID = "test-user-fixed-id-000"


@pytest.fixture
def auth_headers():
    token = create_access_token(TEST_USER_ID)
    return {"Authorization": f"Bearer {token}"}

"""Shared pytest fixtures for Flask application tests.
This module provides common test setup that can be reused by acceptance,
integration, and other route-level tests. Keeping the Flask test client here
avoids duplicating fixture definitions across individual test modules.
"""

import pytest
from app import app  # This imports the Flask app for testing

@pytest.fixture
def client():
    """Provide a Flask test client for sending requests without a live server.
    The client runs requests against the imported Flask application in the
    test process. The context manager manages the client context and ensures
    it is closed after the test using the fixture completes."""
    with app.test_client() as client:
        yield client

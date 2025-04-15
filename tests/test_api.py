# -*- coding: utf-8 -*-
"""Tests for the FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient

# Need to adjust path if tests are run from root or within tests directory
# This assumes tests are run from the project root (where pyproject.toml is)
# If running pytest from within 'tests', sys.path manipulation might be needed
# A better approach is to install the package in editable mode: pip install -e .

try:
    from local_reranker.api import app
except ImportError:
    # Handle case where package is not installed/discoverable
    # This is a basic fallback, proper packaging/installation is preferred
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from local_reranker.api import app

@pytest.fixture(scope="module")
def client():
    """Pytest fixture to create a TestClient for the app."""
    with TestClient(app) as c:
        yield c

def test_health_check(client: TestClient):
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# TODO: Add tests for the /v1/rerank endpoint
# These will be more complex as they require mocking the Reranker class or 
# having a small test model available.

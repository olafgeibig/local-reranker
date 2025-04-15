# -*- coding: utf-8 -*-
"""Integration tests that connect to a manually run server."""

import pytest
import httpx

# Default URL for the manually started server
SERVER_URL = "http://127.0.0.1:8010"

# Mark as integration test
@pytest.mark.integration
def test_rerank_integration():
    """Sends a request to the manually started server."""
    # Ping the server first to give a better error if it's not running
    try:
        with httpx.Client(timeout=5.0) as client:
            ping_response = client.get(f"{SERVER_URL}/health")
            ping_response.raise_for_status()
        print(f"\nServer at {SERVER_URL} is responsive.")
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        pytest.skip(f"Skipping integration test: Server at {SERVER_URL} not reachable: {e}")

    api_url = f"{SERVER_URL}/v1/rerank"
    payload = {
        "model": "jina-reranker-v1-tiny-en", # Ignored, uses default loaded model
        "query": "Which city is the capital of France?",
        "documents": [
            "The Eiffel Tower is a famous landmark in Paris.",
            "Paris is the capital and largest city of France.",
            "Berlin is the capital of Germany."
        ],
        "top_n": 2,
        "return_documents": True
    }
    
    print(f"Sending request to {api_url}")
    try:
        # Increased timeout for the request itself
        with httpx.Client(timeout=30.0) as client:
            response = client.post(api_url, json=payload)
        print(f"Received response status: {response.status_code}")
        response.raise_for_status() # Raise exception for 4xx/5xx errors
        
        data = response.json()
        print(f"Received response data: {data}")
        
        assert "id" in data
        assert "results" in data
        assert len(data["results"]) == 2 # top_n = 2
        
        # Check results structure and content
        scores = []
        indices = []
        for result in data["results"]:
            assert "index" in result
            assert "relevance_score" in result
            assert isinstance(result["relevance_score"], float)
            scores.append(result["relevance_score"])
            indices.append(result["index"])
            assert result["document"] is not None # return_documents = True
            assert "text" in result["document"]
            assert result["document"]["text"] == payload["documents"][result["index"]]
            
        # Check basic ordering (scores should be descending)
        assert scores == sorted(scores, reverse=True)
        # Check specific expected order based on query/docs
        # Expectation: index 1 ("Paris is the capital...") should be ranked higher than index 0 ("Eiffel Tower...")
        assert indices == [1, 0]

    except httpx.RequestError as exc:
        pytest.fail(f"HTTP Request failed: {exc}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during the test: {e}")

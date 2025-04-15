# PRD: Local Reranker

## 1. Project Overview

### Description
A lightweight utility application that runs a local implementation of reranker API endpoints. Initially, the tool maintains API compatibility with the Jina `/v1/rerank` endpoint while allowing for all processing to occur locally for privacy reasons. Future versions will support additional reranking APIs and models.

### Objectives
- Provide a drop-in local replacement for reranker APIs (starting with Jina)
- Prioritize performance on Mac ARM processors while supporting NVIDIA GPUs
- Ensure privacy by processing all data locally
- Make deployment simple through PyPI package distribution
- Enable future expansion to support multiple reranking APIs and models

### Key Features
- Local implementation of the `/v1/rerank` endpoint
- GPU acceleration support for both Mac (MPS) and NVIDIA (CUDA)
- API-compatible with the Jina reranker service
- Optimized for RAG applications on Mac M1 with 16GB RAM

## 2. Technical Requirements

### API Compatibility
- Must implement the exact request/response format of Jina's `/v1/rerank` endpoint
- Support for the `jina-reranker-v2-base-multilingual` model at minimum
- Must handle all parameters supported by the original API:
  - `model`
  - `query`
  - `documents`
  - `top_n`
  - `return_documents`

### Hardware Support
- Primary: Mac with ARM processor (M1/M2/M3)
- Secondary: NVIDIA GPU-equipped machines
- Fallback: CPU execution when GPU is unavailable

### Performance Requirements
- Fast enough for interactive RAG applications on Mac M1 with 16GB RAM
- Efficient memory management to avoid OOM errors
- Low-latency response time (target <500ms for typical document sets)

### Dependencies
- Required Python packages:
  - FastAPI
  - PyTorch
  - Transformers
  - Sentence-Transformers
  - Uvicorn
  - Pydantic
  - Typer (for CLI)
  - Rich (for enhanced console logging)

## 3. Architecture

### Component Diagram
```
┌────────────────┐     ┌─────────────────┐     ┌───────────────┐
│                │     │                 │     │               │
│  FastAPI       │────▶│  ReRanker       │────▶│  Model        │
│  Endpoint      │     │  Service        │     │  (PyTorch)    │
│                │     │                 │     │               │
└────────────────┘     └─────────────────┘     └───────────────┘
                                │                     ▲
                                │                     │
                                ▼                     │
                         ┌─────────────────┐         │
                         │                 │         │
                         │  Device         │─────────┘
                         │  Manager        │
                         │                 │
                         └─────────────────┘
```

### Data Flow
1. Client sends request to the FastAPI endpoint
2. Request is validated and processed by the ReRanker service
3. Device Manager determines optimal execution device (MPS, CUDA, or CPU)
4. Model performs reranking computation
5. Results are formatted and returned to client

### Deployment Strategy
- Python package installable via pip/uv
- CLI command to start the server
- Environment variables for configuration

## 4. API Specification

### Endpoint Definition
- `POST /v1/rerank`

### Request Format
```json
{
  "model": "jina-reranker-v2-base-multilingual",
  "query": "string",
  "top_n": integer,
  "documents": ["string1", "string2", ...],
  "return_documents": boolean
}
```

### Response Format
```json
{
  "results": [
    {
      "id": 0,
      "score": 0.95,
      "text": "string1" // Only included if return_documents is true
    },
    ...
  ]
}
```

### Example Usage
```bash
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v2-base-multilingual",
    "query": "Organic skincare products for sensitive skin",
    "top_n": 3,
    "documents": [
        "Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range.",
        "New makeup trends focus on bold colors and innovative techniques: Step into the world of cutting-edge beauty with this seasons makeup trends."
    ],
    "return_documents": false
  }'
```

## 5. Implementation Plan

### Core Components

#### 0. Logging Module (`logging.py`)
- Console logging configuration
- Request/response logging
- Error logging with appropriate detail level
- Performance metrics logging

#### 1. Server Module (`server.py`)
- FastAPI application
- Endpoint implementation
- Request/response validation with Pydantic models

#### 2. Reranker Module (`reranker.py`)
- Core reranking implementation
- Model loading and caching
- Scoring algorithm

#### 3. Device Module (`device.py`)
- Device detection (MPS, CUDA, CPU)
- Optimal device selection
- Memory management utilities

#### 4. CLI Module (`cli.py`)
- Command-line interface
- Server configuration options
- Utility commands (model download, etc.)

#### 5. Models Module (`models.py`)
- Pydantic models for request/response validation
- Data transformations

### Testing Strategy
- Unit tests for each module
- Integration tests for the API
- Performance benchmarks on different hardware

### Package Structure
```
local-reranker/
├── src/
│   └── local_reranker/
│       ├── __init__.py
│       ├── server.py
│       ├── reranker.py
│       ├── device.py
│       ├── cli.py
│       ├── models.py
│       └── logging.py
├── tests/
│   ├── test_server.py
│   ├── test_reranker.py
│   └── test_device.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## 6. Package & Deployment

### Python Package
- Package name: `local-reranker`
- Entry point: `local-reranker serve`

### PyPI Packaging
- Build with `pyproject.toml` using standard tools
- Publish to PyPI with proper documentation
- Installation: `uv install local-reranker`

### Execution
- Command: `local-reranker serve --host 0.0.0.0 --port 8000`
- Environment variables:
  - `LOCAL_RERANKER_HOST`: Host address (default: 127.0.0.1)
  - `LOCAL_RERANKER_PORT`: Port number (default: 8000)
  - `LOCAL_RERANKER_MODEL`: Model name (default: jina-reranker-v2-base-multilingual)
  - `LOCAL_RERANKER_DEVICE`: Force specific device (auto, cpu, cuda, mps)

## 7. Future Considerations

### Potential Enhancements
- Support for multiple reranker models beyond Jina
- Support for other reranking API formats
- Batch processing for improved throughput
- Metrics and monitoring endpoints
- Docker containerization

### Known Limitations
- Initial model loading time may be slow
- Memory usage may be high for large document sets
- Performance is dependent on hardware capabilities
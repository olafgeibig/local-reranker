# local-reranker

A local reranker service with a Jina compatible API.

## Overview

This project provides a FastAPI-based web service that implements a reranking API endpoint (`/v1/rerank`) compatible with the [Jina AI Rerank API](https://jina.ai/rerank/). It allows you to host a reranking model entirely on your own infrastructure for enhanced privacy and performance.

## Features

*   **Jina Compatible API**: Implements `/v1/rerank` endpoint structure
*   **Local Hosting**: Run reranker model entirely on your own infrastructure
*   **Multiple Backends**: Supports PyTorch everywhere and MLX on native Apple Silicon
*   **Apple Silicon Optimization**: MLX backend is automatically installed on macOS arm64 (M1/M2/M3)
*   **MLX Fallback Reranker**: Automatically wraps MLX-converted Hugging Face models that do not ship a `rerank.py` helper
*   **Sentence Transformers**: Uses powerful `sentence-transformers` library for PyTorch backend
*   **Configurable Model**: Easily switch between different reranker models and backends
*   **Modern FastAPI**: Built using modern FastAPI features like `lifespan` for resource management
*   **Modern Dependencies**: Updated to latest stable versions with sensible minimum requirements

## Requirements

*   Python 3.12+
*   [uv](https://github.com/astral-sh/uv) (for installation and package management - recommended)
*   Sufficient RAM and compute resources (CPU or GPU) depending on the chosen reranker model

### Backend-Specific Requirements

**PyTorch Backend:**
*   PyTorch 2.5+ (automatically installed)
*   CUDA/MPS support for GPU acceleration (optional)
*   Recommended backend for Linux and Docker deployments

**MLX Backend (Apple Silicon only):**
*   Apple Silicon (M1/M2/M3) Mac
*   MLX and MLX-LM are installed automatically on native macOS arm64
*   Optimized for memory efficiency and performance on Apple chips

## Installation

```bash
# Clone the repository
git clone https://github.com/olafgeibig/local-reranker.git
cd local-reranker

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

On native macOS Apple Silicon, MLX dependencies are installed automatically.

On Linux, including Docker, the default backend is `pytorch`. MLX is not supported there because Docker containers run a Linux userspace/kernel interface even on Apple hosts, and MLX requires Apple-native runtime libraries.

## Usage

The CLI supports both modern subcommands and legacy arguments for backward compatibility.

### Modern CLI (Recommended)

```bash
# Start server with subcommand
cli serve --backend <backend_type> [options]

# Show configuration
cli config show
```

### Legacy CLI (Backward Compatible)

```bash
# Old-style arguments still work
cli --backend <backend_type> --model <model> --host <host> --port <port>
```

### Available Backends

*   `pytorch`: PyTorch-based reranker (default, cross-platform)
*   `mlx`: MLX-based reranker (native macOS Apple Silicon only)

### Backend Support By Platform

| Platform | Default backend | MLX support |
| --- | --- | --- |
| Linux / Docker | `pytorch` | Not supported |
| macOS Intel | `pytorch` | Not supported |
| macOS Apple Silicon (native) | `pytorch` | Auto-installed and supported |
| Docker Desktop on Apple Silicon | `pytorch` | Not supported (still Linux container runtime) |

### Command Options

*   `--backend`: Backend type to use (default: pytorch)
*   `--model`: Model name to use (overrides reranker default)
*   `--host`: Host to bind server to (default: 0.0.0.0)
*   `--port`: Port to bind server to (default: 8010)
*   `--log-level`: Uvicorn log level (debug, info, warning, error, critical; default: info)
*   `--reload`: Enable auto-reload for development

### Examples

**PyTorch Backend (default):**

```bash
cli serve --backend pytorch --model jinaai/jina-reranker-v2-base-multilingual
```

**MLX Backend (Apple Silicon):**

```bash
cli serve --backend mlx --model jinaai/jina-reranker-v3-mlx
```

If you explicitly select `--backend mlx` on Linux or non-Apple hardware, startup fails fast with an actionable error telling you to switch back to `pytorch`.

### Backend Selection Via Environment

```bash
# Default for Docker/Linux
export RERANKER_BACKEND_TYPE=pytorch

# Native macOS Apple Silicon (MLX auto-installed)
export RERANKER_BACKEND_TYPE=mlx
```

Command-line flags still override the defaults:

```bash
cli serve --backend pytorch
cli serve --backend mlx
```

### Docker

Build and run the container with the default PyTorch backend:

```bash
docker build -t local-reranker .
docker run --rm -p 8010:8010 \
    -e RERANKER_BACKEND_TYPE=pytorch \
    local-reranker
```

Minimal `docker compose` example:

```yaml
services:
    local-reranker:
        build: .
        ports:
            - "8010:8010"
        environment:
            RERANKER_BACKEND_TYPE: pytorch
```

**Development Mode:**

```bash
cli serve --backend pytorch --reload --log-level debug
```

**Configuration Management:**

```bash
cli config show
```

## API Usage

Once the server is running, you can send requests to the `/v1/rerank` endpoint. Here's an example using `curl`:

```bash
curl -X POST "http://localhost:8010/v1/rerank" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "jina-reranker-v2-base-multilingual", 
           "query": "What are the benefits of using FastAPI?", 
           "documents": [
             "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.",
             "Django is a high-level Python Web framework that encourages rapid development and clean, pragmatic design.",
             "The key features are: Fast, Fast to code, Fewer dependencies, Intuitive, Easy, Short, Robust, Standards-based.",
             "Flask is a micro web framework written in Python."
           ],
           "top_n": 3,
           "return_documents": true
         }'
```

### Parameters

*   `model`: (Currently ignored by API, uses the configured default) The name of the reranker model
*   `query`: The search query string
*   `documents`: A list of strings or dictionaries (`{"text": "..."}`) to be reranked against the query
*   `top_n`: (Optional) The maximum number of results to return
*   `return_documents`: (Optional, default `false`) Whether to include document text in results

## Development

### Setting Up Development Environment

1. **Clone the repository:**
    ```bash
    git clone https://github.com/olafgeibig/local-reranker.git
    cd local-reranker
    ```

2. **Create a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3. **Install development dependencies:**
    ```bash
    uv pip install -e ".[dev]"
    ```

4. **Verify installation:**
    ```bash
    # Test CLI works
    cli config show
    
    # Test server starts
    cli serve --backend pytorch --help
    ```

### Running Tests

Tests are implemented using `pytest`. To run tests:

```bash
# Ensure virtual environment is active
python -m pytest

# Or using uv run
uv run pytest

# Run specific test categories
uv run pytest -m "not integration"  # Skip integration tests
uv run pytest -m "integration"       # Only integration tests
uv run pytest -m "slow"            # Only slow tests
```

### Code Quality

The project uses modern development tools:

```bash
# Run linting
uv run ruff check

# Run type checking
uv run mypy src/

# Run both
uv run ruff check && uv run mypy src/
```

## MLX Backend Troubleshooting

### Common Issues

**MLX not found:**
```bash
# Ensure you're on Apple Silicon
uname -m  # Should show arm64

# Reinstall project dependencies (MLX auto-installs on native macOS arm64)
uv pip install -e ".[dev]"
```

**Model download fails:**
```bash
# Check internet connection
# Try manual download
huggingface-cli download jinaai/jina-reranker-v3-mlx
```

**Performance issues:**
```bash
# Check MLX is using GPU (if available)
python -c "import mlx; print(mlx.metal.is_available())"

# Monitor memory usage
top -o mem | grep python
```

### Using MLX Models Without `rerank.py`

The MLX backend now includes an internal cross-encoder reranker that automatically loads any MLX-converted Hugging Face repository, even when the repo does not provide a `rerank.py` helper. When `rerank.py` is missing or cannot be imported, the server logs a message similar to `Using internal cross-encoder fallback` so you can confirm which path is active. If your model ships a `projector.safetensors` file, place it next to the weights—the fallback reranker will load it to project hidden states into the correct embedding space. When the projector file is absent, the reranker falls back to raw hidden states, so you can still experiment with newly converted models without extra work.

### Configuration Issues

If you're having trouble with MLX backend configuration, try these explicit CLI commands:

```bash
# Force MLX backend with explicit model
cli serve --backend mlx --model jinaai/jina-reranker-v3-mlx

# Check current configuration
cli config show

# Use development mode for debugging
cli serve --backend mlx --reload --log-level debug
```

**Model download fails:**
```bash
# Check internet connection
# Try manual download
huggingface-cli download jinaai/jina-reranker-v3-mlx
```

**Performance issues:**
```bash
# Check MLX is using GPU (if available)
python -c "import mlx; print(mlx.metal.is_available())"

# Monitor memory usage
top -o mem | grep python
```

### Environment Variables

The application uses pydantic-settings for configuration management. You can set the following environment variables to override defaults:

```bash
# Force backend selection
export RERANKER_BACKEND_TYPE=mlx

# Custom model name
export RERANKER_MODEL_NAME=custom-mlx-model

# Custom host and port
export RERANKER_HOST=0.0.0.0
export RERANKER_PORT=8080

# Enable debug logging
export RERANKER_LOG_LEVEL=debug

# Enable auto-reload
export RERANKER_RELOAD=true
```

**Note**: Using the CLI command line options is recommended over environment variables for clarity.

On Linux containers, keep `RERANKER_BACKEND_TYPE=pytorch`. Selecting `mlx` there will fail because Docker/Linux cannot provide the Apple MLX shared libraries required by `mlx.core`.
On Docker Desktop running on Apple Silicon, containers are still Linux runtime environments, so MLX is still unsupported there.

## Project Structure

```
local-reranker/
├── src/local_reranker/
│   ├── __init__.py
│   ├── api.py                   # FastAPI application
│   ├── cli.py                   # Command line interface
│   ├── config.py                # Configuration management
│   ├── models.py                # Pydantic models
│   ├── reranker.py              # Base reranker interface (protocol)
│   ├── reranker_pytorch.py      # PyTorch implementation
│   ├── reranker_mlx.py          # MLX implementation
│   ├── batch_manager.py         # Batch size and document management
│   ├── batch_processor.py       # Shared batch processing logic
│   ├── result_aggregator.py     # Result aggregation and ordering
│   ├── streaming_processor.py   # Streaming result processing
│   ├── tokenization_cache.py    # Tokenization caching
│   └── utils.py                 # Utility functions
├── tests/                       # Test suite
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

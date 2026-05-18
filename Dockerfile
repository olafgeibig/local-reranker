FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Install runtime dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8010

# Copy project files
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src src
COPY uv.lock uv.lock

#Change to the bash shell
SHELL ["/bin/bash", "-c"]

# Create virtual environment and install dependencies
RUN uv venv && source .venv/bin/activate && uv pip install -e .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RERANKER_BACKEND_TYPE=pytorch \
    RERANKER_HOST=0.0.0.0 \
    RERANKER_PORT=8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8010/health')" || exit 1

# Default entrypoint - can be overridden
# Uses PyTorch backend by default, can switch to MLX with --backend mlx
ENTRYPOINT ["python", "-m", "local_reranker.cli"]
CMD ["serve", "--backend", "pytorch", "--host", "0.0.0.0", "--port", "8010"]
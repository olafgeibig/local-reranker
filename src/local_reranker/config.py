# -*- coding: utf-8 -*-
"""Configuration management for the local reranker service."""

import platform
from typing import Dict, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the reranker service."""

    # Reranker configuration
    backend_type: str = "pytorch"
    model_name: Optional[str] = None  # None = use reranker default
    disable_batching: bool = False

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8010
    log_level: str = "info"
    reload: bool = False

    class Config:
        env_prefix = "RERANKER_"
        case_sensitive = False


# Default models for each backend type
RERANKER_DEFAULTS: Dict[str, str] = {
    "pytorch": "jinaai/jina-reranker-v2-base-multilingual",
    "mlx": "jinaai/jina-reranker-v3-mlx",
    # Future implementations
    # "onnx": "some/onnx-model",
    # "tensorflow": "some/tf-model",
}


def get_effective_model_name(settings: Settings) -> str:
    """Get the effective model name based on settings and defaults."""
    if settings.model_name:
        return settings.model_name

    if settings.backend_type in RERANKER_DEFAULTS:
        return RERANKER_DEFAULTS[settings.backend_type]

    raise ValueError(
        f"No default model configured for reranker type: {settings.backend_type}"
    )


def get_available_backends() -> Dict[str, str]:
    """Get available backend types and their descriptions."""
    return {
        "pytorch": "PyTorch-based reranker using sentence-transformers",
        "mlx": "MLX-based reranker optimized for Apple Silicon (M1/M2/M3)",
        # Future implementations
        # "onnx": "ONNX-based reranker for CPU optimization",
        # "tensorflow": "TensorFlow-based reranker",
    }


def get_mlx_platform_error() -> Optional[str]:
    """Return an actionable MLX platform error, if any."""
    system = platform.system()
    machine = platform.machine().lower()

    if system != "Darwin":
        return (
            "MLX is only supported on macOS Apple Silicon. "
            f"Detected {system} on {machine or 'unknown'}, and Linux containers "
            "cannot load the Apple MLX runtime libraries."
        )

    if machine not in {"arm64", "aarch64"}:
        return (
            "MLX requires Apple Silicon arm64 hardware. "
            f"Detected macOS on {machine or 'unknown'}."
        )

    return None


def validate_backend_type(backend_type: str) -> None:
    """Validate that the requested backend can run on this platform."""
    if backend_type != "mlx":
        return

    platform_error = get_mlx_platform_error()
    if platform_error is None:
        return

    raise ValueError(
        f"{platform_error} Use '--backend pytorch' or set "
        "RERANKER_BACKEND_TYPE=pytorch. Install the optional MLX extras only "
        "on macOS Apple Silicon with `pip install 'local-reranker[mlx]'`."
    )

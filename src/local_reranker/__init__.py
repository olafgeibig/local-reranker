# -*- coding: utf-8 -*-
"""local_reranker package.

A lightweight, local reranker API implementation.
"""

__version__ = "0.0.1"  # Placeholder version

__all__ = ["main", "JinaMLXReranker"]


def __getattr__(name: str):
    """Lazily expose package entrypoints and optional MLX symbols."""
    if name == "main":
        from .cli import main

        return main

    if name == "JinaMLXReranker":
        # Keep MLX optional so importing the package on Linux does not try to
        # load Apple-only shared libraries during process startup.
        from .jina_mlx_reranker import JinaMLXReranker

        return JinaMLXReranker

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

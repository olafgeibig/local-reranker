# -*- coding: utf-8 -*-
"""Command line interface for local reranker service."""

import argparse
import logging
import os
import sys

import uvicorn
from .config import (
    Settings,
    get_available_backends,
    get_effective_model_name,
    validate_backend_type,
)

logger = logging.getLogger(__name__)


def run_server(settings: Settings) -> None:
    """Run the Local Reranker API server.

    Args:
        settings: Configuration settings.
    """
    # Set environment variables for the API module to pick up

    os.environ["RERANKER_BACKEND_TYPE"] = settings.backend_type
    if settings.model_name:
        os.environ["RERANKER_MODEL_NAME"] = settings.model_name

    # Configure logging for the application
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(
        "local_reranker.api:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.reload,
    )


def config_show(settings: Settings) -> None:
    """Show current configuration."""
    print("Current Configuration:")
    print(f"  Backend Type: {settings.backend_type}")
    print(f"  Model Name: {get_effective_model_name(settings)}")
    print(f"  Host: {settings.host}")
    print(f"  Port: {settings.port}")
    print(f"  Log Level: {settings.log_level}")
    print(f"  Reload: {settings.reload}")
    print()
    print("Available Backends:")
    for backend_type, description in get_available_backends().items():
        marker = " (current)" if backend_type == settings.backend_type else ""
        print(f"  {backend_type}: {description}{marker}")


def main() -> None:
    """Entry point for the CLI."""
    # Check for backward compatibility first
    # If no subcommand is provided, or if old-style arguments are detected, use old format
    if len(sys.argv) == 1 or (
        len(sys.argv) > 1 and sys.argv[1] not in ["serve", "config"]
    ):
        # Old-style arguments (backward compatibility)
        parser = argparse.ArgumentParser(
            description="Run the Local Reranker API server."
        )
        parser.add_argument(
            "--backend",
            type=str,
            default="pytorch",
            choices=list(get_available_backends().keys()),
            help="Backend type to use (default: pytorch).",
        )
        parser.add_argument(
            "--model",
            type=str,
            help="Model name to use (overrides reranker default).",
        )
        parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to bind the server to (default: 0.0.0.0).",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8010,
            help="Port to bind the server to (default: 8010).",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default="info",
            choices=["debug", "info", "warning", "error", "critical"],
            help="Uvicorn log level (default: info).",
        )
        parser.add_argument(
            "--reload", action="store_true", help="Enable auto-reload for development."
        )
        parser.add_argument(
            "--disable-batching",
            action="store_true",
            help="Disable batching entirely - process all documents in one batch.",
        )
        args = parser.parse_args()
        args.command = "serve"
    else:
        # New-style subcommand format
        parser = argparse.ArgumentParser(description="Local Reranker CLI.")
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Server command
        server_parser = subparsers.add_parser(
            "serve", help="Run the Local Reranker API server."
        )
        server_parser.add_argument(
            "--backend",
            type=str,
            default="pytorch",
            choices=list(get_available_backends().keys()),
            help="Backend type to use (default: pytorch).",
        )
        server_parser.add_argument(
            "--model",
            type=str,
            help="Model name to use (overrides reranker default).",
        )
        server_parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to bind the server to (default: 0.0.0.0).",
        )
        server_parser.add_argument(
            "--port",
            type=int,
            default=8010,
            help="Port to bind the server to (default: 8010).",
        )
        server_parser.add_argument(
            "--log-level",
            type=str,
            default="info",
            choices=["debug", "info", "warning", "error", "critical"],
            help="Uvicorn log level (default: info).",
        )
        server_parser.add_argument(
            "--reload", action="store_true", help="Enable auto-reload for development."
        )
        server_parser.add_argument(
            "--disable-batching",
            action="store_true",
            help="Disable batching entirely - process all documents in one batch.",
        )

        # Config command
        config_parser = subparsers.add_parser(
            "config", help="Configuration management."
        )
        config_subparsers = config_parser.add_subparsers(
            dest="config_action", help="Config actions"
        )
        config_subparsers.add_parser("show", help="Show current configuration.")

        args = parser.parse_args()

    # Handle config command
    if args.command == "config":
        if args.config_action == "show":
            settings = Settings()
            config_show(settings)
        else:
            parser.print_help()
        return

    # Handle serve command
    if args.command == "serve":
        settings = Settings(
            backend_type=args.backend,
            model_name=args.model,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            reload=args.reload,
            disable_batching=args.disable_batching,
        )

        try:
            validate_backend_type(settings.backend_type)
        except ValueError as exc:
            parser.error(str(exc))

        logger.info(
            f"Starting Local Reranker API server on {settings.host}:{settings.port}"
        )
        logger.info(f"Using reranker: {settings.backend_type}")
        logger.info(f"Using model: {get_effective_model_name(settings)}")
        run_server(settings)


if __name__ == "__main__":
    main()

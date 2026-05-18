# -*- coding: utf-8 -*-
"""Tests for the CLI module."""

import pytest
from unittest.mock import patch
import sys
import subprocess
import time
import httpx

# Import the CLI functions
from local_reranker.cli import run_server, main
from local_reranker.config import Settings


class TestRunServer:
    """Test cases for the run_server function."""

    @patch("local_reranker.cli.uvicorn.run")
    def test_run_server_default_parameters(self, mock_uvicorn_run):
        """Test run_server with default parameters."""
        settings = Settings()
        run_server(settings)

        mock_uvicorn_run.assert_called_once_with(
            "local_reranker.api:app",
            host="0.0.0.0",
            port=8010,
            log_level="info",
            reload=False,
        )

    @patch("local_reranker.cli.uvicorn.run")
    def test_run_server_custom_parameters(self, mock_uvicorn_run):
        """Test run_server with custom parameters."""
        settings = Settings(
            host="127.0.0.1",
            port=8000,
            log_level="debug",
            reload=True,
        )
        run_server(settings)

        mock_uvicorn_run.assert_called_once_with(
            "local_reranker.api:app",
            host="127.0.0.1",
            port=8000,
            log_level="debug",
            reload=True,
        )

    @patch("local_reranker.cli.uvicorn.run")
    def test_run_server_various_log_levels(self, mock_uvicorn_run):
        """Test run_server with different log levels."""
        log_levels = ["debug", "info", "warning", "error", "critical"]

        for log_level in log_levels:
            mock_uvicorn_run.reset_mock()
            settings = Settings(log_level=log_level)
            run_server(settings)

            mock_uvicorn_run.assert_called_once_with(
                "local_reranker.api:app",
                host="0.0.0.0",
                port=8010,
                log_level=log_level,
                reload=False,
            )


class TestMainFunction:
    """Test cases for the main function."""

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker"])
    def test_main_default_arguments(self, mock_run_server):
        """Test main function with default arguments."""
        main()

        mock_run_server.assert_called_once()
        call_args = mock_run_server.call_args[0][
            0
        ]  # First positional argument (Settings)

        # Check that Settings object was created with correct parameters
        assert isinstance(call_args, Settings)
        assert call_args.backend_type == "pytorch"
        assert call_args.model_name is None
        assert call_args.host == "0.0.0.0"
        assert call_args.port == 8010
        assert call_args.log_level == "info"
        assert call_args.reload is False

    @patch("local_reranker.cli.run_server")
    @patch(
        "sys.argv",
        [
            "local-reranker",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "debug",
            "--reload",
        ],
    )
    def test_main_custom_arguments(self, mock_run_server):
        """Test main function with custom arguments."""
        main()

        mock_run_server.assert_called_once()
        settings_obj = mock_run_server.call_args[0][
            0
        ]  # First positional argument (Settings)

        assert settings_obj.host == "127.0.0.1"
        assert settings_obj.port == 8000
        assert settings_obj.log_level == "debug"
        assert settings_obj.reload is True

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--help"])
    def test_main_help_argument(self, mock_run_server):
        """Test main function with help argument."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_run_server.assert_not_called()

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "serve", "--host", "127.0.0.1"])
    def test_main_host_argument(self, mock_run_server):
        """Test main function with custom host."""
        main()

        mock_run_server.assert_called_once()
        settings_obj = mock_run_server.call_args[0][
            0
        ]  # First positional argument (Settings)

        assert settings_obj.host == "127.0.0.1"
        assert settings_obj.port == 8010
        assert settings_obj.log_level == "info"
        assert settings_obj.reload is False

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--port", "9000"])
    def test_main_port_argument(self, mock_run_server):
        """Test main function with custom port."""
        main()

        mock_run_server.assert_called_once()
        settings_obj = mock_run_server.call_args[0][
            0
        ]  # First positional argument (Settings)

        assert settings_obj.host == "0.0.0.0"
        assert settings_obj.port == 9000
        assert settings_obj.log_level == "info"
        assert settings_obj.reload is False

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--log-level", "warning"])
    def test_main_log_level_argument(self, mock_run_server):
        """Test main function with custom log level."""
        main()

        mock_run_server.assert_called_once()
        settings_obj = mock_run_server.call_args[0][
            0
        ]  # First positional argument (Settings)

        assert settings_obj.host == "0.0.0.0"
        assert settings_obj.port == 8010
        assert settings_obj.log_level == "warning"
        assert settings_obj.reload is False

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--reload"])
    def test_main_reload_argument(self, mock_run_server):
        """Test main function with reload flag."""
        main()

        mock_run_server.assert_called_once()
        settings_obj = mock_run_server.call_args[0][
            0
        ]  # First positional argument (Settings)

        assert settings_obj.host == "0.0.0.0"
        assert settings_obj.port == 8010
        assert settings_obj.log_level == "info"
        assert settings_obj.reload is True

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--log-level", "invalid"])
    def test_main_invalid_log_level(self, mock_run_server):
        """Test main function with invalid log level."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2  # argparse error code
        mock_run_server.assert_not_called()

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--port", "invalid"])
    def test_main_invalid_port(self, mock_run_server):
        """Test main function with invalid port."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2  # argparse error code
        mock_run_server.assert_not_called()

    @patch("local_reranker.cli.validate_backend_type")
    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--backend", "mlx"])
    def test_main_unsupported_mlx_backend(
        self, mock_run_server, mock_validate_backend_type
    ):
        """Test main function with unsupported MLX backend."""
        mock_validate_backend_type.side_effect = ValueError("MLX unavailable")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2
        mock_run_server.assert_not_called()


def test_package_import_keeps_mlx_lazy():
    """Importing the package should not eagerly import MLX modules."""
    assert "local_reranker.jina_mlx_reranker" not in sys.modules


class TestArgumentParser:
    """Test cases for argument parsing logic."""

    def test_argument_parser_setup(self):
        """Test that the argument parser is set up correctly."""
        # Import the parser setup by testing main function behavior
        with patch("local_reranker.cli.run_server") as mock_run:
            with patch("sys.argv", ["local-reranker", "--help"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 0
                mock_run.assert_not_called()

    def test_all_log_level_choices(self):
        """Test that all expected log level choices are available."""
        expected_choices = ["debug", "info", "warning", "error", "critical"]

        # Test each valid log level
        for log_level in expected_choices:
            with patch("local_reranker.cli.run_server") as mock_run:
                with patch("sys.argv", ["local-reranker", "--log-level", log_level]):
                    main()
                    mock_run.assert_called_once()
                    settings_obj = mock_run.call_args[0][
                        0
                    ]  # First positional argument (Settings)
                    assert settings_obj.log_level == log_level

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--backend", "pytorch"])
    def test_main_reranker_argument(self, mock_run_server):
        """Test main function with custom backend."""
        main()

        mock_run_server.assert_called_once()
        args = mock_run_server.call_args[0][0]  # First positional argument (Settings)
        assert args.backend_type == "pytorch"

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--model", "custom-model-name"])
    def test_main_model_argument(self, mock_run_server):
        """Test main function with custom model."""
        main()

        mock_run_server.assert_called_once()
        args = mock_run_server.call_args[0][0]  # First positional argument (Settings)
        assert args.model_name == "custom-model-name"

    @patch("local_reranker.cli.run_server")
    @patch(
        "sys.argv",
        ["local-reranker", "--backend", "pytorch", "--model", "custom-model"],
    )
    def test_main_reranker_and_model_arguments(self, mock_run_server):
        """Test main function with both backend and model arguments."""
        main()

        mock_run_server.assert_called_once()
        args = mock_run_server.call_args[0][0]  # First positional argument (Settings)
        assert args.backend_type == "pytorch"
        assert args.model_name == "custom-model"

    @patch("builtins.print")
    @patch("sys.argv", ["local-reranker", "config", "show"])
    def test_main_config_show_command(self, mock_print):
        """Test main function with config show command."""
        main()  # Should not exit

        # Verify that print was called (config output)
        mock_print.assert_called()


class TestCLILogging:
    """Test CLI integration and logging."""

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker"])
    @patch("local_reranker.cli.logger")
    def test_main_logging(self, mock_logger, mock_run_server):
        """Test that main function logs startup message."""
        main()

        # Check that logger.info was called with startup message (among other calls)
        mock_logger.info.assert_any_call(
            "Starting Local Reranker API server on 0.0.0.0:8010"
        )
        mock_run_server.assert_called_once()

    @patch("local_reranker.cli.run_server")
    @patch("sys.argv", ["local-reranker", "--host", "127.0.0.1", "--port", "8000"])
    @patch("local_reranker.cli.logger")
    def test_main_logging_with_custom_host_port(self, mock_logger, mock_run_server):
        """Test that main function logs startup message with custom host and port."""
        main()

        mock_logger.info.assert_any_call(
            "Starting Local Reranker API server on 127.0.0.1:8000"
        )
        mock_run_server.assert_called_once()


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    def test_run_server_exception_propagation(self):
        """Test that exceptions from uvicorn.run are properly propagated."""
        settings = Settings()
        with patch(
            "local_reranker.cli.uvicorn.run", side_effect=Exception("Test error")
        ):
            with pytest.raises(Exception, match="Test error"):
                run_server(settings)

    @patch("sys.argv", ["local-reranker", "--nonexistent-argument"])
    def test_main_unknown_argument(self):
        """Test main function with unknown argument."""
        with patch("local_reranker.cli.run_server") as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 2  # argparse error code
            mock_run.assert_not_called()


class TestCLIIntegration:
    """Integration tests for CLI script execution."""

    @pytest.mark.integration
    def test_cli_script_help_output(self):
        """Test that CLI script outputs help correctly."""
        result = subprocess.run(
            [sys.executable, "-m", "local_reranker.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "Run the Local Reranker API server" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--log-level" in result.stdout
        assert "--reload" in result.stdout

    @pytest.mark.integration
    def test_cli_script_invalid_arguments(self):
        """Test CLI script with invalid arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "local_reranker.cli", "--invalid-arg"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "unrecognized arguments: --invalid-arg" in result.stderr

    @pytest.mark.integration
    def test_cli_script_invalid_log_level(self):
        """Test CLI script with invalid log level."""
        result = subprocess.run(
            [sys.executable, "-m", "local_reranker.cli", "--log-level", "invalid"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    @pytest.mark.integration
    def test_cli_script_invalid_port(self):
        """Test CLI script with invalid port."""
        result = subprocess.run(
            [sys.executable, "-m", "local_reranker.cli", "--port", "invalid"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "invalid int value" in result.stderr

    # @pytest.mark.integration
    # @pytest.mark.slow
    # def test_cli_server_startup_and_shutdown(self):
    #     """Test that CLI can start and shutdown server gracefully."""
    #     # Start the server process
    #     process = subprocess.Popen(
    #         [
    #             sys.executable,
    #             "-m",
    #             "local_reranker.cli",
    #             "--host",
    #             "127.0.0.1",
    #             "--port",
    #             "8011",
    #         ],
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.PIPE,
    #         text=True,
    #     )

    #     try:
    #         # Give the server time to start
    #         time.sleep(5)

    #         # Check if process is still running (server started successfully)
    #         stderr_output = process.stderr.read() if process.stderr else ""
    #         assert process.poll() is None, (
    #             f"Server failed to start. stderr: {stderr_output}"
    #         )

    #         # Test that server responds to health check

    #         try:
    #             response = httpx.get("http://127.0.0.1:8011/health", timeout=5.0)
    #             assert response.status_code == 200
    #             assert response.json() == {"status": "ok"}
    #         except httpx.RequestError as e:
    #             pytest.fail(f"Failed to connect to server: {e}")

    #     finally:
    #         # Clean up: terminate the server
    #         process.terminate()
    #         try:
    #             process.wait(timeout=10)
    #         except subprocess.TimeoutExpired:
    #             process.kill()
    #             process.wait()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_cli_entry_point_from_pyproject(self):
        """Test that the CLI entry point defined in pyproject.toml works."""
        # This test assumes the package is installed in development mode
        # First, let's check if the entry point script exists and is callable
        result = subprocess.run(
            ["local-reranker", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # If the command is not found, skip this test
        if result.returncode != 0 and "not found" in result.stderr:
            pytest.skip("local-reranker command not found - package not installed")

        assert result.returncode == 0
        assert "Run the Local Reranker API server" in result.stdout

    @pytest.mark.integration
    def test_cli_script_direct_execution(self):
        """Test running the CLI script directly."""
        script_path = "src/local_reranker/cli.py"

        result = subprocess.run(
            [sys.executable, "-m", "local_reranker.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "Run the Local Reranker API server" in result.stdout

    @pytest.mark.integration
    def test_cli_argument_combinations(self):
        """Test various CLI argument combinations don't cause parsing errors."""
        test_args = [
            ["--host", "127.0.0.1"],
            ["--port", "8000"],
            ["--log-level", "debug"],
            ["--reload"],
            [
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--log-level",
                "debug",
                "--reload",
            ],
        ]

        for args in test_args:
            # Test argument parsing by checking if help still works after adding args
            # We use --help to avoid actually starting the server
            full_args = ["--help"] + args
            result = subprocess.run(
                [sys.executable, "-m", "local_reranker.cli"] + full_args,
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Help should always work regardless of other valid arguments
            assert result.returncode == 0, (
                f"Failed for args: {args}. stderr: {result.stderr}"
            )

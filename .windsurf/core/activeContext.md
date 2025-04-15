# Active Context

*   **Current Focus**: Finalizing initial API setup and testing, Migrating dependency management from pip/requirements.txt to uv/pyproject.toml, Setting up basic testing infrastructure (`pytest`), Debugging dependency installation issues with `uv` and `torch`, Resolving virtual environment activation problems impacting test execution, Identifying necessary refactoring in `api.py` to replace deprecated `@app.on_event` with `lifespan`.
*   **State**: Project structure initialized, Core API (`api.py`) and Reranker logic (`reranker.py`) implemented, Dependencies managed via `uv` and `pyproject.toml`, Basic `/health` endpoint test created in `tests/test_api.py`, Test execution confirmed working *within* the activated virtual environment (`python -m pytest`), FastAPI deprecation warnings present due to `@app.on_event` usage, Manual refactoring to `lifespan` suggested due to tool limitations.
*   **Next Steps**: 
    *   Confirm user has manually applied the suggested `lifespan` refactoring to `api.py`.
    *   Re-run tests after refactoring to ensure warnings are resolved and functionality is intact.
    *   Add tests for the `/v1/rerank` endpoint.
    *   Update project documentation (README).

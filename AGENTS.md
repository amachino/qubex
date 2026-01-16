# Instructions for AI Agents

When working in this repository, it is critical to use the correct Python environment to avoid runtime errors or missing dependency issues.

## Python Environment

This project uses `uv` for dependency management. When running any Python commands (scripts, tests, tools, etc.), you must ensure the commands run within the project's virtual environment.

Please follow one of these approaches:

1. **Use `uv run` (Recommended):**
    Prefix your commands with `uv run`.
    * Example: `uv run pytest`
    * Example: `uv run python src/qubex/main.py`

2. **Activate the Virtual Environment:**
    If you are running multiple commands in a shell session, verify if `.venv` exists and activate it.
    * Command: `source .venv/bin/activate`

If you encounter `ModuleNotFoundError` or similar environment issues, please verify you are following the steps above.

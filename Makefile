.PHONY: sync test lint format clean build

# Install dependencies
sync:
	uv sync --dev --all-extras

# Run tests with pytest
test:
	uv run pytest

# Run type checker (pyright) and linter (ruff)
lint:
	uv run pyright
	uv run ruff check

# Format code with ruff
format:
	uv run ruff format
	uv run ruff check --fix

# Clean up cache and build artifacts
clean:
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache

# Build the package
build:
	uv build

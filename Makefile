.PHONY: sync test lint format typecheck check fix clean build

# Install dependencies
sync:
	@uv sync --dev --all-extras

# Run tests
test:
	@uv run pytest
# Run linting
lint:
	@uv run ruff check

# Format code
format:
	@uv run ruff format

# Type checking
typecheck:
	@uv run pyright

# Run static type checks and linting
check:
	@uv run pyright
	@uv run ruff format --check
	@uv run ruff check

# Fix format and lint issues
fix:
	@uv run ruff format
	@uv run ruff check --fix

# Clean up cache and build artifacts
clean:
	@rm -rf dist build *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .pytest_cache .ruff_cache .mypy_cache

# Build the package
build:
	@uv build

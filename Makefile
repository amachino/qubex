.PHONY: sync test coverage check fix lint lint-fix format format-check type-check clean build docs docs-serve docs-build docs-clean

# Install dependencies
sync:
	uv sync --all-groups --all-extras

# Run unit tests
test:
	uv run pytest

# Run tests with coverage
coverage:
	uv run pytest --cov=qubex --cov-report=term-missing

# Run type, lint, and format checks
check: type-check lint format-check

# Auto-fix formatting and lint issues
fix: format lint-fix

# Lint the codebase
lint:
	uv run ruff check

# Auto-fix lint issues
lint-fix:
	uv run ruff check --fix

# Format the codebase
format:
	uv run ruff format

# Check formatting without modifying files
format-check:
	uv run ruff format --check

# Run static type checking
type-check:
	uv run pyright

# Remove caches and build artifacts
clean:
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache

# Build distribution artifacts
build:
	uv build

# Build documentation site
docs: docs-build

# Serve documentation locally
docs-serve:
	uv run mkdocs serve

# Build documentation into the site/ directory
docs-build:
	uv run mkdocs build

# Remove generated documentation output
docs-clean:
	rm -rf site

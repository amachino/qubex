.PHONY: sync test coverage check fix lint lint-fix format format-check type-check clean build build-all publish-all docs docs-serve docs-build docs-clean

BUILD_CMD := uv run --with build python -m build
TWINE_CMD := uv run --with twine twine
REPOSITORY ?= testpypi
PUBLISH_TARGETS := \
	qxcore:packages/qxcore \
	qxvisualizer:packages/qxvisualizer \
	qxfitting:packages/qxfitting \
	qxpulse:packages/qxpulse \
	qxschema:packages/qxschema \
	qxsimulator:packages/qxsimulator \
	qxdriver-quel1:packages/qxdriver-quel1 \
	qubex:.

# Install dependencies
sync:
	git submodule update --init --recursive --remote
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
	rm -rf .cache .coverage .pytest_cache .ruff_cache .venv dist site
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Build distribution artifacts
build:
	$(BUILD_CMD)

# Build all publishable packages in dependency order
build-all:
	rm -rf dist/publish
	mkdir -p dist/publish
	@set -e; \
	for spec in $(PUBLISH_TARGETS); do \
		name=$${spec%%:*}; \
		path=$${spec#*:}; \
		outdir=dist/publish/$$name; \
		mkdir -p "$$outdir"; \
		$(BUILD_CMD) --outdir "$$outdir" "$$path"; \
	done

# Publish all package artifacts to the selected repository in dependency order
publish-all: build-all
	@set -e; \
	for spec in $(PUBLISH_TARGETS); do \
		name=$${spec%%:*}; \
		$(TWINE_CMD) upload --repository $(REPOSITORY) --skip-existing dist/publish/$$name/*; \
	done

# Build documentation site
docs: docs-build

# Serve documentation locally
docs-serve:
	uv run mkdocs serve --livereload --dirty

# Build documentation into the site/ directory
docs-build:
	uv run mkdocs build

# Remove generated documentation output
docs-clean:
	rm -rf site

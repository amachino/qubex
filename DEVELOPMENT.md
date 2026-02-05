# Development Guide

This document describes the development environment setup and workflow for the Qubex project.

## Development Environment Setup

### 1. Clone Repository and Basic Setup

```bash
git clone https://github.com/amachino/qubex.git
cd qubex

# Development environment setup with uv
uv sync --dev

# For Linux environments requiring backend support
uv sync --extra backend --dev
```

### 2. VS Code Configuration

The project includes VS Code configuration with the following recommended extensions:

- Python (ms-python.python)
- Ruff (charliermarsh.ruff)
- Even Better TOML (tamasfe.even-better-toml)
- markdownlint (davidanson.vscode-markdownlint)

VS Code settings are synchronized with pyproject.toml and automatically configure the appropriate development environment.

## Development Commands

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/pulse/test_pulse.py

# Run specific test function
uv run pytest tests/pulse/test_pulse.py::test_pulse_creation

# Run tests with coverage
uv run pytest --cov=qubex --cov-report=html
```

### Code Quality Checks

```bash
# Run linting with Ruff
uv run ruff check

# Auto-fix with Ruff
uv run ruff check --fix

# Run formatter
uv run ruff format

# Run all code quality checks
uv run ruff check --fix && uv run ruff format
```

### Package Building

```bash
# Build package
uv build

# Check build artifacts
ls dist/
```

### Performance Analysis

```bash
# Profile with pyinstrument
uv run pyinstrument your_script.py

# Or use within code
# from qubex.devtools.profile import profile
# @profile
# def your_function():
#     ...
```

## Project Structure

### Package Layout

The project uses `src/` layout with the following structure:

```
src/qubex/
├── analysis/           # Data analysis & visualization tools
├── backend/           # Hardware abstraction & system management
├── clifford/          # Clifford group operations (for benchmarking)
├── experiment/        # Experiment protocols & calibration
├── measurement/       # Measurement & state classification
├── pulse/             # Pulse generation & scheduling
├── simulator/         # Quantum system simulation
└── third_party/       # Third-party library wrappers
```

### Important Configuration Files

- `pyproject.toml` - Project configuration, dependencies, tool settings
- `uv.lock` - Dependency lock file
- `.vscode/settings.json` - VS Code settings (synchronized with pyproject.toml)
- `docs/examples/` - Usage examples and tutorials

## Development Workflow

### 1. Branching Strategy

- `main` - Stable releases
- `develop` - Development integration branch
- `feature/*` - New feature development
- `fix/*` - Bug fixes

### 2. Code Change Process

```bash
# 1. Create new branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/your-feature

# 2. Development & testing
# Edit code
uv run pytest              # Run tests
uv run ruff check --fix    # Fix linting
uv run ruff format         # Format code

# 3. Commit
git add .
git commit -m "feat: add new feature"

# 4. Push & create pull request
git push origin feature/your-feature
# Create pull request on GitHub
```

## Dependency Management

### Dependency Operations with uv

```bash
# Add new dependency
uv add package_name

# Add development dependency
uv add --dev package_name

# Add optional dependency
uv add --optional backend package_name

# Update dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package package_name
```

## Configuration Details

### pytest Configuration

The following is configured in pyproject.toml:

```toml
[tool.pytest.ini_options]
addopts = "--import-mode=importlib"
```

This ensures proper relative imports work with the `src/` layout.

### Ruff Configuration

```toml
[tool.ruff.lint]
ignore = ["E731", "E741"]  # Ignore lambda and variable name warnings
unfixable = ["F401"]       # Don't auto-fix unused imports
```

## Troubleshooting

### Common Issues

1. **Import errors**: Run `uv run pytest --import-mode=importlib` or re-run `uv sync`
2. **VS Code test execution failure**: Check pytest configuration in `.vscode/settings.json`
3. **Ruff settings not applied**: Restart VS Code or check `ruff.configuration` setting
4. **Dependency conflicts**: Try `uv lock --resolution=highest`

### Environment Cleanup

```bash
# Recreate virtual environment
uv sync --reinstall

# Clear cache
uv cache clean
```

## Common Commands Summary

```bash
# Development environment setup
uv sync --dev

# Testing & quality checks
uv run pytest
uv run ruff check --fix
uv run ruff format

# Add dependencies
uv add package_name
uv add --dev dev_package_name
```

## References

- [examples/](docs/examples/) - Usage examples for each component
- [package_structure.md](docs/notes/package_structure.md) - Package design guidelines
- [improvement_plan.md](docs/notes/improvement_plan.md) - Improvement plan

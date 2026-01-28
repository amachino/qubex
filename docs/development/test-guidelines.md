# Test Guidelines

This project uses **pytest**. Tests are required for all code changes and new features. Follow **test‑first development**: write a failing test, implement the behavior to pass, then refactor.

## Commands

- Run tests: `make test`
- Full quality gate: `make check`
- Optional coverage: `make coverage`

## Principles

- **Test before code**: add a failing test first, then implement.
- **Regression tests**: every bug fix must add a test that fails before the fix.
- **Behavior‑focused**: test observable behavior, not internal implementation details.
- **Small & fast**: keep unit tests quick and deterministic.
- **Lightweight BDD**: use a 1‑line spec docstring and clear test names.

## Structure & Naming

- Place tests under `tests/` and mirror the `src/` layout where appropriate.
- File names: `test_*.py`
- Function names: `test_*`
- Class names (if used): `Test*`
- One behavior per test; keep assertions focused and readable.

## Arrange‑Act‑Assert

Structure each test clearly:

```python
def test_example():
    """Given X, when Y, then Z."""
    # Arrange
    obj = make_example()

    # Act
    result = obj.compute()

    # Assert
    assert result == 42
```

## Assertions & Numeric Tolerances

- Use plain `assert` (pytest rewrites for better diffs).
- For floating‑point comparisons, use `numpy.testing` or `pytest.approx`.
- Always state tolerances explicitly for numeric results.

```python
from numpy.testing import assert_allclose

assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
```

## Parameterization & Edge Cases

- Prefer `@pytest.mark.parametrize` to reduce duplication.
- Cover normal cases, boundaries, and invalid inputs.

## Fixtures

- Use fixtures for setup/teardown and shared data.
- Put shared fixtures in `tests/**/conftest.py`.
- Keep fixture scope as small as possible.
- Keep randomness deterministic (seed RNGs inside fixtures).

## Errors, Warnings, and Logging

- Use `pytest.raises` and `pytest.warns` for error paths.
- Use `caplog` for log assertions and `capsys` for stdout/stderr.

## IO, Network, and Hardware

- Prefer `tmp_path` for file IO and avoid writing to the repo.
- Tests should not rely on network or hardware by default.
- If external dependencies are unavoidable, **skip with a clear reason**.

```python
import pytest

pytest.skip("requires hardware")
```

## Import Behavior

Pytest runs with `--import-mode=importlib`. Avoid sys.path hacks and rely on package imports from `src/`.

## Minimal Test Template

```python
def test_behavior_name():
    """Given X, when Y, then Z."""
    # Arrange

    # Act

    # Assert
    assert ...
```

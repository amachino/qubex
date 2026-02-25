# Test guidelines

This project uses **pytest**. Tests are required for all code changes and new features. Follow **test‑first development**: write a failing test, implement the behavior to pass, then refactor.

## Commands

- Run tests: `make test`
- Full quality gate: `make check`
- Optional coverage: `make coverage`
- Verify test collection: `uv run pytest --collect-only -q tests`

## Principles

- **Test before code**: add a failing test first, then implement.
- **Regression tests**: every bug fix must add a test that fails before the fix.
- **Behavior‑focused**: test observable behavior, not internal implementation details.
- **Small & fast**: keep unit tests quick and deterministic.
- **Lightweight BDD**: use a 1‑line spec docstring and clear test names.

## Structure & naming

- Place tests under `tests/` and mirror the `src/` layout where appropriate.
- File names: `test_*.py` (required; non-matching names may not be collected)
- Function names: `test_*`
- Class names (if used): `Test*`
- One behavior per test; keep assertions focused and readable.

## Collection safety

- Keep all runnable tests in files matching `test_*.py`.
- When adding or renaming tests, run collection check and confirm expected tests appear.
- Do not rely on direct file-path invocation to run tests that normal `pytest` collection misses.

## Arrange‑act‑assert

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

## Assertions & numeric tolerances

- Use plain `assert` (pytest rewrites for better diffs).
- For floating‑point comparisons, use `numpy.testing` or `pytest.approx`.
- Always state tolerances explicitly for numeric results.
- For success-path tests, assert an observable postcondition (return value, state change, call args, or output).
- Do not use "no exception raised" as the only assertion unless the API contract is explicitly "must not raise."

```python
from numpy.testing import assert_allclose

assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
```

## Parameterization & edge cases

- Prefer `@pytest.mark.parametrize` to reduce duplication.
- Cover normal cases, boundaries, and invalid inputs.
- Consolidate repeated input/output tables into parameterized tests.

## Fixtures

- Use fixtures for setup/teardown and shared data.
- Put shared fixtures in `tests/**/conftest.py`.
- Keep fixture scope as small as possible.
- Keep randomness deterministic (seed RNGs inside fixtures).
- Isolate global/singleton state changes and restore state in fixtures.

## Determinism & timing

- Avoid `time.sleep()` in tests whenever possible.
- Prefer deterministic synchronization (`Event`, mock clock, monkeypatch, polling with bounded timeout).
- Keep waits short and explicit; document why a timeout value is safe.

## Errors, warnings, and logging

- Use `pytest.raises` and `pytest.warns` for error paths.
- Use `caplog` for log assertions and `capsys` for stdout/stderr.

## IO, network, and hardware

- Prefer `tmp_path` for file IO and avoid writing to the repo.
- Tests should not rely on network or hardware by default.
- If external dependencies are unavoidable, **skip with a clear reason**.

```python
import pytest

pytest.skip("requires hardware")
```

## Public API focus

- Prefer testing via public APIs and observable behavior.
- Avoid coupling tests to private members (`_internal`) unless validating a required compatibility contract.
- If a private contract must be tested, keep scope narrow and document why public API cannot cover it.

## Import behavior

Pytest runs with `--import-mode=importlib`. Avoid sys.path hacks and rely on package imports from `src/`.

## Minimal test template

```python
def test_behavior_name():
    """Given X, when Y, then Z."""
    # Arrange

    # Act

    # Assert
    assert ...
```

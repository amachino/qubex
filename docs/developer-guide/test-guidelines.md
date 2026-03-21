# Test guidelines

This project uses **pytest**. Tests are required for all code changes and new features. Follow **test‑first development**: write a failing test, implement the behavior to pass, then refactor.

## Commands

- Run tests: `uv run pytest`
- Lint: `uv run ruff check`
- Format: `uv run ruff format`
- Type check: `uv run pyright`
- Full quality gate: `uv run ruff check && uv run pyright && uv run pytest`

## Principles

- **Test before code**: add a failing test first, then implement.
- **Regression tests**: every bug fix must add a test that fails before the fix.
- **Behavior‑focused**: test observable behavior, not internal implementation details.
- **Small & fast**: keep unit tests quick and deterministic.
- **Docstring readability first**: use a 1-line spec docstring that is natural and easy to scan.
- **Style is flexible**: `Given/when/then` is recommended for scenario-style tests, but not required.
- **Local consistency**: keep docstring style consistent within the same file/module.

## Structure & naming

- Place tests under `tests/` and mirror the `src/` layout where appropriate.
- File names: `test_*.py`
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
    """Given valid input, when compute is called, then the expected value is returned."""
    # Arrange
    obj = make_example()

    # Act
    result = obj.compute()

    # Assert
    assert result == 42
```

Both of these docstring styles are acceptable:

```python
"""Given labels added in insertion order, when building PulseSchedule, then label order is preserved."""
"""PulseSchedule should preserve label insertion order."""
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
- Do not hard-code temporary paths such as `/tmp/...` in tests. Ruff flags these as `S108`.
- Build temporary files and directories from `tmp_path` or `tmp_path_factory` instead.
- Tests should not rely on network or hardware by default.
- If external dependencies are unavoidable, **skip with a clear reason**.

```python
from pathlib import Path

def test_example(tmp_path: Path) -> None:
    """Given file output, when writing test data, then tmp_path should be used."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    assert config_dir.exists()
```

```python
import pytest

pytest.skip("requires hardware")
```

## Public API focus

- Prefer testing via public APIs and observable behavior.
- Avoid coupling tests to private members (`_internal`) unless validating a required compatibility contract.
- If a private contract must be tested, keep scope narrow and document why public API cannot cover it.
- Avoid adding tests that only assert symbol absence/presence via `hasattr`.
- Add `hasattr`-based checks only when API surface itself is an explicit compatibility contract.
- When API-surface checks are required, keep them minimal and pair them with behavior tests for user-visible outcomes.

## Import behavior

Pytest runs with `--import-mode=importlib`. Avoid sys.path hacks and rely on package imports from `src/`.

## Minimal test template

```python
def test_behavior_name():
    """One-line behavior spec."""
    # Arrange

    # Act

    # Assert
    assert ...
```

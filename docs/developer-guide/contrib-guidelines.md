# Contrib module guidelines

`src/qubex/contrib/` is for community-contributed experimental features.
Use this area for utilities and workflows that are useful but not yet part of the core stable API.

## When to use `contrib`

Use `contrib` when at least one is true:

- The feature is experimental or iteration speed is prioritized.
- The API may evolve based on user feedback.
- The capability is specialized and not required by most users.

Prefer core modules when behavior is stable and broadly applicable.

## Module design

- Keep module boundaries clear: one topic per file.
- Expose user-facing functions from `src/qubex/contrib/__init__.py`.
- Add new exports to `__all__` in `src/qubex/contrib/__init__.py`.
- Avoid hidden side effects at import time.

## API expectations

- Use clear, typed function signatures.
- Follow NumPy-style docstrings.
- Document assumptions, required context, and hardware constraints.
- Raise explicit errors for invalid inputs.

## Tests

- Add tests under `tests/contrib/<module_name>/`.
- Mirror exported functions with functional API tests.
- Use a one-line spec docstring for each test.
- Add regression tests for bug fixes.

## Promotion path

If a `contrib` feature becomes stable and broadly used:

1. Propose migration to a core package.
2. Define compatibility strategy for existing imports.
3. Update docs and deprecation notices if paths change.

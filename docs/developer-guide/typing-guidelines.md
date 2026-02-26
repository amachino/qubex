# Typing guidelines

This page defines typing conventions used in Qubex code.

## Naming

See [Naming guidelines](naming-guidelines.md#typevar-naming) for `TypeVar` naming and declaration rules.

## Compatibility note

Qubex currently supports Python `>=3.10`.
When writing runtime code that must remain compatible with Python 3.10 and 3.11,
use `TypeVar`.

## Dynamic attribute access (`getattr`)

Prefer explicit attribute and method access over dynamic `getattr`-based dispatch.
Use `getattr` only when attribute names are selected at runtime and cannot be represented clearly with static code.

### Appropriate use cases

- Plugin or command systems where handler names come from validated runtime input.
- Compatibility shims that probe optional attributes across dependency versions.
- Generic proxy/wrapper utilities that intentionally forward attributes dynamically.

### Avoid `getattr` when

- The target attribute or method is known at development time.
- A small finite branch is clearer as dictionary dispatch or `if`/`match`.
- The code is core domain logic where static typing and refactor safety are priorities.

### Required safeguards

- Constrain attribute names with an explicit allowlist.
- Raise explicit errors for unsupported names.
- Do not silently swallow missing attributes with broad defaults unless the fallback is explicitly required.
- Add tests for both accepted and rejected attribute names.

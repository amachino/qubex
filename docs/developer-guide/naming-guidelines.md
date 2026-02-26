# Naming guidelines

This page defines naming conventions used in Qubex code.

## Module symbol visibility

- Use `__all__` to define the module's intended public API.
- Use leading underscore names (for example, `_INTERNAL_CONSTANT`) for non-public module symbols.
- Do not rely on `__all__` alone for internal symbols; keep internal intent explicit with leading underscore naming.

## Constants

- Use `UPPER_SNAKE_CASE` for constants.

## Count values

- For values that represent counts, prefer `n_<plural_noun>` naming (for example, `n_shots`, `n_trials`, `n_qubits`).

## TypeVar naming

Use the following naming policy for `TypeVar`.

- Generic type variables: `T`
- Mapping-related type variables: `K`, `V`
- Domain-specific type variables: `<Name>T` (for example, `OptionT`)
- Variance suffixes: `_co` and `_contra`

### TypeVar examples

```python
from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
OptionT = TypeVar("OptionT")
ResultT_co = TypeVar("ResultT_co", covariant=True)
ConsumerT_contra = TypeVar("ConsumerT_contra", contravariant=True)
```

### TypeVar additional rules

- Define type variables explicitly in the local module.
- Do not rely on implicit shared names such as `typing_extensions.T` or `typing.T`.
- Keep naming style consistent within a module.

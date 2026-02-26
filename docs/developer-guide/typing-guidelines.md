# Typing guidelines

This page defines typing conventions used in Qubex code.

## TypeVar naming

Use the following naming policy for `TypeVar`.

- Generic type variables: `T`
- Mapping-related type variables: `K`, `V`
- Domain-specific type variables: `<Name>T` (for example, `OptionT`)
- Variance suffixes: `_co` and `_contra`

## Examples

```python
from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
OptionT = TypeVar("OptionT")
ResultT_co = TypeVar("ResultT_co", covariant=True)
ConsumerT_contra = TypeVar("ConsumerT_contra", contravariant=True)
```

## Additional rules

- Define type variables explicitly in the local module.
- Do not rely on implicit shared names such as `typing_extensions.T` or `typing.T`.
- Keep naming style consistent within a module.

## Compatibility note

Qubex currently supports Python `>=3.10`.
When writing runtime code that must remain compatible with Python 3.10 and 3.11,
use `TypeVar`.

# Typing guidelines

This page defines typing conventions used in Qubex code.

## Generic type parameter names

- Prefer short canonical names for generic type parameters: `T`, `U`, `K`, `V`.
- Use descriptive names only when a single-letter name would harm readability.
- Avoid suffix-only names such as `SequencerT` in new code.

### Preferred pattern with `TypeVar`

```python
from typing import TypeVar

T = TypeVar("T")
```

### Preferred pattern with Python 3.12 type parameters

```python
def max[T](args: Iterable[T]) -> T:
    ...


class list[T]:
    def __getitem__(self, index: int, /) -> T:
        ...

    def append(self, element: T) -> None:
        ...
```

## Compatibility note

Qubex currently supports Python `>=3.10`.
When writing runtime code that must remain compatible with Python 3.10 and 3.11, use `TypeVar`.
The naming convention above still applies.

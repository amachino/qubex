# Typing guidelines

This page defines typing conventions used in Qubex code.

## Naming

See [Naming guidelines](naming-guidelines.md#typevar-naming) for `TypeVar` naming and declaration rules.

## Compatibility note

Qubex currently supports Python `>=3.10`.
When writing runtime code that must remain compatible with Python 3.10 and 3.11,
use `TypeVar`.

## Sentinel values (`MISSING`)

Use the shared sentinel from `qubex.core.sentinel` when an API must distinguish
between:

- argument omitted
- argument explicitly set to `None`

Prefer the shared `MISSING` sentinel over ad hoc `object()` instances.

### When to use `MISSING`

Use `MISSING` when `None` is already a meaningful value and you need a third
state for "not provided".

Typical cases:

- optional keyword arguments with inheritance or fallback behavior
- deprecated-option normalization
- compatibility APIs that need to preserve legacy omission semantics

Do not use `MISSING` when `None` already means "use the default" and there is no
observable difference between omitted and `None`.

### Annotation pattern

Annotate these parameters with `typing_extensions.Sentinel`.

```python
from typing_extensions import Sentinel

from qubex.core.sentinel import MISSING


def configure(
    *,
    figure: Figure | None | Sentinel = MISSING,
) -> None:
    ...
```

Avoid broad fallback annotations such as `object` for this pattern. They hide
intent and weaken type checking.

### Comparison rule

Check the sentinel by identity.

```python
if value is MISSING:
    ...
```

Do not compare sentinels by equality.

### Narrowing pattern

When a helper improves readability, use a local `TypeGuard` to narrow away
`Sentinel`.

```python
from typing import TypeGuard, TypeVar
from typing_extensions import Sentinel

from qubex.core.sentinel import MISSING

T = TypeVar("T")


def _is_given(value: T | Sentinel) -> TypeGuard[T]:
    return value is not MISSING
```

This is the preferred pattern when a value is read multiple times after the
missing check.

### Defining new sentinels

Prefer reusing the shared `MISSING` sentinel.

Create a new sentinel with `make_sentinel()` only when a separate sentinel name
is part of a real API contract and sharing `MISSING` would make behavior less
clear.

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

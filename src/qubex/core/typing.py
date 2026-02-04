"""Shared typing helpers for core models."""

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy.typing as npt

if TYPE_CHECKING:
    import tunits

    # tunits.ValueArray is generic only in type stubs; runtime class is not subscriptable.
    # Keep generics for static type checking while avoiding TypeError at runtime.
    ValueArray = tunits.ValueArray[Any]
else:
    try:
        import tunits as _tunits

        ValueArray = _tunits.ValueArray
    except ModuleNotFoundError:  # pragma: no cover - optional dependency path.
        ValueArray = Any

ValueArrayLike: TypeAlias = ValueArray | npt.NDArray | list | tuple

"""Shared typing helpers for core models."""

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy.typing as npt
import tunits

if TYPE_CHECKING:
    # tunits.ValueArray is generic only in type stubs; runtime class is not subscriptable.
    # Keep generics for static type checking while avoiding TypeError at runtime.
    ValueArray = tunits.ValueArray[Any]
else:
    ValueArray = tunits.ValueArray

ValueArrayLike: TypeAlias = ValueArray | npt.NDArray | list | tuple

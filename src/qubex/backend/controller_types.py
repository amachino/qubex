"""Shared controller type aliases for backend-family selection."""

from __future__ import annotations

from typing import Literal, TypeAlias

from .quel1 import Quel1BackendController
from .quel3 import Quel3BackendController

BackendKind = Literal["quel1", "quel3"]
BackendController: TypeAlias = Quel1BackendController | Quel3BackendController


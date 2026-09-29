"""Check optional installations without suppressing driver import failures."""

from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import pytest


def require_distributions(*names: str) -> None:
    """Skip when a named distribution is absent, including during collection."""
    try:
        for name in names:
            version(name)
    except PackageNotFoundError as exc:
        pytest.skip(
            f"Optional backend dependency is not installed: {exc.name}",
            allow_module_level=True,
        )


def require_quel1_backend() -> Literal["qubecalib", "qxdriver_quel1"]:
    """Require the QuEL-1 driver selected by the installed quelware version."""
    require_distributions("quel-ic-config")
    if version("quel-ic-config").split(".")[:2] == ["0", "8"]:
        require_distributions("qubecalib")
        return "qubecalib"
    require_distributions("qxdriver-quel1")
    return "qxdriver_quel1"

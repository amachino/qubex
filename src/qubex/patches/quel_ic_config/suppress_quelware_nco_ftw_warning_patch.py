"""Suppress quelware NCO FTW validator warnings under recent `pydantic`."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def _is_quelware_0_10_or_later() -> bool:
    """Return whether installed `quel_ic_config` is 0.10 or later."""
    try:
        v = version("quel_ic_config")
    except PackageNotFoundError:
        return False

    parts = v.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return False

    major, minor = int(parts[0]), int(parts[1])
    return major > 0 or (major == 0 and minor >= 10)


def _patch_abstract_nco_ftw_init(ad9082_nco_module: Any) -> None:
    """Patch `AbstractNcoFtw.__init__` to bypass `pydantic`'s warning path."""
    abstract_nco_ftw_cls = getattr(ad9082_nco_module, "AbstractNcoFtw", None)
    if abstract_nco_ftw_cls is None:
        return
    if getattr(abstract_nco_ftw_cls, "__qubex_nco_ftw_warning_patch_applied__", False):
        return

    def _init(self: Any, /, **data: Any) -> None:
        self.__pydantic_validator__.validate_python(data, self_instance=self)

    _init.__pydantic_base_init__ = True
    abstract_nco_ftw_cls.__init__ = _init
    abstract_nco_ftw_cls.__qubex_nco_ftw_warning_patch_applied__ = True


def apply_quelware_nco_ftw_warning_patch() -> None:
    """Apply quelware NCO FTW warning suppression patch for quelware 0.10+."""
    if not _is_quelware_0_10_or_later():
        return

    try:
        ad9082_nco_module = importlib.import_module("quel_ic_config.ad9082_nco")
    except ImportError:
        return

    _patch_abstract_nco_ftw_init(ad9082_nco_module)


apply_quelware_nco_ftw_warning_patch()

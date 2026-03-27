# ruff: noqa: SLF001

"""Patch e7awghal capture-buffer allocation for DSP classification."""

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


def _patch_capunit_simplified(capunit_module: Any) -> None:
    """Patch `CapUnitSimplified` to allocate integer-sized 2-bit buffers."""
    capunit_cls = getattr(capunit_module, "CapUnitSimplified", None)
    minimum_align = getattr(capunit_module, "_CAP_MINIMUM_ALIGN", None)
    if capunit_cls is None or minimum_align is None:
        return
    if getattr(capunit_cls, "__qubex_classification_buffer_patch_applied__", False):
        return

    original_allocate_read_buffer = getattr(capunit_cls, "_allocate_read_buffer", None)
    if original_allocate_read_buffer is None:
        return

    def _allocate_read_buffer(self: Any, **kwargs):
        if self._current_param is None:
            raise AssertionError("_allocate_read_buffer() requires self._current_param")
        if not self._current_param.classification_enable:
            return original_allocate_read_buffer(self, **kwargs)

        sample_count = int(self._current_param.get_datasize_in_sample())
        bufsize = (sample_count + 3) // 4
        return self._mm.allocate(bufsize, minimum_align=minimum_align, **kwargs)

    capunit_cls._allocate_read_buffer = _allocate_read_buffer
    capunit_cls.__qubex_classification_buffer_patch_applied__ = True


def apply_e7awghal_classification_buffer_patch() -> None:
    """Apply the e7awghal classification-buffer compatibility patch."""
    if not _is_quelware_0_10_or_later():
        return

    try:
        capunit_module = importlib.import_module("e7awghal.capunit")
    except ImportError:
        return

    _patch_capunit_simplified(capunit_module)


apply_e7awghal_classification_buffer_patch()

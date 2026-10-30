"""Patch helpers for e7awghal capture buffer allocation."""
# ruff: noqa: SLF001

from __future__ import annotations

from typing import Any

import numpy as np


def apply_e7awghal_capture_buffer_patch() -> None:
    """
    Patch e7awghal capture buffer handling for classified captures.

    e7awghal can pass a NumPy floating scalar to the memory manager for a
    classified buffer. Normalize only that classified allocation size. Normal
    captures delegate directly to the original implementation.
    """
    try:
        from e7awghal import capunit
    except ImportError:
        return

    capunit_cls = getattr(capunit, "CapUnitSimplified", None)
    if capunit_cls is None:
        return
    buffer_patch_applied = getattr(
        capunit_cls,
        "__qubex_capture_buffer_patch_applied__",
        False,
    )

    original_allocate_read_buffer = getattr(
        capunit_cls,
        "__qubex_original_allocate_read_buffer__",
        capunit_cls._allocate_read_buffer,
    )

    def _allocate_read_buffer(self: Any, **kwargs: Any) -> Any:
        current_param = self._current_param
        if current_param is None or not current_param.classification_enable:
            return original_allocate_read_buffer(self, **kwargs)

        bufsize = int(np.ceil(current_param.get_datasize_in_sample() / 4))
        minimum_align = capunit.__dict__["_CAP_MINIMUM_ALIGN"]
        return self._mm.allocate(
            bufsize,
            minimum_align=minimum_align,
            **kwargs,
        )

    if not buffer_patch_applied:
        capunit_cls._allocate_read_buffer = _allocate_read_buffer
        capunit_cls.__qubex_capture_buffer_patch_applied__ = True
        capunit_cls.__qubex_original_allocate_read_buffer__ = (
            original_allocate_read_buffer
        )

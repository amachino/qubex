"""Patch helpers for e7awghal capture buffer allocation."""
# ruff: noqa: SLF001

from __future__ import annotations

from typing import Any

import numpy as np


def apply_e7awghal_capture_buffer_patch() -> None:
    """
    Patch e7awghal capture buffer handling for classified captures.

    e7awghal can pass NumPy floating scalars to the memory manager for
    classified buffers.  Keep the upstream behavior, but normalize byte size
    and capture address to Python ints.
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
        if self._current_param is None:
            raise AssertionError("_allocate_read_buffer() requires self._current_param")

        if self._current_param.classification_enable:
            bufsize = int(np.ceil(self._current_param.get_datasize_in_sample() / 4))
            minimum_align = capunit.__dict__["_CAP_MINIMUM_ALIGN"]
            return self._mm.allocate(
                bufsize,
                minimum_align=minimum_align,
                **kwargs,
            )
        return original_allocate_read_buffer(self, **kwargs)

    def _set_cap_param_address_only(self: Any, mobj: Any) -> None:
        capture_address = (int(self._mm._address_offset) + int(mobj.address_top)) >> 5
        self._write_param_reg(
            self._PARAM_CAPTURE_ADDRESS_REG_ADDR,
            np.uint32(capture_address),
        )

    if not buffer_patch_applied:
        capunit_cls._allocate_read_buffer = _allocate_read_buffer
        capunit_cls._set_cap_param_address_only = _set_cap_param_address_only
        capunit_cls.__qubex_capture_buffer_patch_applied__ = True
        capunit_cls.__qubex_original_allocate_read_buffer__ = (
            original_allocate_read_buffer
        )

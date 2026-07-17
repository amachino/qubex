"""Tests for e7awghal capture buffer allocation patch."""
# ruff: noqa: SLF001

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.patches.quel_ic_config import e7awghal_capture_buffer_patch


def test_apply_e7awghal_capture_buffer_patch_does_not_fail_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given e7awghal is unavailable, when applying the patch, then it is skipped."""
    monkeypatch.delitem(sys.modules, "e7awghal", raising=False)
    monkeypatch.delitem(sys.modules, "e7awghal.capunit", raising=False)

    e7awghal_capture_buffer_patch.apply_e7awghal_capture_buffer_patch()


def test_apply_e7awghal_capture_buffer_patch_casts_classified_buffer_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given classified capture allocation, when patched, then buffer size is an int."""

    class FakeParam:
        """Fake capture parameter with classification enabled."""

        classification_enable = True

        def get_datasize_in_sample(self) -> int:
            return 5

    class FakeMemoryManager:
        """Fake memory manager recording the requested allocation."""

        def __init__(self) -> None:
            self.bufsize: object | None = None
            self.minimum_align: object | None = None
            self._address_offset = np.float64(0.0)

        def allocate(self, bufsize: object, *, minimum_align: object) -> object:
            self.bufsize = bufsize
            self.minimum_align = minimum_align
            return SimpleNamespace(address_top=np.float64(96.0))

    class FakeCapUnitSimplified:
        """Fake capunit exposing the e7awghal methods patched by Qubex."""

        _PARAM_CAPTURE_ADDRESS_REG_ADDR = 0x10

        def __init__(self) -> None:
            self._current_param = FakeParam()
            self._mm = FakeMemoryManager()
            self.written: tuple[int, np.uint32] | None = None

        def _allocate_read_buffer(self, **kwargs: object) -> object:
            raise AssertionError("The original classified allocator should be patched")

        def _set_cap_param_address_only(self, mobj: Any) -> None:
            capture_address = (self._mm._address_offset + mobj.address_top) >> 5
            self._write_param_reg(
                self._PARAM_CAPTURE_ADDRESS_REG_ADDR,
                np.uint32(capture_address),
            )

        def _write_param_reg(self, address: int, value: np.uint32) -> None:
            self.written = (address, value)

    fake_e7awghal = cast(Any, types.ModuleType("e7awghal"))
    fake_capunit = cast(Any, types.ModuleType("e7awghal.capunit"))
    fake_capunit.CapUnitSimplified = FakeCapUnitSimplified
    fake_capunit._CAP_MINIMUM_ALIGN = 32
    fake_e7awghal.capunit = fake_capunit
    monkeypatch.setitem(sys.modules, "e7awghal", fake_e7awghal)
    monkeypatch.setitem(sys.modules, "e7awghal.capunit", fake_capunit)

    e7awghal_capture_buffer_patch.apply_e7awghal_capture_buffer_patch()
    unit = FakeCapUnitSimplified()
    unit._mm._address_offset = np.float64(64.0)

    mobj = unit._allocate_read_buffer()
    unit._set_cap_param_address_only(mobj)

    assert isinstance(unit._mm.bufsize, int)
    assert unit._mm.bufsize == 2
    assert unit._mm.minimum_align == 32
    assert unit.written == (0x10, np.uint32(5))

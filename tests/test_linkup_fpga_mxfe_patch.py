"""Tests for quel_ic_config linkup FPGA MxFE patch."""

from __future__ import annotations

import sys
import types

import pytest

from qubex.backend.quel1.quel1_backend_constants import (
    DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT,
)
from qubex.patches.quel_ic_config import linkup_fpga_mxfe_patch


def test_apply_linkup_fpga_mxfe_patch_sets_reconnect_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quel_ic_config is importable, when patch applies, then reconnect threshold is restored."""
    linkup_class_attr = "LinkupFpgaMxfe"
    threshold_attr = "_DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT"

    class FakeLinkupFpgaMxfe:
        """Fake LinkupFpgaMxfe class with a patchable reconnect threshold."""

        _DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT = 100000.0

    fake_module = types.ModuleType("quel_ic_config")
    setattr(fake_module, linkup_class_attr, FakeLinkupFpgaMxfe)
    monkeypatch.setitem(sys.modules, "quel_ic_config", fake_module)

    linkup_fpga_mxfe_patch.apply_linkup_fpga_mxfe_patch()

    assert (
        getattr(FakeLinkupFpgaMxfe, threshold_attr)
        == DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT
    )

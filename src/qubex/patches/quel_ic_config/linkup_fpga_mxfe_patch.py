"""Patch helpers for link-up of FPGA MxFE configuration."""

from __future__ import annotations


def apply_linkup_fpga_mxfe_patch() -> None:
    """Patch default background-noise threshold used at reconnect."""
    try:
        from quel_ic_config import LinkupFpgaMxfe
    except ImportError:
        return

    LinkupFpgaMxfe._DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT = 100000  # noqa: SLF001  # type: ignore[attr-defined]

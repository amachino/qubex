"""Patch helpers for link-up of FPGA MxFE configuration."""

from __future__ import annotations

from qubex.backend.quel1.quel1_backend_constants import (
    DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT,
)


def apply_linkup_fpga_mxfe_patch() -> None:
    """
    Patch the noise threshold used when qubecalib calls `quel_ic_config`.

    Notes
    -----
    This allows Qubex to control the default reconnect background-noise
    threshold through qubecalib. Remove this patch once qubecalib no longer
    depends on `quel_ic_config` for that path.
    """
    try:
        from quel_ic_config import LinkupFpgaMxfe
    except ImportError:
        return

    LinkupFpgaMxfe._DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT = (  # noqa: SLF001  # type: ignore[attr-defined]
        DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT
    )

"""Patches for Quel IC configuration tooling."""

from __future__ import annotations

from .disable_quelware_filelock_patch import apply_quelware_filelock_patch
from .e7awghal_classification_buffer_patch import (
    apply_e7awghal_classification_buffer_patch,
)
from .linkup_fpga_mxfe_patch import apply_linkup_fpga_mxfe_patch
from .suppress_duplicated_proxy_patch import apply_quelware_duplicated_proxy_patch
from .suppress_quelware_nco_ftw_warning_patch import (
    apply_quelware_nco_ftw_warning_patch,
)


def apply_quelware_runtime_patches() -> None:
    """Apply runtime patches needed by qubex for quelware integration."""
    apply_quelware_filelock_patch()
    apply_quelware_duplicated_proxy_patch()
    apply_quelware_nco_ftw_warning_patch()
    apply_e7awghal_classification_buffer_patch()
    apply_linkup_fpga_mxfe_patch()

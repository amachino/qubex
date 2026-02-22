"""Skew calibration manager for QuEL-1 backend controller."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContextReader

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        SkewRuntimeProtocol,
    )


class Quel1SkewManager:
    """Handle skew YAML loading and skew measurement operations for QuEL-1."""

    def __init__(self, *, runtime_context: Quel1RuntimeContextReader) -> None:
        self._runtime_context = runtime_context

    def load_skew_yaml(self, file_path: str | Path) -> None:
        """Load skew calibration YAML into qubecalib system database."""
        self._runtime_context.qubecalib.sysdb.load_skew_yaml(str(file_path))

    def run_skew_measurement(
        self,
        *,
        skew_yaml_path: str | Path,
        box_yaml_path: str | Path,
        clockmaster_ip: str,
        box_names: list[str],
        estimate: bool = True,
    ) -> tuple[SkewRuntimeProtocol, Any]:
        """Run skew measurement workflow and return skew runtime and plot figure."""
        skew = self._runtime_context.driver.Skew.from_yaml(
            str(skew_yaml_path),
            box_yaml=str(box_yaml_path),
            clockmaster_ip=clockmaster_ip,
            boxes=box_names,
        )
        skew.system.resync()
        skew.measure()
        if estimate:
            skew.estimate()
        fig = skew.plot()
        return skew, fig

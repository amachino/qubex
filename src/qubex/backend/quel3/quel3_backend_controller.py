"""Quel3 backend controller scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from qubex.backend.quel1 import SAMPLING_PERIOD, Quel1BackendController


class Quel3BackendController(Quel1BackendController):
    """
    Quel3 controller scaffold with measurement-layer capability hints.

    Notes
    -----
    This class intentionally reuses the existing QuEL-1 control-plane
    implementation for shared configuration operations while exposing
    QuEL-3-specific measurement capability metadata.
    """

    MEASUREMENT_BACKEND_KIND: Literal["quel3"] = "quel3"
    MEASUREMENT_CONSTRAINT_MODE: Literal["relaxed"] = "relaxed"
    MEASUREMENT_RESULT_AVG_SAMPLE_STRIDE: int = 4
    DEFAULT_SAMPLING_PERIOD: float = float(SAMPLING_PERIOD)

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        sampling_period_ns: float | None = None,
    ) -> None:
        """
        Initialize a Quel3 controller scaffold.

        Parameters
        ----------
        config_path : str | Path | None, optional
            Optional config path passed to the shared base controller.
        sampling_period_ns : float | None, optional
            Session sampling period used by measurement-layer adapters.
        """
        super().__init__(config_path=config_path)
        if sampling_period_ns is not None:
            self.DEFAULT_SAMPLING_PERIOD = float(sampling_period_ns)

    def resolve_instrument_alias(self, target: str) -> str:
        """Resolve quelware instrument alias for a measurement target."""
        return target

    def execute_measurement(self, *, payload: object) -> object:
        """
        Execute a Quel3 measurement payload.

        Parameters
        ----------
        payload : object
            Backend execution payload produced by the measurement adapter.
        """
        raise NotImplementedError(
            "Quel3 execution is not wired yet. Implement execute_measurement() using quelware."
        )

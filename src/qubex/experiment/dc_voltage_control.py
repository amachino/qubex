"""Bound DC voltage control for one experiment mux."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace

from qubex.external_devices import DCVoltageProfile

from .models.dc_voltage_state import DCVoltageState


class DCVoltageControl:
    """Apply and sweep DC voltage for one bound mux."""

    def __init__(
        self,
        *,
        apply_voltage: Callable[[float, DCVoltageProfile], None],
        apply_voltage_immediately: Callable[[float, DCVoltageProfile], None],
        idle: Callable[[DCVoltageProfile], None],
        get_state: Callable[[], DCVoltageState],
        profile: DCVoltageProfile,
    ) -> None:
        """Initialize bound DC voltage operations."""
        self._apply_voltage = apply_voltage
        self._apply_voltage_immediately = apply_voltage_immediately
        self._idle = idle
        self._get_state = get_state
        self._profile = profile

    @property
    def state(self) -> DCVoltageState:
        """Return the current voltage and output state."""
        return self._get_state()

    def apply_voltage(
        self,
        voltage: float,
        *,
        tolerance: float | None = None,
    ) -> DCVoltageState:
        """
        Ramp an enabled output from its current voltage to a target.

        Parameters
        ----------
        voltage : float
            Target voltage in V.
        tolerance : float or None, optional
            Allowed voltage readback error in V. Uses the configured mux
            profile when omitted.

        Returns
        -------
        DCVoltageState
            Readback state at the target voltage.
        """
        self._apply_voltage(
            float(voltage),
            self._profile_with_tolerance(tolerance),
        )
        return self.state

    def apply_voltage_immediately(
        self,
        voltage: float,
        *,
        tolerance: float | None = None,
    ) -> DCVoltageState:
        """Apply a voltage to an enabled output without ramping."""
        self._apply_voltage_immediately(
            float(voltage),
            self._profile_with_tolerance(tolerance),
        )
        return self.state

    def sweep(
        self,
        *,
        sweep_range: Iterable[float],
        tolerance: float | None = None,
    ) -> Iterator[DCVoltageState]:
        """Ramp to each voltage and yield its readback state."""
        for voltage in sweep_range:
            yield self.apply_voltage(float(voltage), tolerance=tolerance)

    def idle(self, *, tolerance: float | None = None) -> None:
        """Ramp back to the configured idle voltage."""
        self._idle(self._profile_with_tolerance(tolerance))

    def _profile_with_tolerance(
        self,
        tolerance: float | None,
    ) -> DCVoltageProfile:
        """Return the profile with an optional readback tolerance override."""
        if tolerance is None:
            return self._profile
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        return replace(self._profile, readback_tolerance_v=tolerance)

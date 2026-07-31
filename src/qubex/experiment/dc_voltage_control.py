"""Bound DC voltage control for one experiment mux."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator

from qubex.external_devices import DCVoltageProfile

from .models.dc_voltage_state import DCVoltageState


class DCVoltageControl:
    """Apply and sweep DC voltage for one bound mux."""

    def __init__(
        self,
        *,
        set_voltage: Callable[[float, float], None],
        get_state: Callable[[], DCVoltageState],
        turn_on: Callable[[], None],
        turn_off: Callable[[], None],
        profile: DCVoltageProfile,
    ) -> None:
        """Initialize bound DC voltage operations."""
        self._set_voltage = set_voltage
        self._get_state = get_state
        self._turn_on = turn_on
        self._turn_off = turn_off
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
        Enable the output and ramp from its current voltage to a target.

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
        resolved_tolerance = self._resolve_tolerance(tolerance)
        state = self.state
        if state.is_on:
            start = state.voltage
        else:
            start = self._profile.safe_voltage_v
            self._set_voltage(start, resolved_tolerance)
            self._turn_on()
        for setpoint in self._ramp_setpoints(start=start, voltage=float(voltage)):
            self._set_voltage(setpoint, resolved_tolerance)
            time.sleep(self._profile.update_interval_s)
        return self.state

    def apply_voltage_immediately(
        self,
        voltage: float,
        *,
        tolerance: float | None = None,
    ) -> DCVoltageState:
        """Enable the output and apply a voltage without ramping."""
        state = self.state
        self._set_voltage(float(voltage), self._resolve_tolerance(tolerance))
        if not state.is_on:
            self._turn_on()
        return self.state

    def turn_on(self) -> DCVoltageState:
        """Turn on the bound output and return its readback state."""
        self._turn_on()
        return self.state

    def turn_off(self) -> DCVoltageState:
        """Turn off the bound output and return its readback state."""
        self._turn_off()
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

    def shutdown(self, *, tolerance: float | None = None) -> None:
        """Ramp to the configured safe voltage and turn the output off."""
        try:
            if self.state.is_on:
                self.apply_voltage(
                    self._profile.safe_voltage_v,
                    tolerance=tolerance,
                )
        finally:
            self._turn_off()

    def _ramp_setpoints(self, *, start: float, voltage: float) -> list[float]:
        """Return incremental setpoints after the start through the target."""
        if start == voltage:
            return []
        step = self._profile.ramp_rate_v_per_s * self._profile.update_interval_s
        direction = 1.0 if voltage >= start else -1.0
        setpoints: list[float] = []
        current = float(start)
        while abs(voltage - current) > step:
            current += direction * step
            setpoints.append(current)
        if not setpoints or setpoints[-1] != voltage:
            setpoints.append(float(voltage))
        return setpoints

    def _resolve_tolerance(self, tolerance: float | None) -> float:
        """Resolve an optional readback tolerance against the mux profile."""
        if tolerance is None:
            return self._profile.readback_tolerance_v
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        return tolerance

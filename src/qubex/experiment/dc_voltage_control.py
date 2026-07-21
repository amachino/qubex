"""Bound DC voltage control for one experiment mux."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator

from .models.dc_voltage_state import DCVoltageState


class DCVoltageControl:
    """Control and sweep the DC voltage bound to one mux."""

    def __init__(
        self,
        *,
        set_voltage: Callable[[float, float], None],
        get_state: Callable[[], DCVoltageState],
    ) -> None:
        """Initialize the bound DC voltage operations."""
        self._set_voltage = set_voltage
        self._get_state = get_state

    @property
    def state(self) -> DCVoltageState:
        """Return the current voltage and output state."""
        return self._get_state()

    def set(
        self,
        voltage: float,
        *,
        tolerance: float = 1e-3,
    ) -> DCVoltageState:
        """Set a voltage and return its readback state."""
        self._set_voltage(voltage, tolerance)
        return self.state

    def sweep(
        self,
        *,
        sweep_range: Iterable[float],
        delay: float = 0.0,
        tolerance: float = 1e-3,
    ) -> Iterator[DCVoltageState]:
        """
        Sweep voltage values and yield each readback state.

        Parameters
        ----------
        sweep_range : Iterable[float]
            Voltage setpoints in V.
        delay : float
            Settling delay after each setpoint in seconds.
        tolerance : float
            Allowed voltage readback error in V.

        Yields
        ------
        DCVoltageState
            Readback state after applying each setpoint and delay.

        Raises
        ------
        ValueError
            If `delay` is negative.
        """
        if delay < 0:
            raise ValueError("delay must be non-negative.")

        for voltage in sweep_range:
            self._set_voltage(float(voltage), tolerance)
            if delay > 0:
                time.sleep(delay)
            yield self.state

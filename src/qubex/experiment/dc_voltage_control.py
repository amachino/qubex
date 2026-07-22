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
        self._restore_profile: tuple[float, float, float, float] | None = None

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

    def ramp_to(
        self,
        voltage: float,
        *,
        start: float,
        step: float,
        delay: float = 0.0,
        tolerance: float = 1e-3,
        restore_on_exit: bool = True,
    ) -> DCVoltageState:
        """
        Approach a target voltage incrementally from a specified voltage.

        Parameters
        ----------
        voltage : float
            Target voltage in V.
        start : float
            Initial voltage in V. Select this above or below the target to make
            the approach direction reproducible when the device has hysteresis.
        step : float
            Positive maximum voltage increment in V.
        delay : float
            Settling delay after every setpoint in seconds.
        tolerance : float
            Allowed voltage readback error in V.
        restore_on_exit : bool
            Whether to ramp back to `start` with the same step and delay when
            the bound DC control context exits.

        Returns
        -------
        DCVoltageState
            Readback state at the exact target voltage.

        Raises
        ------
        ValueError
            If `step` is not positive or `delay` is negative.
        """
        if step <= 0:
            raise ValueError("step must be positive.")

        setpoints = self._ramp_setpoints(
            start=start,
            voltage=voltage,
            step=step,
        )

        states = list(
            self.sweep(
                sweep_range=setpoints,
                delay=delay,
                tolerance=tolerance,
            )
        )
        self._restore_profile = (
            (float(start), float(step), float(delay), float(tolerance))
            if restore_on_exit
            else None
        )
        return states[-1]

    @staticmethod
    def _ramp_setpoints(
        *,
        start: float,
        voltage: float,
        step: float,
    ) -> list[float]:
        """Return setpoints from the start through the exact target."""
        direction = 1.0 if voltage >= start else -1.0
        setpoints = [float(start)]
        current = float(start)
        while abs(voltage - current) > step:
            current += direction * step
            setpoints.append(current)
        if setpoints[-1] != voltage:
            setpoints.append(float(voltage))
        return setpoints

    def _restore_ramp_start(self) -> None:
        """Ramp back to the recorded start voltage before context shutdown."""
        if self._restore_profile is None:
            return
        start, step, delay, tolerance = self._restore_profile
        current = self.state.voltage
        setpoints = self._ramp_setpoints(
            start=current,
            voltage=start,
            step=step,
        )[1:]
        for _ in self.sweep(
            sweep_range=setpoints,
            delay=delay,
            tolerance=tolerance,
        ):
            pass

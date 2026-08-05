"""Bound DC voltage control for one experiment mux."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from qubex.external_devices import DCVoltageProfile

from .models.dc_voltage_state import DCVoltageState


class DCVoltageControl:
    """Apply and sweep DC voltage for one bound mux."""

    def __init__(
        self,
        *,
        apply_voltage: Callable[[float, DCVoltageProfile], None],
        idle: Callable[[DCVoltageProfile], None],
        get_state: Callable[[], DCVoltageState],
        profile: DCVoltageProfile,
    ) -> None:
        """Initialize bound DC voltage operations."""
        self._apply_voltage = apply_voltage
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
    ) -> DCVoltageState:
        """
        Ramp an enabled output from its current voltage to a target.

        Parameters
        ----------
        voltage : float
            Target voltage in V.

        Returns
        -------
        DCVoltageState
            Readback state at the target voltage.
        """
        self._apply_voltage(
            float(voltage),
            self._profile,
        )
        return self.state

    def sweep(
        self,
        *,
        sweep_range: Iterable[float],
    ) -> Iterator[DCVoltageState]:
        """Ramp to each voltage and yield its readback state."""
        for voltage in sweep_range:
            yield self.apply_voltage(float(voltage))

    def idle(self) -> None:
        """Ramp back to the configured idle voltage."""
        self._idle(self._profile)

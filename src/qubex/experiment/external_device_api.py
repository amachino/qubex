"""Experiment-facing API for external device operations."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from qubex.system.control_parameters import DCVoltageOnExit

from .dc_voltage_control import DCVoltageControl
from .models.dc_voltage_state import DCVoltageState

if TYPE_CHECKING:
    from .experiment_context import ExperimentContext


class ExternalDeviceAPI:
    """Operate external devices (DC voltage sources) from an experiment."""

    def __init__(self, *, context: ExperimentContext) -> None:
        """Bind the API to one experiment context."""
        self._ctx = context

    @contextmanager
    def dc_voltage_control(
        self,
        *,
        mux: int | str | None = None,
        on_exit: DCVoltageOnExit = "idle",
    ) -> Iterator[DCVoltageControl]:
        """
        Yield DC voltage operations bound to one mux.

        Parameters
        ----------
        mux : int or str, optional
            Mux index or label. Required when multiple muxes are active.
        on_exit : {"idle", "hold"}
            Exit behavior. `idle` (default) ramps back to the idle voltage;
            `hold` leaves the bias applied.

        Yields
        ------
        DCVoltageControl
            Operations bound to the resolved mux.
        """
        with self._ctx.dc_voltage_control(mux=mux, on_exit=on_exit) as control:
            yield control

    def get_dc_voltage_state(
        self,
        *,
        mux: int | str | None = None,
    ) -> DCVoltageState:
        """Return DC voltage and output-state readback for one mux."""
        return self._ctx.get_dc_voltage_state(mux=mux)

    def get_dc_voltage_states(self) -> dict[int, DCVoltageState]:
        """
        Return DC readback states for every wired mux.

        Returns
        -------
        dict[int, DCVoltageState]
            States keyed by mux index, read on one device connection.
        """
        return self._ctx.get_dc_voltage_states()

    def reset_dc_voltages(
        self,
        muxes: int | str | Collection[int | str] | None = None,
        confirm: bool = True,
    ) -> dict[int, DCVoltageState]:
        """
        Bring the selected muxes to their reset voltages, outputs on.

        Off outputs come up at `reset_voltage_v` (default 0 V) before
        being switched on; on outputs are ramped to it. Afterwards the
        selection is in the known reset state, regardless of stale stored
        setpoints.

        Parameters
        ----------
        muxes : int, str, or collection of them, optional
            Target mux indices or labels. All wired muxes when omitted.
        confirm : bool, optional
            Whether to prompt before writing to the hardware.

        Returns
        -------
        dict[int, DCVoltageState]
            Readback states keyed by mux index after resetting.
        """
        return self._ctx.reset_dc_voltages(muxes=muxes, confirm=confirm)

    def bias_dc_voltages(
        self,
        muxes: int | str | Collection[int | str] | None = None,
        confirm: bool = True,
    ) -> dict[int, DCVoltageState]:
        """
        Ramp the selected calibrated muxes to their bias voltages.

        When `muxes` is omitted, every wired mux with a calibrated
        `bias_voltage` in `jpa_params.yaml` is biased and the rest are
        skipped; an explicitly selected mux without one raises. Ramp back
        with `idle_dc_voltages()`.

        Parameters
        ----------
        muxes : int, str, or collection of them, optional
            Target mux indices or labels. All wired muxes when omitted.
        confirm : bool, optional
            Whether to prompt before writing to the hardware.

        Returns
        -------
        dict[int, DCVoltageState]
            Readback states keyed by mux index after biasing.
        """
        return self._ctx.bias_dc_voltages(muxes=muxes, confirm=confirm)

    def idle_dc_voltages(
        self,
        muxes: int | str | Collection[int | str] | None = None,
        confirm: bool = True,
    ) -> dict[int, DCVoltageState]:
        """
        Ramp the selected muxes back to their idle voltages.

        Parameters
        ----------
        muxes : int, str, or collection of them, optional
            Target mux indices or labels. All wired muxes when omitted.
        confirm : bool, optional
            Whether to prompt before writing to the hardware.

        Returns
        -------
        dict[int, DCVoltageState]
            Readback states keyed by mux index after idling.
        """
        return self._ctx.idle_dc_voltages(muxes=muxes, confirm=confirm)

"""Amplification services for measurement workflows."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator
from contextlib import contextmanager

from qubex.external_devices import DCVoltageExitMode
from qubex.measurement.measurement_context import MeasurementContext
from qubex.system import ControlParameters, ExperimentSystem
from qubex.system.control_parameters import DCVoltageOnExit

logger = logging.getLogger(__name__)


class MeasurementAmplificationService:
    """Manage amplification and DC operations for measurement APIs."""

    def __init__(
        self,
        *,
        context: MeasurementContext,
    ) -> None:
        self._context = context

    @property
    def context(self) -> MeasurementContext:
        """Return measurement context accessor."""
        return self._context

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Return the active experiment-system model."""
        return self.context.experiment_system

    @property
    def control_params(self) -> ControlParameters:
        """Return control parameters from the active experiment system."""
        return self.experiment_system.control_params

    @contextmanager
    def apply_dc_voltages(
        self,
        targets: str | Collection[str],
        *,
        on_exit: DCVoltageOnExit | None = None,
    ) -> Iterator[None]:
        """
        Apply amplification-point DC voltages to the specified targets.

        Parameters
        ----------
        targets : str | Collection[str]
            Target label or target labels.
        on_exit : {"idle", "hold"} or None, optional
            Exit behavior. `idle` (default) ramps each bias back to its idle
            voltage; `hold` leaves the applied biases on.

        Notes
        -----
        Muxes without a calibrated `dc_voltage` in `jpa_params.yaml` are
        skipped: no DC voltage source is touched for them.
        """
        if isinstance(targets, str):
            targets = [targets]
        qubits = [
            self.experiment_system.resolve_qubit_label(target) for target in targets
        ]
        muxes = {
            self.experiment_system.get_mux_by_qubit(qubit).index for qubit in qubits
        }
        uncalibrated = {
            mux for mux in muxes if not self.control_params.has_dc_voltage(mux)
        }
        if uncalibrated:
            logger.info(
                "Skipping DC voltage application for muxes without a "
                "calibrated `dc_voltage` in `jpa_params.yaml`: %s",
                sorted(uncalibrated),
            )
        muxes -= uncalibrated
        if not muxes:
            yield
            return
        profiles = {
            mux: self.context.system_manager.resolve_dc_voltage_profile(mux)
            for mux in muxes
        }
        requests = {
            profile.channel: (self.control_params.get_dc_voltage(mux), profile)
            for mux, profile in profiles.items()
        }
        exit_mode = DCVoltageExitMode(on_exit) if on_exit else DCVoltageExitMode.IDLE
        exit_modes = {profile.channel: exit_mode for profile in profiles.values()}
        with self.context.system_manager.dc_voltage_controller.apply_voltages(
            requests,
            exit_modes=exit_modes,
        ):
            yield

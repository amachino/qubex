"""Amplification services for measurement workflows."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator
from contextlib import contextmanager

from qubex.measurement.measurement_context import MeasurementContext
from qubex.system import ControlParameters, ExperimentSystem

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
    ) -> Iterator[None]:
        """
        Apply amplification-point DC voltages to the specified targets.

        Parameters
        ----------
        targets : str | Collection[str]
            Target label or target labels.

        Notes
        -----
        Muxes without a calibrated `bias_voltage` in `jpa_params.yaml` are
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
            mux for mux in muxes if not self.control_params.has_bias_voltage(mux)
        }
        if uncalibrated:
            logger.info(
                "Skipping DC voltage application for muxes without a "
                "calibrated `bias_voltage` in `jpa_params.yaml`: %s",
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
            profile.channel: (self.control_params.get_bias_voltage(mux), profile)
            for mux, profile in profiles.items()
        }
        with self.context.system_manager.dc_voltage_controller.apply_voltages(
            requests,
        ):
            yield

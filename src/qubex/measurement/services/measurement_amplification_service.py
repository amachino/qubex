"""Amplification services for measurement workflows."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from contextlib import contextmanager

from qubex.backend.dc_voltage_controller import dc_voltage
from qubex.measurement.measurement_context import MeasurementContext
from qubex.system import ControlParameters, ExperimentSystem


class MeasurementAmplificationService:
    """Manage temporary amplification/DC operations for measurement APIs."""

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
    def apply_dc_voltages(self, targets: str | Collection[str]) -> Iterator[None]:
        """
        Temporarily apply DC voltages to the specified targets.

        Parameters
        ----------
        targets : str | Collection[str]
            Target label or target labels.
        """
        if isinstance(targets, str):
            targets = [targets]
        qubits = [
            self.experiment_system.resolve_qubit_label(target) for target in targets
        ]
        muxes = {
            self.experiment_system.get_mux_by_qubit(qubit).index for qubit in qubits
        }
        voltages = {mux + 1: self.control_params.get_dc_voltage(mux) for mux in muxes}
        with dc_voltage(voltages):
            yield

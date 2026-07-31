"""Amplification services for measurement workflows."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from contextlib import contextmanager

from qubex.external_devices import DCVoltageExitMode, DCVoltageExitPolicy
from qubex.measurement.measurement_context import MeasurementContext
from qubex.system import ControlParameters, ExperimentSystem
from qubex.system.control_parameters import DCVoltageOnExit


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
        on_exit : {"off", "low_noise", "restore", "hold"} or None, optional
            Exit behavior override. Uses each mux's JPA parameter when omitted.
        """
        if isinstance(targets, str):
            targets = [targets]
        qubits = [
            self.experiment_system.resolve_qubit_label(target) for target in targets
        ]
        muxes = {
            self.experiment_system.get_mux_by_qubit(qubit).index for qubit in qubits
        }
        profiles = {
            mux: self.context.system_manager.resolve_dc_voltage_profile(mux)
            for mux in muxes
        }
        requests = {
            profile.channel: (self.control_params.get_dc_voltage(mux), profile)
            for mux, profile in profiles.items()
        }
        exit_policies = {
            profile.channel: self._resolve_exit_policy(mux=mux, on_exit=on_exit)
            for mux, profile in profiles.items()
        }
        with self.context.system_manager.dc_voltage_controller.apply_voltages(
            requests,
            exit_policies=exit_policies,
        ):
            yield

    def _resolve_exit_policy(
        self,
        *,
        mux: int,
        on_exit: DCVoltageOnExit | None,
    ) -> DCVoltageExitPolicy:
        """Resolve one measurement exit mode into a generic controller policy."""
        mode = on_exit or self.control_params.get_dc_voltage_exit_mode(mux)
        if mode == "off":
            return DCVoltageExitPolicy(mode=DCVoltageExitMode.SHUTDOWN)
        if mode == "hold":
            return DCVoltageExitPolicy(mode=DCVoltageExitMode.HOLD)
        if mode == "restore":
            return DCVoltageExitPolicy(mode=DCVoltageExitMode.RESTORE)
        if mode == "low_noise":
            return DCVoltageExitPolicy(
                mode=DCVoltageExitMode.TARGET,
                target_voltage_v=self.control_params.get_low_noise_dc_voltage(mux),
            )
        raise ValueError(f"Unsupported DC voltage exit mode: {mode!r}.")

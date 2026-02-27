"""Execution services for measurement workflows."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import numpy as np
from qxpulse import PulseSchedule, RampType

from qubex.backend import (
    BackendController,
)
from qubex.backend.quel1 import (
    ExecutionMode,
    Quel1BackendController,
)
from qubex.backend.quel3 import Quel3BackendController
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge
from qubex.measurement.classifiers.state_classifier import StateClassifier
from qubex.measurement.measurement_config_factory import MeasurementConfigFactory
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_context import MeasurementContext
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.measurement_schedule_builder import (
    CapturePlacement,
    MeasurementScheduleBuilder,
)
from qubex.measurement.measurement_schedule_runner import MeasurementScheduleRunner
from qubex.measurement.models.measure_result import (
    MeasureResult,
    MultipleMeasureResult,
)
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.quel1_measurement_options import Quel1MeasurementOptions
from qubex.measurement.models.sweep_measurement_result import (
    NDSweepMeasurementResult,
    SweepAxes,
    SweepKey,
    SweepMeasurementResult,
    SweepPoint,
    SweepValue,
)
from qubex.measurement.services.measurement_session_service import (
    MeasurementSessionService,
)
from qubex.system import (
    ConfigLoader,
    ControlParams,
    ExperimentSystem,
    Mux,
    PortType,
    SystemManager,
    Target,
)
from qubex.typing import IQArray, TargetMap

logger = logging.getLogger(__name__)

T = TypeVar("T")
OptionT = TypeVar("OptionT")
RFSwitchState = Literal["pass", "block", "open", "loop"]


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="measurement")
    return bridge.run(factory, timeout=timeout)


class MeasurementExecutionService:
    """Own measurement execution orchestration in the measurement layer."""

    def __init__(
        self,
        *,
        context: MeasurementContext,
        session_service: MeasurementSessionService,
        classifiers: TargetMap[StateClassifier],
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> None:
        self._context = context
        self._session_service = session_service
        self._classifiers = classifiers
        self._execution_mode: ExecutionMode | None = execution_mode
        self._clock_health_checks: bool | None = clock_health_checks

    @property
    def context(self) -> MeasurementContext:
        """Return the measurement context."""
        return self._context

    @property
    def session_service(self) -> MeasurementSessionService:
        """Return the session lifecycle service."""
        return self._session_service

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        """Return classifier mapping used for result conversion."""
        return self._classifiers

    @property
    def system_manager(self) -> SystemManager:
        """Return the shared system manager."""
        return self.context.system_manager

    @property
    def config_loader(self) -> ConfigLoader:
        """Return the configuration loader."""
        return self.context.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Return the active experiment system."""
        return self.context.experiment_system

    @property
    def backend_controller(self) -> BackendController:
        """Return the active backend controller."""
        return self.session_service.backend_controller

    @property
    def mux_dict(self) -> dict[str, Mux]:
        """Return MUX objects indexed by qubit label."""
        return self.context.mux_dict

    @property
    def pulse_factory(self) -> MeasurementPulseFactory:
        """Return a pulse factory bound to current system state."""
        target_registry = getattr(self.experiment_system, "target_registry", None)
        return MeasurementPulseFactory(
            control_params=self.control_params,
            mux_dict=self.mux_dict,
            target_registry=target_registry,
        )

    @property
    def schedule_builder(self) -> MeasurementScheduleBuilder:
        """Return a schedule builder bound to current system state."""
        target_registry = getattr(self.experiment_system, "target_registry", None)
        return MeasurementScheduleBuilder(
            control_params=self.control_params,
            pulse_factory=self.pulse_factory,
            targets=self.targets,
            mux_dict=self.mux_dict,
            target_registry=target_registry,
            constraint_profile=self.constraint_profile,
        )

    @property
    def measurement_config_factory(self) -> MeasurementConfigFactory:
        """Return a measurement-config factory."""
        return MeasurementConfigFactory(
            experiment_system=self.experiment_system,
        )

    @property
    def sampling_period(self) -> float:
        """Return sampling period in ns."""
        return self.constraint_profile.sampling_period_ns

    @property
    def constraint_profile(self) -> MeasurementConstraintProfile:
        """Return backend timing and alignment constraints."""
        sampling_period = self.backend_controller.sampling_period
        if isinstance(self.backend_controller, Quel3BackendController):
            return MeasurementConstraintProfile.quel3(sampling_period)
        if isinstance(self.backend_controller, Quel1BackendController):
            return MeasurementConstraintProfile.quel1(sampling_period)
        raise TypeError(
            "Unsupported backend controller for constraint profile selection."
        )

    @property
    def measurement_schedule_runner(self) -> MeasurementScheduleRunner:
        """Return the schedule-execution runner."""
        return MeasurementScheduleRunner(
            backend_controller=self.backend_controller,
            experiment_system=self.experiment_system,
            execution_mode=self._execution_mode,
            clock_health_checks=self._clock_health_checks,
        )

    @property
    def control_params(self) -> ControlParams:
        """Return active control parameters."""
        return self.experiment_system.control_params

    @property
    def chip_id(self) -> str:
        """Return the active chip identifier."""
        return self.experiment_system.chip.id

    @property
    def targets(self) -> dict[str, Target]:
        """Return available targets indexed by label."""
        return {target.label: target for target in self.experiment_system.targets}

    @property
    def nco_frequencies(self) -> dict[str, float]:
        """Return NCO frequencies indexed by target label."""
        return {
            target.label: self.experiment_system.get_nco_frequency(target.label)
            for target in self.experiment_system.targets
        }

    @property
    def awg_frequencies(self) -> dict[str, float]:
        """Return AWG frequencies indexed by target label."""
        return {
            target.label: self.experiment_system.get_awg_frequency(target.label)
            for target in self.experiment_system.targets
        }

    @staticmethod
    def _resolve_device_config(
        backend_controller: BackendController,
    ) -> dict:
        """Resolve backend device configuration mapping."""
        box_config = getattr(backend_controller, "box_config", None)
        if isinstance(box_config, dict):
            return box_config
        return {}

    @staticmethod
    def _warn_deprecated_alias(
        *,
        old_name: str,
        new_name: str | None = None,
        message: str | None = None,
    ) -> None:
        """Emit a deprecation warning for a legacy option."""
        if message is None:
            if new_name is None:
                message = f"`{old_name}` is deprecated."
            else:
                message = f"`{old_name}` is deprecated; use `{new_name}`."
        warnings.warn(
            message,
            DeprecationWarning,
            stacklevel=3,
        )

    @classmethod
    def _resolve_deprecated_alias(
        cls,
        *,
        new_value: OptionT | None,
        old_value: OptionT | None,
        old_name: str,
        new_name: str,
    ) -> OptionT | None:
        """Resolve an old/new alias pair and validate conflicts."""
        if old_value is None:
            return new_value
        cls._warn_deprecated_alias(
            old_name=old_name,
            new_name=new_name,
        )
        if new_value is not None and new_value != old_value:
            raise ValueError(
                f"`{old_name}` conflicts with `{new_name}`. Provide only `{new_name}`."
            )
        return old_value if new_value is None else new_value

    def get_awg_frequency(self, target: str) -> float:
        """
        Get the AWG frequency for the target.

        Parameters
        ----------
        target : str
            Target label.


        Returns
        -------
        float
            AWG frequency in Hz.

        """
        return self.experiment_system.get_awg_frequency(target)

    def get_diff_frequency(self, target: str) -> float:
        """
        Get the difference frequency for the target.

        Parameters
        ----------
        target : str
            Target label.


        Returns
        -------
        float
            Difference frequency in Hz.

        """
        return self.experiment_system.get_diff_frequency(target)

    def _resolve_loopback_capture_targets(
        self,
        *,
        schedule: PulseSchedule,
    ) -> list[str]:
        """Resolve read-in and monitor capture targets for loopback acquisition."""
        active_qubits: list[str] = []
        active_boxes: list[str] = []
        for label in schedule.labels:
            target = self.targets.get(label)
            if target is not None:
                active_boxes.append(str(target.channel.port.box_id))
            try:
                qubit_label = self.experiment_system.resolve_qubit_label(label)
            except ValueError:
                continue
            active_qubits.append(str(qubit_label))

        def _resolve_read_in_capture_target(target: Any) -> str:
            """Prefer READ_IN port ID so loopback labels match monitor port labels."""
            channel = getattr(target, "channel", None)
            port = None if channel is None else getattr(channel, "port", None)
            port_id = None if port is None else getattr(port, "id", None)
            if isinstance(port_id, str) and port_id:
                return port_id
            return str(target.label)

        read_in_target_by_qubit: dict[str, str] = {}
        for target in self.experiment_system.read_in_targets:
            try:
                qubit_label = self.experiment_system.resolve_qubit_label(target.label)
            except ValueError:
                continue
            if qubit_label not in read_in_target_by_qubit:
                read_in_target_by_qubit[qubit_label] = _resolve_read_in_capture_target(
                    target
                )

        read_capture_targets = [
            read_in_target_by_qubit[qubit]
            for qubit in dict.fromkeys(active_qubits)
            if qubit in read_in_target_by_qubit
        ]

        active_box_set = set(active_boxes)
        if not read_capture_targets and active_box_set:
            read_capture_targets.extend(
                _resolve_read_in_capture_target(target)
                for target in self.experiment_system.read_in_targets
                if target.channel.port.box_id in active_box_set
            )
        if not active_box_set:
            active_box_set = {
                box.id for box in self.experiment_system.control_system.boxes
            }

        monitor_capture_targets: list[str] = []
        for box in self.experiment_system.control_system.boxes:
            if box.id not in active_box_set:
                continue
            monitor_capture_targets.extend(
                port.id for port in box.ports if port.type == PortType.MNTR_IN
            )

        return list(dict.fromkeys([*read_capture_targets, *monitor_capture_targets]))

    def _filter_loopback_capture_targets(
        self,
        *,
        capture_targets: Sequence[str],
        port_type: PortType,
    ) -> list[str]:
        """Filter loopback capture targets by resolved port type."""
        filtered: list[str] = []
        for target in capture_targets:
            port = self._resolve_loopback_capture_port(target_or_port_id=target)
            if port is None:
                continue
            if port.type == port_type:
                filtered.append(target)
        return list(dict.fromkeys(filtered))

    @staticmethod
    def _is_e7awg_capture_data_error(exc: Exception) -> bool:
        """Return whether an exception indicates broken captured data."""
        return type(exc).__name__ == "E7awgCaptureDataError"

    @staticmethod
    def _is_rfswitch_unsupported_error(exc: Exception) -> bool:
        """Return whether an exception indicates RF-switch is unsupported."""
        if type(exc).__name__ == "NoRfSwitchError":
            return True
        if isinstance(exc, ValueError):
            message = str(exc).lower()
            return "invalid port of" in message or "no switch available" in message
        return False

    def _resolve_loopback_box_ids(
        self,
        *,
        schedule: PulseSchedule,
        capture_targets: Sequence[str],
    ) -> list[str]:
        """Resolve box IDs involved in loopback capture execution."""
        box_ids: list[str] = []

        for label in schedule.labels:
            target = self.targets.get(label)
            if target is None:
                continue
            box_ids.append(str(target.channel.port.box_id))

        for target in capture_targets:
            port = self._resolve_loopback_capture_port(target_or_port_id=target)
            if port is None:
                continue
            box_ids.append(str(port.box_id))

        if not box_ids:
            box_ids.extend(self.context.box_ids)

        return list(dict.fromkeys(box_ids))

    def _initialize_loopback_capture_units(
        self,
        *,
        box_ids: Sequence[str],
    ) -> None:
        """Initialize AWG/CAP units for loopback capture when supported."""
        initialize_awg_and_capunits = getattr(
            self.backend_controller,
            "initialize_awg_and_capunits",
            None,
        )
        if not callable(initialize_awg_and_capunits):
            return

        resolved_box_ids = list(dict.fromkeys(box_ids))
        if not resolved_box_ids:
            return
        initialize_awg_and_capunits(resolved_box_ids)

    def _resolve_loopback_capture_port(
        self,
        *,
        target_or_port_id: str,
    ) -> Any | None:
        """Resolve one loopback capture port from target label or port ID."""
        control_system = self.experiment_system.control_system
        try:
            return control_system.get_port_by_id(target_or_port_id)
        except KeyError:
            pass

        get_cap_target = getattr(self.experiment_system, "get_cap_target", None)
        if not callable(get_cap_target):
            return None
        try:
            cap_target = get_cap_target(target_or_port_id)
        except KeyError:
            return None
        channel = getattr(cap_target, "channel", None)
        return None if channel is None else getattr(channel, "port", None)

    def _resolve_loopback_rfswitch_updates(
        self,
        *,
        capture_targets: Sequence[str],
    ) -> dict[str, RFSwitchState]:
        """Build loopback RF-switch overrides keyed by port ID."""
        control_system = self.experiment_system.control_system
        updates: dict[str, RFSwitchState] = {}

        for capture_target in capture_targets:
            port = self._resolve_loopback_capture_port(target_or_port_id=capture_target)
            if port is None:
                continue

            if port.type == PortType.READ_IN:
                updates[port.id] = "loop"
                for box_port in control_system.get_box(port.box_id).ports:
                    if box_port.type == PortType.READ_OUT:
                        updates[box_port.id] = "block"
            elif port.type == PortType.MNTR_IN:
                updates[port.id] = "loop"

        return updates

    def _set_port_rfswitch(
        self,
        *,
        port: Any,
        rfswitch: RFSwitchState,
    ) -> None:
        """Set one port RF switch on hardware and in experiment model."""
        config_port = getattr(self.backend_controller, "config_port", None)
        if not callable(config_port):
            raise NotImplementedError(
                "Active backend does not support RF-switch configuration."
            )

        try:
            config_port(
                box_name=port.box_id,
                port=port.number,
                rfswitch=rfswitch,
            )
        except Exception as exc:
            if self._is_rfswitch_unsupported_error(exc):
                logger.warning(
                    "Skip RF-switch update for %s on %s because the port does not support RF switch configuration.",
                    port.id,
                    port.box_id,
                )
                return
            raise

        self.experiment_system.control_system.set_port_params(
            box_id=port.box_id,
            port_number=port.number,
            rfswitch=rfswitch,
        )

    @contextmanager
    def _temporary_loopback_rfswitches(
        self,
        *,
        capture_targets: Sequence[str],
    ) -> Iterator[None]:
        """Temporarily configure RF switches for loopback capture and restore them."""
        config_port = getattr(self.backend_controller, "config_port", None)
        if not callable(config_port):
            yield
            return

        updates = self._resolve_loopback_rfswitch_updates(
            capture_targets=capture_targets
        )
        if not updates:
            yield
            return

        control_system = self.experiment_system.control_system
        original_rfswitches: dict[str, RFSwitchState] = {}
        for port_id in sorted(updates):
            try:
                port = control_system.get_port_by_id(port_id)
            except KeyError:
                continue
            original_rfswitches[port_id] = cast(RFSwitchState, str(port.rfswitch))

        try:
            for port_id in sorted(updates):
                try:
                    port = control_system.get_port_by_id(port_id)
                except KeyError:
                    continue
                desired_rfswitch = updates[port_id]
                if str(port.rfswitch) == desired_rfswitch:
                    continue
                self._set_port_rfswitch(
                    port=port,
                    rfswitch=desired_rfswitch,
                )
            yield
        finally:
            for port_id in sorted(original_rfswitches):
                try:
                    port = control_system.get_port_by_id(port_id)
                except KeyError:
                    continue
                restore_rfswitch = original_rfswitches[port_id]
                if str(port.rfswitch) == restore_rfswitch:
                    continue
                self._set_port_rfswitch(
                    port=port,
                    rfswitch=restore_rfswitch,
                )

    async def run_measurement(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        """
        Run measurement with the given schedule and configuration.

        Parameters
        ----------
        schedule : MeasurementSchedule
            Measurement schedule.

        config : MeasurementConfig
            Measurement configuration.


        Returns
        -------
        MeasurementResult
            Measurement result.

        """
        return await self.measurement_schedule_runner.execute(
            schedule=schedule,
            config=config,
            quel1_options=quel1_options,
        )

    async def run_sweep_measurement(
        self,
        schedule: Callable[[SweepValue], MeasurementSchedule],
        *,
        sweep_values: Sequence[SweepValue],
        config: MeasurementConfig | None = None,
    ) -> SweepMeasurementResult:
        """
        Run sweep measurement pointwise.

        Parameters
        ----------
        schedule : Callable[[SweepValue], MeasurementSchedule]
            Callback that builds one measurement schedule per sweep value.
        sweep_values : Sequence[SweepValue]
            Ordered sweep values to execute.
        config : MeasurementConfig | None, optional
            Shared measurement configuration for all points.

        Returns
        -------
        SweepMeasurementResult
            Sweep result list in the same order as input values.
        """
        resolved_config = self.create_measurement_config() if config is None else config
        normalized_values = [*sweep_values]
        results: list[MeasurementResult] = []
        for sweep_value in normalized_values:
            measurement_schedule = schedule(sweep_value)
            result = await self.run_measurement(
                schedule=measurement_schedule,
                config=resolved_config,
            )
            results.append(result)
        return SweepMeasurementResult(
            sweep_values=normalized_values,
            config=resolved_config,
            results=results,
        )

    async def run_ndsweep_measurement(
        self,
        schedule: Callable[[SweepPoint], MeasurementSchedule],
        *,
        sweep_points: dict[SweepKey, Sequence[SweepValue]],
        sweep_axes: SweepAxes | None = None,
        config: MeasurementConfig | None = None,
    ) -> NDSweepMeasurementResult:
        """
        Run N-dimensional Cartesian-product sweep measurement pointwise.

        Parameters
        ----------
        schedule : Callable[[SweepPoint], MeasurementSchedule]
            Callback that builds one measurement schedule per resolved sweep point.
        sweep_points : dict[SweepKey, Sequence[SweepValue]]
            Axis-value table (`axis key -> ordered values`).
        sweep_axes : SweepAxes | None, optional
            Axis order for Cartesian product. If omitted, insertion order of
            `sweep_points` is used.
        config : MeasurementConfig | None, optional
            Shared measurement configuration for all points.

        Returns
        -------
        NDSweepMeasurementResult
            Cartesian sweep result with C-order flattening (last axis varies fastest).

        Raises
        ------
        ValueError
            If `sweep_axes` does not match `sweep_points` keys exactly.
        """
        resolved_config = self.create_measurement_config() if config is None else config
        resolved_axes = (
            tuple(sweep_points.keys()) if sweep_axes is None else tuple(sweep_axes)
        )
        if len(set(resolved_axes)) != len(resolved_axes):
            raise ValueError("sweep_axes must not contain duplicate keys.")
        if set(resolved_axes) != set(sweep_points.keys()):
            raise ValueError(
                "sweep_axes must contain each sweep_points key exactly once."
            )

        normalized_axes_points = {axis: [*sweep_points[axis]] for axis in resolved_axes}
        shape = tuple(len(normalized_axes_points[axis]) for axis in resolved_axes)

        results: list[MeasurementResult] = []
        for ndindex in np.ndindex(shape):
            point: SweepPoint = {
                axis: normalized_axes_points[axis][axis_index]
                for axis, axis_index in zip(resolved_axes, ndindex, strict=True)
            }
            measurement_schedule = schedule(point)
            result = await self.run_measurement(
                schedule=measurement_schedule,
                config=resolved_config,
            )
            results.append(result)

        return NDSweepMeasurementResult(
            sweep_points=normalized_axes_points,
            sweep_axes=resolved_axes,
            shape=shape,
            config=resolved_config,
            results=results,
        )

    def measure_noise(
        self,
        targets: Collection[str],
        *,
        duration: float,
    ) -> MeasurementResult:
        """
        Measure readout noise.

        Parameters
        ----------
        targets : Collection[str]
            Target labels to measure.
        duration : float
            Readout duration in ns.

        Returns
        -------
        MeasurementResult
            Measurement results.

        """
        pulse_schedule = PulseSchedule.from_waveforms(
            {target: np.zeros(0) for target in targets}
        )
        measurement_config = self.create_measurement_config(
            n_shots=1,
            shot_averaging=True,
            time_integration=False,
            state_classification=False,
        )
        measurement_schedule = self.build_measurement_schedule(
            pulse_schedule=pulse_schedule,
            readout_duration=duration,
            readout_amplitudes=dict.fromkeys(targets, 0),
            readout_amplification=False,
            final_measurement=True,
        )
        return _run_async(
            lambda: self.run_measurement(
                schedule=measurement_schedule,
                config=measurement_config,
            )
        )

    def measure(
        self,
        waveforms: Mapping[str, IQArray],
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        frequencies: dict[str, float] | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramp_time: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_amplification: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
        classification_line_param0: tuple[float, float, float] | None = None,
        classification_line_param1: tuple[float, float, float] | None = None,
        plot: bool = False,
        **deprecated_options: Any,
    ) -> MeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        waveforms : Mapping[str, IQArray]
            Control waveforms for each target.

        n_shots : int | None, optional
            Number of shots.
        shot_interval : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether to average shots on hardware.
        frequencies : dict[str, float] | None, optional
            Channel-frequency overrides keyed by schedule label.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each qubit.

        readout_duration : float, optional
            Readout duration in ns.

        readout_pre_margin : float, optional
            Readout pre-margin in ns.

        readout_post_margin : float, optional
            Readout post-margin in ns.

        readout_ramp_time : float, optional
            Readout ramp time in ns.

        readout_drag_coeff : float, optional
            Readout drag coefficient.

        readout_ramp_type : RampType, optional
            Readout ramp type.

        readout_amplification : bool | None, optional
            Whether to apply readout amplification pulses.
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.
        classification_line_param0 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 0.
        classification_line_param1 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 1.

        Returns
        -------
        MeasureResult
            Measurement results.

        """
        result = self.execute(
            schedule=waveforms,
            n_shots=n_shots,
            shot_interval=shot_interval,
            shot_averaging=shot_averaging,
            frequencies=frequencies,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramp_time=readout_ramp_time,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            readout_amplification=readout_amplification,
            final_measurement=True,
            time_integration=time_integration,
            state_classification=state_classification,
            classification_line_param0=classification_line_param0,
            classification_line_param1=classification_line_param1,
            plot=plot,
            **deprecated_options,
        )
        data = {target: measures[0] for target, measures in result.data.items()}
        return MeasureResult(
            mode=result.mode,
            data=data,
            config=result.config,
        )

    def execute(
        self,
        schedule: PulseSchedule | TargetMap[IQArray],
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        frequencies: dict[str, float] | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramp_time: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_amplification: bool | None = None,
        final_measurement: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
        classification_line_param0: tuple[float, float, float] | None = None,
        classification_line_param1: tuple[float, float, float] | None = None,
        plot: bool = False,
        save_result: bool = True,
        **deprecated_options: Any,
    ) -> MultipleMeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        schedule : PulseSchedule | TargetMap[IQArray]
            Pulse schedule or control waveforms.

        n_shots : int | None, optional
            Number of shots.
        shot_interval : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether to average shots on hardware.
        frequencies : dict[str, float] | None, optional
            Channel-frequency overrides keyed by schedule label.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each qubit.

        readout_duration : float, optional
            Readout duration in ns.

        readout_pre_margin : float, optional
            Readout pre-margin in ns.

        readout_post_margin : float, optional
            Readout post-margin in ns.

        readout_ramp_time : float, optional
            Readout ramp time in ns.

        readout_drag_coeff : float, optional
            Readout drag coefficient.

        readout_ramp_type : RampType, optional
            Readout ramp type.

        readout_amplification : bool | None, optional
            Whether to apply readout amplification pulses.
        final_measurement : bool | None, optional
            Whether to append a final readout measurement.
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.
        classification_line_param0 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 0.
        classification_line_param1 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 1.
        plot : bool, optional
            Whether to plot the results.

        Returns
        -------
        MultipleMeasureResult
            Measurement results.

        """
        legacy_options: dict[str, Any] = dict(deprecated_options)
        legacy_keys = {
            "mode",
            "shots",
            "interval",
            "readout_ramptime",
            "add_last_measurement",
            "add_pump_pulses",
            "enable_dsp_demodulation",
            "enable_dsp_sum",
            "enable_dsp_classification",
            "line_param0",
            "line_param1",
        }
        unknown_keys = sorted(set(legacy_options) - legacy_keys)
        if unknown_keys:
            joined = ", ".join(f"`{key}`" for key in unknown_keys)
            raise TypeError(f"Unexpected keyword argument(s): {joined}")

        legacy_mode = legacy_options.pop("mode", None)
        if legacy_mode is not None:
            self._warn_deprecated_alias(
                old_name="mode",
                new_name="shot_averaging",
            )
            legacy_shot_averaging = legacy_mode == "avg"
            if shot_averaging is not None and shot_averaging != legacy_shot_averaging:
                raise ValueError(
                    "`mode` conflicts with `shot_averaging`. "
                    "Provide only `shot_averaging`."
                )
            if shot_averaging is None:
                shot_averaging = legacy_shot_averaging
        if shot_averaging is None:
            shot_averaging = True

        n_shots = self._resolve_deprecated_alias(
            new_value=n_shots,
            old_value=legacy_options.pop("shots", None),
            old_name="shots",
            new_name="n_shots",
        )
        shot_interval = self._resolve_deprecated_alias(
            new_value=shot_interval,
            old_value=legacy_options.pop("interval", None),
            old_name="interval",
            new_name="shot_interval",
        )
        readout_ramp_time = self._resolve_deprecated_alias(
            new_value=readout_ramp_time,
            old_value=legacy_options.pop("readout_ramptime", None),
            old_name="readout_ramptime",
            new_name="readout_ramp_time",
        )
        final_measurement = self._resolve_deprecated_alias(
            new_value=final_measurement,
            old_value=legacy_options.pop("add_last_measurement", None),
            old_name="add_last_measurement",
            new_name="final_measurement",
        )
        if final_measurement is None:
            final_measurement = False

        readout_amplification = self._resolve_deprecated_alias(
            new_value=readout_amplification,
            old_value=legacy_options.pop("add_pump_pulses", None),
            old_name="add_pump_pulses",
            new_name="readout_amplification",
        )
        if readout_amplification is None:
            readout_amplification = False

        legacy_enable_dsp_demodulation = legacy_options.pop(
            "enable_dsp_demodulation", None
        )
        if legacy_enable_dsp_demodulation is not None:
            self._warn_deprecated_alias(
                old_name="enable_dsp_demodulation",
                message=(
                    "`enable_dsp_demodulation` is deprecated and ignored "
                    "because demodulation is always enabled."
                ),
            )
            if legacy_enable_dsp_demodulation is False:
                raise ValueError(
                    "enable_dsp_demodulation is deprecated and always enabled; "
                    "remove this argument or pass None."
                )

        time_integration = self._resolve_deprecated_alias(
            new_value=time_integration,
            old_value=legacy_options.pop("enable_dsp_sum", None),
            old_name="enable_dsp_sum",
            new_name="time_integration",
        )
        if time_integration is None:
            time_integration = True

        state_classification = self._resolve_deprecated_alias(
            new_value=state_classification,
            old_value=legacy_options.pop("enable_dsp_classification", None),
            old_name="enable_dsp_classification",
            new_name="state_classification",
        )
        if state_classification is None:
            state_classification = False

        classification_line_param0 = self._resolve_deprecated_alias(
            new_value=classification_line_param0,
            old_value=legacy_options.pop("line_param0", None),
            old_name="line_param0",
            new_name="classification_line_param0",
        )
        classification_line_param1 = self._resolve_deprecated_alias(
            new_value=classification_line_param1,
            old_value=legacy_options.pop("line_param1", None),
            old_name="line_param1",
            new_name="classification_line_param1",
        )

        if not isinstance(schedule, PulseSchedule):
            schedule = PulseSchedule.from_waveforms(schedule)

        measurement_config = self.measurement_config_factory.create(
            n_shots=n_shots,
            shot_interval=shot_interval,
            shot_averaging=shot_averaging,
            time_integration=time_integration,
            state_classification=state_classification,
        )

        measurement_schedule = self.build_measurement_schedule(
            pulse_schedule=schedule,
            frequencies=frequencies,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramp_time=readout_ramp_time,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            readout_amplification=readout_amplification,
            final_measurement=final_measurement,
            plot=plot,
        )

        if classification_line_param0 is None and classification_line_param1 is None:
            result = _run_async(
                lambda: self.run_measurement(
                    schedule=measurement_schedule,
                    config=measurement_config,
                )
            )
        else:
            quel1_options = Quel1MeasurementOptions(
                classification_line_param0=classification_line_param0,
                classification_line_param1=classification_line_param1,
            )
            result = _run_async(
                lambda: self.run_measurement(
                    schedule=measurement_schedule,
                    config=measurement_config,
                    quel1_options=quel1_options,
                )
            )

        rawdata_dir = self.system_manager.rawdata_dir
        if rawdata_dir is not None and save_result:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            result_path = Path(rawdata_dir) / f"{timestamp}.nc"
            result.save(result_path)

        return MeasurementResultConverter.to_multiple_measure_result(
            result,
            config=self._resolve_device_config(self.backend_controller),
            classifiers=self.classifiers,
        )

    def capture_loopback(
        self,
        schedule: PulseSchedule | TargetMap[IQArray],
        *,
        n_shots: int | None = None,
    ) -> MeasurementResult:
        """
        Capture full-span loopback data on read-in and monitor input channels.

        Parameters
        ----------
        schedule : PulseSchedule | TargetMap[IQArray]
            Pulse schedule or control waveforms to execute.
        n_shots : int | None, optional
            Number of shots.

        Returns
        -------
        MeasurementResult
            Measurement result for loopback capture windows.
        """
        if not isinstance(schedule, PulseSchedule):
            schedule = PulseSchedule.from_waveforms(schedule)

        base_schedule = schedule.copy()
        capture_targets = self._resolve_loopback_capture_targets(schedule=base_schedule)
        measurement_config = self.measurement_config_factory.create(
            n_shots=n_shots,
            shot_averaging=False,
            time_integration=False,
            state_classification=False,
        )

        def _run_once(targets: Sequence[str]) -> MeasurementResult:
            measurement_schedule = self.build_measurement_schedule(
                pulse_schedule=base_schedule.copy(),
                capture_placement="entire_schedule",
                capture_targets=list(targets),
                final_measurement=False,
                readout_amplification=False,
                plot=False,
            )
            loopback_box_ids = self._resolve_loopback_box_ids(
                schedule=base_schedule,
                capture_targets=targets,
            )
            self._initialize_loopback_capture_units(box_ids=loopback_box_ids)
            with self._temporary_loopback_rfswitches(capture_targets=targets):
                return _run_async(
                    lambda: self.run_measurement(
                        schedule=measurement_schedule,
                        config=measurement_config,
                        quel1_options=Quel1MeasurementOptions(demodulation=False),
                    )
                )

        try:
            return _run_once(capture_targets)
        except Exception as exc:
            if not self._is_e7awg_capture_data_error(exc):
                raise
            logger.warning(
                "Loopback capture failed with broken-data error; retrying once after capture-unit initialization."
            )
            try:
                return _run_once(capture_targets)
            except Exception as retry_exc:
                if not self._is_e7awg_capture_data_error(retry_exc):
                    raise
                read_in_only_targets = self._filter_loopback_capture_targets(
                    capture_targets=capture_targets,
                    port_type=PortType.READ_IN,
                )
                if not read_in_only_targets or read_in_only_targets == list(
                    capture_targets
                ):
                    raise
                logger.warning(
                    "Loopback capture still failed; retrying with READ_IN targets only."
                )
                return _run_once(read_in_only_targets)

    def create_measurement_config(
        self,
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
    ) -> MeasurementConfig:
        """
        Create a `MeasurementConfig` from optional runtime overrides.

        Parameters
        ----------
        n_shots : int | None, optional
            Number of shots.
        shot_interval : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether to average shots on hardware.
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.

        Returns
        -------
        MeasurementConfig
            Created measurement configuration.

        """
        return self.measurement_config_factory.create(
            n_shots=n_shots,
            shot_interval=shot_interval,
            shot_averaging=shot_averaging,
            time_integration=time_integration,
            state_classification=state_classification,
        )

    def build_measurement_schedule(
        self,
        pulse_schedule: PulseSchedule,
        *,
        frequencies: dict[str, float] | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramp_time: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_drag_coeff: float | None = None,
        readout_amplification: bool | None = None,
        final_measurement: bool | None = None,
        capture_placement: CapturePlacement | None = None,
        capture_targets: list[str] | None = None,
        plot: bool | None = None,
    ) -> MeasurementSchedule:
        """Build a `MeasurementSchedule` from a pulse schedule and options."""
        measurement_schedule = self.schedule_builder.build(
            schedule=pulse_schedule,
            frequencies=frequencies,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramp_time=readout_ramp_time,
            readout_ramp_type=readout_ramp_type,
            readout_drag_coeff=readout_drag_coeff,
            readout_amplification=readout_amplification,
            final_measurement=final_measurement,
            capture_placement=capture_placement,
            capture_targets=capture_targets,
            plot=plot,
        )
        return measurement_schedule

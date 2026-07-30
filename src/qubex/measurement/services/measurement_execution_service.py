"""Execution services for measurement workflows."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike
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
from qubex.measurement.models.capture_data import CaptureData
from qubex.measurement.models.capture_schedule import CaptureSchedule
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
    ControlParameters,
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
MeasurementResultSplitPlan: TypeAlias = list[dict[str, int]]


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
            measurement_defaults=getattr(
                self.experiment_system, "measurement_defaults", None
            ),
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
        sampling_period = self.backend_controller.sampling_period_ns
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
    def control_params(self) -> ControlParameters:
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
        return await self.measurement_schedule_runner.execute_async(
            schedule=schedule,
            config=config,
            quel1_options=quel1_options,
        )

    async def run_sweep_measurement(
        self,
        schedule: Callable[[SweepValue], MeasurementSchedule],
        *,
        sweep_values: ArrayLike | Sequence[SweepValue],
        config: MeasurementConfig | None = None,
        on_point: Callable[[SweepValue, MeasurementResult], None] | None = None,
    ) -> SweepMeasurementResult:
        """
        Run sweep measurement pointwise.

        Parameters
        ----------
        schedule : Callable[[SweepValue], MeasurementSchedule]
            Callback that builds one measurement schedule per sweep value.
        sweep_values : ArrayLike | Sequence[SweepValue]
            Ordered sweep values to execute.
        config : MeasurementConfig | None, optional
            Shared measurement configuration for all points.
        on_point : Callable[[SweepValue, MeasurementResult], None] | None, optional
            Callback invoked after each point measurement completes.

        Returns
        -------
        SweepMeasurementResult
            Sweep result list in the same order as input values.
        """
        resolved_config = self.create_measurement_config() if config is None else config
        normalized_values = cast(list[SweepValue], np.asarray(sweep_values).tolist())
        if self._can_use_batch_execution():
            measurement_schedules = [
                schedule(sweep_value) for sweep_value in normalized_values
            ]
            results = await self._execute_measurement_schedules(
                schedules=measurement_schedules,
                config=resolved_config,
            )
        else:
            results: list[MeasurementResult] = []
            for sweep_value in normalized_values:
                measurement_schedule = schedule(sweep_value)
                result = await self.run_measurement(
                    schedule=measurement_schedule,
                    config=resolved_config,
                )
                results.append(result)
                if on_point is not None:
                    on_point(sweep_value, result)
            return SweepMeasurementResult(
                sweep_values=normalized_values,
                config=resolved_config,
                results=results,
            )
        if on_point is not None:
            for sweep_value, result in zip(normalized_values, results, strict=True):
                on_point(sweep_value, result)
        return SweepMeasurementResult(
            sweep_values=normalized_values,
            config=resolved_config,
            results=results,
        )

    async def run_ndsweep_measurement(
        self,
        schedule: Callable[[SweepPoint], MeasurementSchedule],
        *,
        sweep_points: Mapping[SweepKey, Sequence[SweepValue]],
        sweep_axes: SweepAxes | None = None,
        config: MeasurementConfig | None = None,
    ) -> NDSweepMeasurementResult:
        """
        Run N-dimensional Cartesian-product sweep measurement pointwise.

        Parameters
        ----------
        schedule : Callable[[SweepPoint], MeasurementSchedule]
            Callback that builds one measurement schedule per resolved sweep point.
        sweep_points : Mapping[SweepKey, Sequence[SweepValue]]
            Axis-value table (`axis key -> ordered values`).
        sweep_axes : SweepAxes | None, optional
            Axis order for Cartesian product. If omitted, insertion order of
            `sweep_points` is used for `dict` inputs. Other mapping inputs must
            provide this explicitly.
        config : MeasurementConfig | None, optional
            Shared measurement configuration for all points.

        Returns
        -------
        NDSweepMeasurementResult
            Cartesian sweep result with C-order flattening (last axis varies fastest).

        Raises
        ------
        ValueError
            If `sweep_axes` does not match `sweep_points` keys exactly, or if
            `sweep_axes` is omitted for a non-`dict` mapping input.
        """
        resolved_config = self.create_measurement_config() if config is None else config
        if sweep_axes is None:
            if not isinstance(sweep_points, dict):
                raise ValueError(
                    "sweep_axes must be provided when sweep_points is not a "
                    "dict-derived insertion-ordered mapping."
                )
            resolved_axes = tuple(sweep_points.keys())
        else:
            resolved_axes = tuple(sweep_axes)
        if len(set(resolved_axes)) != len(resolved_axes):
            raise ValueError("sweep_axes must not contain duplicate keys.")
        if set(resolved_axes) != set(sweep_points.keys()):
            raise ValueError(
                "sweep_axes must contain each sweep_points key exactly once."
            )

        normalized_axes_points = {axis: [*sweep_points[axis]] for axis in resolved_axes}
        shape = tuple(len(normalized_axes_points[axis]) for axis in resolved_axes)
        if self._can_use_batch_execution():
            points: list[dict[SweepKey, SweepValue]] = [
                {
                    axis: normalized_axes_points[axis][axis_index]
                    for axis, axis_index in zip(resolved_axes, ndindex, strict=True)
                }
                for ndindex in np.ndindex(shape)
            ]
            measurement_schedules = [schedule(point) for point in points]
            results = await self._execute_measurement_schedules(
                schedules=measurement_schedules,
                config=resolved_config,
            )
        else:
            results: list[MeasurementResult] = []
            for ndindex in np.ndindex(shape):
                point: dict[SweepKey, SweepValue] = {
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

    async def _execute_measurement_schedules(
        self,
        *,
        schedules: list[MeasurementSchedule],
        config: MeasurementConfig,
    ) -> list[MeasurementResult]:
        """Execute multiple schedules either as one packed timeline or batch requests."""
        runner = self.measurement_schedule_runner
        if self._should_pack_measurement_schedules(
            runner=runner,
            schedules=schedules,
            config=config,
        ):
            return await self._execute_measurement_schedules_as_packed_timeline(
                runner=runner,
                schedules=schedules,
                config=config,
            )
        return await runner.execute_batch_async(
            schedules=schedules,
            config=config,
        )

    async def _execute_measurement_schedules_as_packed_timeline(
        self,
        *,
        runner: MeasurementScheduleRunner,
        schedules: list[MeasurementSchedule],
        config: MeasurementConfig,
    ) -> list[MeasurementResult]:
        """Execute packed timeline chunks and split merged results per schedule."""
        results: list[MeasurementResult] = []
        for chunk in self._split_measurement_schedules_into_packed_timeline_chunks(
            schedules=schedules,
            config=config,
        ):
            results.extend(
                await self._execute_measurement_schedule_chunk_as_packed_timeline(
                    runner=runner,
                    schedules=chunk,
                    config=config,
                )
            )
        return results

    async def _execute_measurement_schedule_chunk_as_packed_timeline(
        self,
        *,
        runner: MeasurementScheduleRunner,
        schedules: list[MeasurementSchedule],
        config: MeasurementConfig,
    ) -> list[MeasurementResult]:
        """Execute one packed timeline chunk and split merged results per schedule."""
        split_plan = self._build_measurement_result_split_plan(schedules=schedules)
        merged_schedule = self._merge_measurement_schedules(
            schedules=schedules,
            shot_interval=config.shot_interval,
        )
        merged_measurement_result = await runner.execute_async(
            schedule=merged_schedule,
            config=config,
        )
        return self._split_merged_measurement_result(
            merged_result=merged_measurement_result,
            split_plan=split_plan,
        )

    def _split_measurement_schedules_into_packed_timeline_chunks(
        self,
        *,
        schedules: list[MeasurementSchedule],
        config: MeasurementConfig,
    ) -> list[list[MeasurementSchedule]]:
        """
        Split packed timeline chunks by the repeated-duration soft limit.

        A single over-limit schedule remains in its own chunk.
        """
        limit = config.max_repeated_timeline_duration_ns
        if limit is None:
            return [schedules]

        chunks: list[list[MeasurementSchedule]] = []
        current_chunks: list[MeasurementSchedule] = []
        for schedule in schedules:
            if self._packed_timeline_repeated_duration_within_limit(
                limit, config.n_shots, config.shot_interval, [*current_chunks, schedule]
            ):
                current_chunks.append(schedule)
            else:
                if len(current_chunks) == 0:
                    chunks.append([schedule])
                else:
                    chunks.append(current_chunks)
                    current_chunks = [schedule]

        if len(current_chunks) > 0:
            chunks.append(current_chunks)

        return chunks

    @staticmethod
    def _packed_timeline_repeated_duration_within_limit(
        limit: float,
        n_shots: int,
        shot_interval: float,
        schedules: Sequence[MeasurementSchedule],
    ) -> bool:
        total_duration = 0.0
        for schedule in schedules:
            total_duration += schedule.pulse_schedule.duration

        repeated_total_duration = (
            total_duration + (len(schedules) - 1) * shot_interval
        ) * n_shots
        return repeated_total_duration <= limit

    def _should_pack_measurement_schedules(
        self,
        *,
        runner: MeasurementScheduleRunner,
        schedules: Sequence[MeasurementSchedule],
        config: MeasurementConfig,
    ) -> bool:
        """Return whether all schedules should be packed into one timeline."""
        del runner
        if len(schedules) <= 1:
            return False
        if not config.should_use_schedule_packing:
            return False
        return self._can_merge_measurement_schedules(schedules=schedules)

    def _can_merge_measurement_schedules(
        self,
        *,
        schedules: Sequence[MeasurementSchedule],
    ) -> bool:
        """Return whether schedules can be concatenated without changing channel metadata."""
        if len(schedules) <= 1:
            return False
        first_schedule = schedules[0].pulse_schedule
        if not isinstance(first_schedule, PulseSchedule):
            return False
        labels = first_schedule.labels
        sampling_periods = self._schedule_sampling_periods(first_schedule)
        if sampling_periods is None:
            return False
        frequencies = self._schedule_frequencies(first_schedule)
        if frequencies is None:
            return False

        for schedule in schedules[1:]:
            pulse_schedule = schedule.pulse_schedule
            if not isinstance(pulse_schedule, PulseSchedule):
                return False
            if pulse_schedule.labels != labels:
                return False
            if self._schedule_sampling_periods(pulse_schedule) != sampling_periods:
                return False
            next_frequencies = self._schedule_frequencies(pulse_schedule)
            if next_frequencies is None:
                return False
            for label, frequency in frequencies.items():
                if not self._same_optional_frequency(
                    frequency,
                    next_frequencies[label],
                ):
                    return False
        return True

    @staticmethod
    def _schedule_sampling_periods(
        pulse_schedule: PulseSchedule,
    ) -> dict[str, float] | None:
        """Return sampling periods keyed by label when all channel metadata is valid."""
        sampling_periods: dict[str, float] = {}
        for label in pulse_schedule.labels:
            try:
                sequence = pulse_schedule.get_sequence(label, copy=False)
            except KeyError:
                return None
            sampling_period = getattr(sequence, "sampling_period", None)
            if not isinstance(sampling_period, (int, float)):
                return None
            sampling_periods[label] = float(sampling_period)
        return sampling_periods

    @staticmethod
    def _schedule_frequencies(
        pulse_schedule: PulseSchedule,
    ) -> dict[str, float | None] | None:
        """Return frequencies keyed by label when frequency metadata is valid."""
        raw_frequencies = pulse_schedule.get_frequencies()
        if not isinstance(raw_frequencies, Mapping):
            return None
        frequencies: dict[str, float | None] = {}
        labels = set(pulse_schedule.labels)
        for label, frequency in raw_frequencies.items():
            if label not in labels:
                return None
            if frequency is None:
                frequencies[label] = None
                continue
            if not isinstance(frequency, (int, float)):
                return None
            frequencies[label] = float(frequency)
        return frequencies

    @staticmethod
    def _same_optional_frequency(
        left: float | None,
        right: float | None,
    ) -> bool:
        """Return whether two optional schedule frequencies are equivalent."""
        if left is None or right is None:
            return left is right
        return bool(np.isclose(left, right))

    def _build_measurement_result_split_plan(
        self,
        *,
        schedules: list[MeasurementSchedule],
    ) -> MeasurementResultSplitPlan:
        """Build per-schedule visible capture counts keyed by canonical output target."""
        split_plan: MeasurementResultSplitPlan = []
        for schedule in schedules:
            capture_counts: dict[str, int] = {}
            for capture in schedule.capture_schedule.captures:
                if capture.is_workaround:
                    continue
                for channel in capture.channels:
                    output_target = self._resolve_capture_output_target(channel)
                    capture_counts[output_target] = (
                        capture_counts.get(output_target, 0) + 1
                    )
            split_plan.append(capture_counts)
        return split_plan

    def _resolve_capture_output_target(self, capture_channel: str) -> str:
        """Resolve the canonical measurement-result key for one capture channel."""
        target_registry = self._context.experiment_system.target_registry
        return str(target_registry.measurement_output_label(capture_channel))

    def _merge_measurement_schedules(
        self,
        *,
        schedules: list[MeasurementSchedule],
        shot_interval: float,
    ) -> MeasurementSchedule:
        """Build one measurement schedule by concatenating schedules separated by shot interval."""
        if len(schedules) == 0:
            raise ValueError("At least one schedule is required.")

        merged_pulse_schedule = schedules[0].pulse_schedule.copy()
        merged_captures = [
            capture.model_copy(update={"channels": capture.channels.copy()})
            for capture in schedules[0].capture_schedule.captures
        ]

        for schedule in schedules[1:]:
            # TODO: Base packed shifts on the full capture timeline, not only pulse duration.
            shift_duration = merged_pulse_schedule.duration + shot_interval
            merged_pulse_schedule.barrier()
            merged_pulse_schedule.pad(shift_duration)
            merged_pulse_schedule.call(schedule.pulse_schedule, copy=True)
            merged_captures.extend(
                capture.model_copy(
                    update={
                        "start_time": capture.start_time + shift_duration,
                        "channels": capture.channels.copy(),
                    }
                )
                for capture in schedule.capture_schedule.captures
            )

        return MeasurementSchedule(
            pulse_schedule=merged_pulse_schedule,
            capture_schedule=CaptureSchedule(captures=merged_captures),
        )

    def _split_merged_measurement_result(
        self,
        *,
        merged_result: MeasurementResult,
        split_plan: MeasurementResultSplitPlan,
    ) -> list[MeasurementResult]:
        """Split one merged measurement result by canonical capture ordering."""
        remaining_data: dict[str, list[CaptureData]] = {
            target: [*captures] for target, captures in merged_result.data.items()
        }
        split_results: list[MeasurementResult] = []

        for capture_counts in split_plan:
            chunk_data: dict[str, list[CaptureData]] = {}
            for target, capture_count in capture_counts.items():
                captures = remaining_data.setdefault(target, [])
                if capture_count > len(captures):
                    raise ValueError(
                        f"Not enough captures for target `{target}` to split "
                        "packed timeline: "
                        f"expected {capture_count}, got {len(captures)}."
                    )
                chunk_data[target] = captures[:capture_count]
                remaining_data[target] = captures[capture_count:]

            classifier_refs = None
            if merged_result.classifier_refs is not None:
                classifier_refs = {
                    target: classifier_ref
                    for target, classifier_ref in merged_result.classifier_refs.items()
                    if target in chunk_data
                }
                if len(classifier_refs) == 0:
                    classifier_refs = None

            split_results.append(
                MeasurementResult(
                    data=chunk_data,
                    measurement_config=merged_result.measurement_config,
                    device_config=merged_result.device_config,
                    classifier_refs=classifier_refs,
                )
            )

        for captures in remaining_data.values():
            if len(captures) > 0:
                raise ValueError(
                    "Packed measurement result contains unmatched captures after split."
                )

        return split_results

    def _can_use_batch_execution(self) -> bool:
        """Return whether backend batch execution can be used safely."""
        execute_batch_async = getattr(
            self.backend_controller, "execute_batch_async", None
        )
        if not callable(execute_batch_async):
            return False
        run_measurement_func = getattr(self.run_measurement, "__func__", None)
        if run_measurement_func is not MeasurementExecutionService.run_measurement:
            return False
        try:
            _ = self.experiment_system
        except ValueError:
            return False
        return True

    async def measure_noise(
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
        return await self.run_measurement(
            schedule=measurement_schedule,
            config=measurement_config,
        )

    def measure(
        self,
        waveforms: Mapping[str, IQArray],
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
        frequencies: dict[str, float] | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramp_time: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_amplification: bool | None = None,
        classification_line_param0: tuple[float, float, float] | None = None,
        classification_line_param1: tuple[float, float, float] | None = None,
        plot: bool | None = None,
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
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.
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
        classification_line_param0 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 0.
        classification_line_param1 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 1.

        Returns
        -------
        MeasureResult
            Measurement results.

        """
        if (
            time_integration is None
            and deprecated_options.get("enable_dsp_sum") is None
        ):
            time_integration = False

        result = self.execute(
            schedule=waveforms,
            n_shots=n_shots,
            shot_interval=shot_interval,
            shot_averaging=shot_averaging,
            time_integration=time_integration,
            state_classification=state_classification,
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
        time_integration: bool | None = None,
        state_classification: bool | None = None,
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
        classification_line_param0: tuple[float, float, float] | None = None,
        classification_line_param1: tuple[float, float, float] | None = None,
        plot: bool | None = None,
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
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.
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
        classification_line_param0 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 0.
        classification_line_param1 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 1.
        plot : bool | None, optional
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
        if plot is None:
            plot = False

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
            result = self.measurement_schedule_runner.execute_sync(
                schedule=measurement_schedule,
                config=measurement_config,
            )
        else:
            quel1_options = Quel1MeasurementOptions(
                classification_line_param0=classification_line_param0,
                classification_line_param1=classification_line_param1,
            )
            result = self.measurement_schedule_runner.execute_sync(
                schedule=measurement_schedule,
                config=measurement_config,
                quel1_options=quel1_options,
            )

        rawdata_dir = self.system_manager.rawdata_dir
        if rawdata_dir is not None:
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
            schedule_packing_enabled=self._resolve_schedule_packing_enabled(),
            max_repeated_timeline_duration_ns=(
                self._resolve_max_repeated_timeline_duration_ns()
            ),
        )

    def _resolve_schedule_packing_enabled(self) -> bool:
        """Return whether measurement schedules should be packed into timelines."""
        config = self._resolve_schedule_packing_config()
        value = config.get("enabled", False)
        if isinstance(value, bool):
            return value
        raise TypeError("`measurement.schedule_packing.enabled` must be a boolean.")

    def _resolve_max_repeated_timeline_duration_ns(self) -> float | None:
        """Return repeated packed-timeline duration limit in ns."""
        config = self._resolve_schedule_packing_config()
        value = config.get("max_repeated_timeline_duration_ns")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                "`measurement.schedule_packing.max_repeated_timeline_duration_ns` "
                "must be a number."
            )
        if value <= 0:
            raise ValueError(
                "`measurement.schedule_packing.max_repeated_timeline_duration_ns` "
                "must be positive."
            )
        return float(value)

    def _resolve_schedule_packing_config(self) -> dict[str, Any]:
        """Return schedule packing config from generic measurement config."""
        measurement_config = getattr(self.config_loader, "measurement_config", {})
        if not isinstance(measurement_config, Mapping):
            raise TypeError("`measurement` section must be a mapping.")
        value = measurement_config.get("schedule_packing", {})
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        raise TypeError("`measurement.schedule_packing` must be a mapping.")

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

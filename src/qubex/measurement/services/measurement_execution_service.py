"""Execution services for measurement workflows."""

from __future__ import annotations

import logging
import threading
import warnings
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from typing import Any, TypeVar

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
from qubex.core.async_bridge import AsyncBridge
from qubex.measurement.classifiers.state_classifier import StateClassifier
from qubex.measurement.measurement_config_factory import MeasurementConfigFactory
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_context import MeasurementContext
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
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
    SystemManager,
    Target,
)
from qubex.typing import IQArray, TargetMap

logger = logging.getLogger(__name__)

R = TypeVar("R")
TOption = TypeVar("TOption")
_SYNC_BRIDGE_TIMEOUT_SECONDS = 300.0
_MEASUREMENT_ASYNC_BRIDGE_LOCK = threading.Lock()
_MEASUREMENT_ASYNC_BRIDGE: AsyncBridge | None = None


def _get_measurement_async_bridge() -> AsyncBridge:
    """Return the module-level async bridge singleton."""
    global _MEASUREMENT_ASYNC_BRIDGE
    with _MEASUREMENT_ASYNC_BRIDGE_LOCK:
        if _MEASUREMENT_ASYNC_BRIDGE is None:
            _MEASUREMENT_ASYNC_BRIDGE = AsyncBridge(
                default_timeout=_SYNC_BRIDGE_TIMEOUT_SECONDS,
                thread_name="qubex-measurement-async-bridge",
            )
        return _MEASUREMENT_ASYNC_BRIDGE


def _run_async(
    factory: Callable[[], Awaitable[R]],
    *,
    timeout: float = _SYNC_BRIDGE_TIMEOUT_SECONDS,
) -> R:
    """Run one awaitable factory from synchronous APIs."""
    return _get_measurement_async_bridge().run(factory, timeout=timeout)


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
        new_value: TOption | None,
        old_value: TOption | None,
        old_name: str,
        new_name: str,
    ) -> TOption | None:
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
            The target label.

        Returns
        -------
        float
            The AWG frequency in Hz.
        """
        return self.experiment_system.get_awg_frequency(target)

    def get_diff_frequency(self, target: str) -> float:
        """
        Get the difference frequency for the target.

        Parameters
        ----------
        target : str
            The target label.

        Returns
        -------
        float
            The difference frequency in Hz.
        """
        return self.experiment_system.get_diff_frequency(target)

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
            The measurement schedule.
        config : MeasurementConfig
            The measurement configuration.

        Returns
        -------
        MeasurementResult
            The measurement result.
        """
        if quel1_options is None:
            result = await self.measurement_schedule_runner.execute(
                schedule=schedule,
                config=config,
            )
        else:
            result = await self.measurement_schedule_runner.execute(
                schedule=schedule,
                config=config,
                quel1_options=quel1_options,
            )
        return result

    async def run_sweep_measurement(
        self,
        schedule: Callable[[SweepPoint], MeasurementSchedule],
        *,
        sweep_points: Sequence[SweepPoint],
        config: MeasurementConfig | None = None,
    ) -> SweepMeasurementResult:
        """
        Run sweep measurement pointwise.

        Parameters
        ----------
        schedule : Callable[[SweepPoint], MeasurementSchedule]
            Callback that builds one measurement schedule per sweep point.
        sweep_points : Sequence[SweepPoint]
            Ordered sweep points to execute.
        config : MeasurementConfig | None, optional
            Shared measurement configuration for all points.

        Returns
        -------
        SweepMeasurementResult
            Sweep result list in the same order as input points.
        """
        resolved_config = self.create_measurement_config() if config is None else config
        normalized_points = [dict(point) for point in sweep_points]
        results: list[MeasurementResult] = []
        for point in normalized_points:
            measurement_schedule = schedule(dict(point))
            result = await self.run_measurement(
                schedule=measurement_schedule,
                config=resolved_config,
            )
            results.append(result)
        return SweepMeasurementResult(
            sweep_points=normalized_points,
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
    ) -> MeasureResult:
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
        MeasureResult
            The measurement results.
        """
        return self.measure(
            waveforms={target: np.zeros(0) for target in targets},
            shot_averaging=True,
            n_shots=1,
            readout_duration=duration,
            readout_amplitudes=dict.fromkeys(targets, 0),
            time_integration=False,
        )

    def measure(
        self,
        waveforms: Mapping[str, IQArray],
        *,
        n_shots: int | None = None,
        shot_interval_ns: float | None = None,
        shot_averaging: bool | None = None,
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
            The control waveforms for each target.
        n_shots : int | None, optional
            Number of shots.
        shot_interval_ns : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether to average shots on hardware.
        readout_amplitudes : dict[str, float], optional
            The readout amplitude for each qubit.
        readout_duration : float, optional
            The readout duration in ns.
        readout_pre_margin : float, optional
            The readout pre-margin in ns.
        readout_post_margin : float, optional
            The readout post-margin in ns.
        readout_ramp_time : float, optional
            The readout ramp time in ns.
        readout_drag_coeff : float, optional
            The readout drag coefficient.
        readout_ramp_type : RampType, optional
            The readout ramp type.
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
            The measurement results.
        """
        result = self.execute(
            schedule=waveforms,
            n_shots=n_shots,
            shot_interval_ns=shot_interval_ns,
            shot_averaging=shot_averaging,
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
        shot_interval_ns: float | None = None,
        shot_averaging: bool | None = None,
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
            The pulse schedule or control waveforms.
        n_shots : int | None, optional
            Number of shots.
        shot_interval_ns : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether to average shots on hardware.
        readout_amplitudes : dict[str, float], optional
            The readout amplitude for each qubit.
        readout_duration : float, optional
            The readout duration in ns.
        readout_pre_margin : float, optional
            The readout pre-margin in ns.
        readout_post_margin : float, optional
            The readout post-margin in ns.
        readout_ramp_time : float, optional
            The readout ramp time in ns.
        readout_drag_coeff : float, optional
            The readout drag coefficient.
        readout_ramp_type : RampType, optional
            The readout ramp type.
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
            The measurement results.
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
        shot_interval_ns = self._resolve_deprecated_alias(
            new_value=shot_interval_ns,
            old_value=legacy_options.pop("interval", None),
            old_name="interval",
            new_name="shot_interval_ns",
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
            shot_interval_ns=shot_interval_ns,
            shot_averaging=shot_averaging,
            time_integration=time_integration,
            state_classification=state_classification,
        )

        measurement_schedule = self.build_measurement_schedule(
            pulse_schedule=schedule,
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
                line_param0=classification_line_param0,
                line_param1=classification_line_param1,
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
            result.save(rawdata_dir)

        return MeasurementResultConverter.to_multiple_measure_result(
            result,
            config=self._resolve_device_config(self.backend_controller),
            classifiers=self.classifiers,
        )

    def create_measurement_config(
        self,
        *,
        n_shots: int | None = None,
        shot_interval_ns: float | None = None,
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
        shot_interval_ns : float | None, optional
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
            The created measurement configuration.
        """
        return self.measurement_config_factory.create(
            n_shots=n_shots,
            shot_interval_ns=shot_interval_ns,
            shot_averaging=shot_averaging,
            time_integration=time_integration,
            state_classification=state_classification,
        )

    def build_measurement_schedule(
        self,
        pulse_schedule: PulseSchedule,
        *,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramp_time: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_drag_coeff: float | None = None,
        readout_amplification: bool = False,
        final_measurement: bool = False,
        plot: bool = False,
    ) -> MeasurementSchedule:
        """Build a `MeasurementSchedule` from a pulse schedule and options."""
        measurement_schedule = self.schedule_builder.build(
            schedule=pulse_schedule,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramp_time=readout_ramp_time,
            readout_ramp_type=readout_ramp_type,
            readout_drag_coeff=readout_drag_coeff,
            readout_amplification=readout_amplification,
            final_measurement=final_measurement,
            plot=plot,
        )
        return measurement_schedule

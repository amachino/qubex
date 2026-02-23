"""Execution services for measurement workflows."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping

import numpy as np
from qxpulse import PulseSchedule, RampType

from qubex.backend import (
    BackendController,
    ConfigLoader,
    ControlParams,
    ExperimentSystem,
    Mux,
    SystemManager,
    Target,
)
from qubex.backend.quel1 import (
    ExecutionMode,
)
from qubex.typing import IQArray, MeasurementMode, TargetMap

from .classifiers.state_classifier import StateClassifier
from .measurement_config_factory import MeasurementConfigFactory
from .measurement_constraint_profile import MeasurementConstraintProfile
from .measurement_context import MeasurementContext
from .measurement_pulse_factory import MeasurementPulseFactory
from .measurement_result_converter import MeasurementResultConverter
from .measurement_schedule_builder import MeasurementScheduleBuilder
from .measurement_schedule_runner import MeasurementScheduleRunner
from .measurement_session_service import MeasurementSessionService
from .models.measure_result import (
    MeasureResult,
    MultipleMeasureResult,
)
from .models.measurement_config import MeasurementConfig
from .models.measurement_result import MeasurementResult
from .models.measurement_schedule import MeasurementSchedule

logger = logging.getLogger(__name__)


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
        """Return measurement context accessor."""
        return self._context

    @property
    def session_service(self) -> MeasurementSessionService:
        """Return session lifecycle service."""
        return self._session_service

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        """Return classifier mapping used for result conversion."""
        return self._classifiers

    @property
    def system_manager(self) -> SystemManager:
        """Return system manager from measurement context."""
        return self.context.system_manager

    @property
    def config_loader(self) -> ConfigLoader:
        """Get the configuration loader."""
        return self.context.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        return self.context.experiment_system

    @property
    def backend_controller(self) -> BackendController:
        """Get the backend controller."""
        return self.session_service.backend_controller

    @property
    def mux_dict(self) -> dict[str, Mux]:
        """Get a dictionary of muxes indexed by qubit labels."""
        return self.context.mux_dict

    @property
    def pulse_factory(self) -> MeasurementPulseFactory:
        """Create a pulse factory from current system state."""
        target_registry = getattr(self.experiment_system, "target_registry", None)
        return MeasurementPulseFactory(
            control_params=self.control_params,
            mux_dict=self.mux_dict,
            target_registry=target_registry,
        )

    @property
    def schedule_builder(self) -> MeasurementScheduleBuilder:
        """Create a measurement schedule builder from current system state."""
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
        """Create a measurement config factory from current system state."""
        return MeasurementConfigFactory(
            experiment_system=self.experiment_system,
        )

    @property
    def sampling_period(self) -> float:
        """Resolve sampling period (ns) from backend-controller contract."""
        return self.constraint_profile.sampling_period_ns

    @property
    def constraint_profile(self) -> MeasurementConstraintProfile:
        """Resolve backend constraint profile from backend-controller hints."""
        profile = getattr(
            self.backend_controller, "MEASUREMENT_CONSTRAINT_PROFILE", None
        )
        mode = getattr(self.backend_controller, "MEASUREMENT_CONSTRAINT_MODE", "quel1")
        sampling_period = self.backend_controller.sampling_period
        if isinstance(profile, MeasurementConstraintProfile):
            return profile
        if mode == "quel3":
            return MeasurementConstraintProfile.quel3(sampling_period)
        return MeasurementConstraintProfile.quel1(sampling_period)

    @property
    def measurement_schedule_runner(self) -> MeasurementScheduleRunner:
        """Return executor implementation used by schedule execution APIs."""
        return MeasurementScheduleRunner.create_default(
            backend_controller=self.backend_controller,
            experiment_system=self.experiment_system,
            execution_mode=self._execution_mode,
            clock_health_checks=self._clock_health_checks,
        )

    @property
    def control_params(self) -> ControlParams:
        """Get the control parameters."""
        return self.experiment_system.control_params

    @property
    def chip_id(self) -> str:
        """Get the chip ID."""
        return self.experiment_system.chip.id

    @property
    def targets(self) -> dict[str, Target]:
        """Get the targets."""
        return {target.label: target for target in self.experiment_system.targets}

    @property
    def nco_frequencies(self) -> dict[str, float]:
        """Get the NCO frequencies."""
        return {
            target.label: self.experiment_system.get_nco_frequency(target.label)
            for target in self.experiment_system.targets
        }

    @property
    def awg_frequencies(self) -> dict[str, float]:
        """Get the AWG frequencies."""
        return {
            target.label: self.experiment_system.get_awg_frequency(target.label)
            for target in self.experiment_system.targets
        }

    @staticmethod
    def _resolve_device_config(
        backend_controller: BackendController,
    ) -> dict:
        """Resolve backend device config if the backend exposes it."""
        box_config = getattr(backend_controller, "box_config", None)
        if isinstance(box_config, dict):
            return box_config
        return {}

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

    def execute_measurement_schedule(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
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
        result = self.measurement_schedule_runner.execute(
            schedule=schedule,
            config=config,
        )
        return result

    def measure_noise(
        self,
        targets: Collection[str],
        *,
        duration: float,
        enable_dsp_sum: bool = False,
    ) -> MeasureResult:
        """
        Measure readout noise.

        Parameters
        ----------
        targets : Collection[str]
            Target labels to measure.
        duration : float
            Readout duration in ns.
        enable_dsp_sum : bool, optional
            Whether to enable DSP summation.

        Returns
        -------
        MeasureResult
            The measurement results.
        """
        return self.measure(
            waveforms={target: np.zeros(0) for target in targets},
            mode="avg",
            shots=1,
            readout_duration=duration,
            readout_amplitudes=dict.fromkeys(targets, 0),
            enable_dsp_sum=enable_dsp_sum,
        )

    def measure(
        self,
        waveforms: Mapping[str, IQArray],
        *,
        mode: MeasurementMode = "avg",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        add_pump_pulses: bool | None = None,
        enable_dsp_demodulation: bool | None = None,
        enable_dsp_sum: bool = True,
        enable_dsp_classification: bool | None = None,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
        plot: bool = False,
    ) -> MeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        waveforms : Mapping[str, IQArray]
            The control waveforms for each target.
        mode : MeasurementMode, optional
            The measurement mode.
        shots : int, optional
            The number of shots.
        interval : float, optional
            The interval in ns.
        readout_amplitudes : dict[str, float], optional
            The readout amplitude for each qubit.
        readout_duration : float, optional
            The readout duration in ns.
        readout_pre_margin : float, optional
            The readout pre-margin in ns.
        readout_post_margin : float, optional
            The readout post-margin in ns.
        readout_ramptime : float, optional
            The readout ramp time in ns.
        readout_drag_coeff : float, optional
            The readout drag coefficient.
        readout_ramp_type : RampType, optional
            The readout ramp type.
        add_pump_pulses : bool | None, optional
            Whether to add pump pulses.
        enable_dsp_demodulation : bool | None, optional
            Whether to enable DSP demodulation.
        enable_dsp_sum : bool, optional
            Whether to enable DSP summation.
        enable_dsp_classification : bool | None, optional
            Whether to enable DSP classification.

        Returns
        -------
        MeasureResult
            The measurement results.
        """
        if add_pump_pulses is None:
            add_pump_pulses = False
        if enable_dsp_demodulation is None:
            enable_dsp_demodulation = True
        if enable_dsp_classification is None:
            enable_dsp_classification = False

        result = self.execute(
            schedule=waveforms,
            mode=mode,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            add_last_measurement=True,
            add_pump_pulses=add_pump_pulses,
            enable_dsp_demodulation=enable_dsp_demodulation,
            enable_dsp_sum=enable_dsp_sum,
            enable_dsp_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
            plot=plot,
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
        mode: MeasurementMode = "avg",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        add_last_measurement: bool = False,
        add_pump_pulses: bool | None = None,
        enable_dsp_demodulation: bool | None = None,
        enable_dsp_sum: bool = True,
        enable_dsp_classification: bool | None = None,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
        plot: bool = False,
        save_result: bool = True,
    ) -> MultipleMeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        schedule : PulseSchedule | TargetMap[IQArray]
            The pulse schedule or control waveforms.
        mode : MeasurementMode, optional
            The measurement mode.
        shots : int, optional
            The number of shots.
        interval : float, optional
            The interval in ns.
        readout_amplitudes : dict[str, float], optional
            The readout amplitude for each qubit.
        readout_duration : float, optional
            The readout duration in ns.
        readout_pre_margin : float, optional
            The readout pre-margin in ns.
        readout_post_margin : float, optional
            The readout post-margin in ns.
        readout_ramptime : float, optional
            The readout ramp time in ns.
        readout_drag_coeff : float, optional
            The readout drag coefficient.
        readout_ramp_type : RampType, optional
            The readout ramp type.
        add_last_measurement : bool, optional
            Whether to add the last measurement.
        add_pump_pulses : bool, optional
            Whether to add pump pulses.
        enable_dsp_sum : bool, optional
            Whether to enable DSP summation.
        enable_dsp_classification : bool, optional
            Whether to enable DSP classification.
        plot : bool, optional
            Whether to plot the results.

        Returns
        -------
        MultipleMeasureResult
            The measurement results.
        """
        if add_pump_pulses is None:
            add_pump_pulses = False
        if enable_dsp_demodulation is None:
            enable_dsp_demodulation = True
        if enable_dsp_classification is None:
            enable_dsp_classification = False

        if not isinstance(schedule, PulseSchedule):
            schedule = PulseSchedule.from_waveforms(schedule)

        run_config = self.measurement_config_factory.create(
            mode=mode,
            shots=shots,
            interval=interval,
            enable_dsp_demodulation=enable_dsp_demodulation,
            enable_dsp_sum=enable_dsp_sum,
            enable_dsp_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )

        measurement_schedule = self.build_measurement_schedule(
            pulse_schedule=schedule,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            add_last_measurement=add_last_measurement,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )

        result = self.execute_measurement_schedule(
            schedule=measurement_schedule,
            config=run_config,
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
        mode: MeasurementMode = "avg",
        shots: int | None = None,
        interval: float | None = None,
        frequencies: dict[str, float] | None = None,
        enable_dsp_demodulation: bool | None = None,
        enable_dsp_sum: bool | None = None,
        enable_dsp_classification: bool | None = None,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> MeasurementConfig:
        """
        Create a `MeasurementConfig` from optional runtime overrides.

        Parameters
        ----------
        mode : MeasurementMode, optional
            The measurement mode, by default "avg".
        shots : int | None, optional
            The number of shots, by default None.
        interval : float | None, optional
            The interval in ns, by default None.
        frequencies : dict[str, float] | None, optional
            The target frequencies in Hz, by default None.
        enable_dsp_demodulation : bool | None, optional
            Whether to enable DSP demodulation, by default None.
        enable_dsp_sum : bool | None, optional
            Whether to enable DSP summation, by default None.
        enable_dsp_classification : bool | None, optional
            Whether to enable DSP classification, by default None.
        line_param0 : tuple[float, float, float] | None, optional
            The DSP line parameter 0, by default None.
        line_param1 : tuple[float, float, float] | None, optional
            The DSP line parameter 1, by default None.

        Returns
        -------
        MeasurementConfig
            The created measurement configuration.
        """
        measurement_config = self.measurement_config_factory.create(
            mode=mode,
            shots=shots,
            interval=interval,
            frequencies=frequencies,
            enable_dsp_demodulation=enable_dsp_demodulation,
            enable_dsp_sum=enable_dsp_sum,
            enable_dsp_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
        return measurement_config

    def build_measurement_schedule(
        self,
        pulse_schedule: PulseSchedule,
        *,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_drag_coeff: float | None = None,
        add_last_measurement: bool = False,
        add_pump_pulses: bool = False,
        plot: bool = False,
    ) -> MeasurementSchedule:
        """Build a `MeasurementSchedule` from a pulse schedule and options."""
        measurement_schedule = self.schedule_builder.build(
            schedule=pulse_schedule,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramptime=readout_ramptime,
            readout_ramp_type=readout_ramp_type,
            readout_drag_coeff=readout_drag_coeff,
            add_last_measurement=add_last_measurement,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )
        return measurement_schedule

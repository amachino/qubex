"""Measurement facade for end-to-end measurement workflows."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy.typing as npt
from qxpulse import PulseSchedule, RampType
from typing_extensions import deprecated

from qubex.backend import (
    BackendController,
    BackendKind,
)
from qubex.backend.quel1 import ExecutionMode
from qubex.measurement.measurement_config_factory import MeasurementConfigFactory
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import (
    MeasurementResult,
)
from qubex.system import (
    ConfigLoader,
    ControlParams,
    ExperimentSystem,
    Mux,
    SystemManager,
    Target,
)
from qubex.typing import ConfigurationMode, IQArray, TargetMap

from .classifiers.state_classifier import StateClassifier
from .measurement_constraint_profile import MeasurementConstraintProfile
from .measurement_context import MeasurementContext
from .measurement_pulse_factory import MeasurementPulseFactory
from .measurement_schedule_builder import (
    CapturePlacement,
    MeasurementScheduleBuilder,
)
from .measurement_schedule_runner import MeasurementScheduleRunner
from .models.measure_result import (
    MeasureResult,
    MultipleMeasureResult,
)
from .models.measurement_schedule import MeasurementSchedule
from .models.sweep_measurement_result import (
    NDSweepMeasurementResult,
    SweepAxes,
    SweepKey,
    SweepMeasurementResult,
    SweepPoint,
    SweepValue,
)
from .services import (
    MeasurementAmplificationService,
    MeasurementClassificationService,
    MeasurementExecutionService,
    MeasurementSessionService,
)

logger = logging.getLogger(__name__)


class Measurement:
    """
    Facade class for end-to-end measurement workflows.

    `Measurement` owns the high-level workflow while delegating concrete
    responsibilities to focused collaborators: context access
    (`MeasurementContext`), configuration/backend lifecycle
    (`MeasurementSessionService`), schedule assembly
    (`MeasurementScheduleBuilder` and `MeasurementPulseFactory`), and execution
    orchestration (`MeasurementExecutionService`), classifier state
    (`MeasurementClassificationService`), and temporary DC operations
    (`MeasurementAmplificationService`).

    """

    _execution_mode: ExecutionMode | None = None
    _clock_health_checks: bool | None = None
    DEFAULT_LOAD_CONFIGS: Final[bool] = True
    DEFAULT_CONNECT_DEVICES: Final[bool] = False
    DEFAULT_CONFIGURATION_MODE: Final[ConfigurationMode] = "ge-cr-cr"

    def __init__(
        self,
        *,
        chip_id: str,
        qubits: Collection[str],
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        load_configs: bool | None = None,
        connect_devices: bool | None = None,
        configuration_mode: ConfigurationMode | None = None,
        backend_kind: BackendKind | None = None,
        _execution_mode: ExecutionMode | None = None,
        _clock_health_checks: bool | None = None,
    ):
        """
        Initialize a measurement facade with optional session bootstrap.

        Parameters
        ----------
        chip_id : str
            Chip identifier used to resolve configuration resources.
        qubits : Collection[str]
            Qubit labels managed by this measurement instance.
        config_dir : Path | str | None, optional
            Base directory that contains system configuration files.
        params_dir : Path | str | None, optional
            Base directory that contains control parameter files.
        load_configs : bool | None, optional
            Whether to call `load` during initialization. If `None`,
            `DEFAULT_LOAD_CONFIGS` is used.
        connect_devices : bool | None, optional
            Whether to call `connect` during initialization. If `None`,
            `DEFAULT_CONNECT_DEVICES` is used.
        configuration_mode : ConfigurationMode | None, optional
            Configuration variant passed to `load`. If `None`,
            `DEFAULT_CONFIGURATION_MODE` is used.
        backend_kind : BackendKind | None, optional
            Backend family to initialize through configuration loading.
        _execution_mode : ExecutionMode | None, optional
            Internal execution mode override forwarded to the execution service.
        _clock_health_checks : bool | None, optional
            Internal flag for backend clock-health checks in parallel execution.

        Examples
        --------
        >>> from qubex.measurement import Measurement
        >>> session = Measurement(
        ...     chip_id="64Q",
        ...     qubits=["Q00", "Q01"],
        ... )
        """
        self._chip_id: Final = chip_id
        self._qubits: Final = list(qubits)
        self._execution_mode: Final[ExecutionMode | None] = _execution_mode
        self._clock_health_checks: Final[bool | None] = _clock_health_checks
        self._system_manager = SystemManager.shared()
        self._context = MeasurementContext(
            system_manager=self._system_manager,
            qubits=self._qubits,
        )
        self._classification_service = MeasurementClassificationService(classifiers={})
        self._amplification_service = MeasurementAmplificationService(
            context=self._context
        )
        self._session_service = MeasurementSessionService(
            system_manager=self._system_manager,
            context=self._context,
        )
        self._execution_service = MeasurementExecutionService(
            context=self._context,
            session_service=self._session_service,
            classifiers=self._classification_service.classifiers,
            execution_mode=self._execution_mode,
            clock_health_checks=self._clock_health_checks,
        )
        if load_configs is None:
            load_configs = self.DEFAULT_LOAD_CONFIGS
        if connect_devices is None:
            connect_devices = self.DEFAULT_CONNECT_DEVICES
        if configuration_mode is None:
            configuration_mode = self.DEFAULT_CONFIGURATION_MODE
        if load_configs:
            self.load(
                config_dir=config_dir,
                params_dir=params_dir,
                configuration_mode=configuration_mode,
                backend_kind=backend_kind,
            )
        if connect_devices:
            self.connect()

    def load(
        self,
        config_dir: Path | str | None,
        params_dir: Path | str | None,
        configuration_mode: ConfigurationMode | None = None,
        backend_kind: BackendKind | None = None,
    ) -> None:
        """
        Load configuration resources into the current session.

        Parameters
        ----------
        config_dir : Path | str | None
            Directory that contains system configuration files.
        params_dir : Path | str | None
            Directory that contains control parameter files.
        configuration_mode : ConfigurationMode | None, optional
            Configuration variant. If `None`, backend defaults are used.
        backend_kind : BackendKind | None, optional
            Backend family used when creating the backend controller.

        Notes
        -----
        This method updates `config_loader`, `experiment_system`, and backend
        runtime dependencies.
        """
        self.session_service.load(
            chip_id=self._chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
            backend_kind=backend_kind,
        )

    def connect(
        self,
        *,
        sync_clocks: bool | None = None,
        parallel: bool | None = None,
    ) -> None:
        """
        Connect backend resources for the active session.

        Parameters
        ----------
        sync_clocks : bool | None, optional
            Whether to synchronize hardware clocks before measurement.
        parallel : bool | None, optional
            Whether backend setup steps should run in parallel where supported.

        Notes
        -----
        This method performs hardware-side connection and may take noticeable
        time depending on backend state.
        """
        self.session_service.connect(sync_clocks=sync_clocks, parallel=parallel)

    def reload(
        self,
        *,
        configuration_mode: ConfigurationMode | None = None,
    ) -> None:
        """
        Reload configuration and reconnect backend resources.

        Parameters
        ----------
        configuration_mode : ConfigurationMode | None, optional
            Configuration variant passed to `load`. If `None`, the previously
            configured variant is reused by the loader/service.

        Notes
        -----
        This method calls `load` using the current loader paths and then
        reconnects hardware resources via `connect`.
        """
        self.load(
            config_dir=self.config_loader.config_path,
            params_dir=self.config_loader.params_path,
            configuration_mode=configuration_mode,
        )
        self.connect()

    @property
    def qubit_labels(self) -> list[str]:
        """Return the configured qubit labels."""
        return self._qubits

    @property
    @deprecated("Use `qubit_labels` instead.")
    def qubits(self) -> list[str]:
        """Return the configured qubit labels."""
        return self._qubits

    @property
    def box_ids(self) -> list[str]:
        """Return backend box identifiers for the active session."""
        return self.context.box_ids

    @property
    def mux_dict(self) -> dict[str, Mux]:
        """Return MUX objects indexed by qubit label."""
        return self.context.mux_dict

    @property
    def system_manager(self) -> SystemManager:
        """Return the shared system manager."""
        return self._system_manager

    @property
    def context(self) -> MeasurementContext:
        """Return the measurement context."""
        return self._context

    @property
    def session_service(self) -> MeasurementSessionService:
        """Return the session lifecycle service."""
        return self._session_service

    @property
    def execution_service(self) -> MeasurementExecutionService:
        """Return the measurement execution service."""
        return self._execution_service

    @property
    def classification_service(self) -> MeasurementClassificationService:
        """Return the classification service."""
        return self._classification_service

    @property
    def amplification_service(self) -> MeasurementAmplificationService:
        """Return the readout amplification service."""
        return self._amplification_service

    @property
    def pulse_factory(self) -> MeasurementPulseFactory:
        """Return a pulse factory bound to current system state."""
        return self.execution_service.pulse_factory

    @property
    def schedule_builder(self) -> MeasurementScheduleBuilder:
        """Return a schedule builder bound to current system state."""
        return self.execution_service.schedule_builder

    @property
    def measurement_config_factory(self) -> MeasurementConfigFactory:
        """Return a measurement-config factory bound to current defaults."""
        return self.execution_service.measurement_config_factory

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
        return self.context.backend_controller

    @property
    def sampling_period(self) -> float:
        """Return sampling period in ns."""
        return self.execution_service.sampling_period

    @property
    def constraint_profile(self) -> MeasurementConstraintProfile:
        """Return backend timing and alignment constraints."""
        return self.execution_service.constraint_profile

    @property
    def measurement_schedule_runner(self) -> MeasurementScheduleRunner:
        """Return the schedule-execution runner."""
        return self.execution_service.measurement_schedule_runner

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

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        """Return currently registered state classifiers."""
        return self.classification_service.classifiers

    def get_awg_frequency(self, target: str) -> float:
        """
        Return the AWG frequency for one target.

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
        Return the intermediate (difference) frequency for one target.

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

    def update_classifiers(self, classifiers: TargetMap[StateClassifier]) -> None:
        """
        Replace registered state classifiers.

        Parameters
        ----------
        classifiers : TargetMap[StateClassifier]
            Classifiers indexed by target label.
        """
        self.classification_service.update_classifiers(classifiers)

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> npt.NDArray:
        """
        Return a combined confusion matrix for the given targets.

        Parameters
        ----------
        targets : Collection[str]
            Target labels included in Kronecker-product order.

        Returns
        -------
        npt.NDArray
            Combined confusion matrix.
        """
        return self.classification_service.get_confusion_matrix(targets)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> npt.NDArray:
        """
        Return the inverse of the combined confusion matrix.

        Parameters
        ----------
        targets : Collection[str]
            Target labels included in Kronecker-product order.

        Returns
        -------
        npt.NDArray
            Inverse combined confusion matrix.
        """
        return self.classification_service.get_inverse_confusion_matrix(targets)

    def is_connected(self) -> bool:
        """
        Return whether backend resources are connected.

        Returns
        -------
        bool
            `True` if connected, otherwise `False`.
        """
        return self.session_service.is_connected()

    def disconnect(self) -> None:
        """
        Disconnect backend resources held by the measurement session.

        Notes
        -----
        This method forwards directly to the session lifecycle service and
        resets runtime connectivity state.
        """
        self.session_service.disconnect()

    def check_link_status(
        self,
        box_list: list[str],
        *,
        parallel: bool | None = None,
    ) -> dict:
        """
        Return link status for the specified backend boxes.

        Parameters
        ----------
        box_list : list[str]
            Backend box identifiers.
        parallel : bool | None, optional
            Whether to query each box in parallel where supported. If `None`,
            the session-level default is used.

        Returns
        -------
        dict
            Backend-specific link status payload keyed by box identifier.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> session.check_link_status(["Q73A", "U10B"])
        """
        return self.session_service.check_link_status(box_list, parallel=parallel)

    def check_clock_status(self, box_list: list[str]) -> dict:
        """
        Return clock status for the specified backend boxes.

        Parameters
        ----------
        box_list : list[str]
            Backend box identifiers.

        Returns
        -------
        dict
            Backend-specific clock status payload keyed by box identifier.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> session.check_clock_status(["Q73A", "U10B"])
        """
        return self.session_service.check_clock_status(box_list)

    def linkup(self, box_list: list[str], noise_threshold: int | None = None) -> None:
        """
        Run link-up and clock synchronization for backend boxes.

        Parameters
        ----------
        box_list : list[str]
            Backend box identifiers.
        noise_threshold : int | None, optional
            Optional threshold used by backend-specific link quality checks.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> session.linkup(["Q73A", "U10B"])
        """
        self.session_service.linkup(box_list, noise_threshold=noise_threshold)

    def relinkup(self, box_list: list[str]) -> None:
        """
        Re-run link-up for already configured backend boxes.

        Parameters
        ----------
        box_list : list[str]
            Backend box identifiers.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> session.relinkup(["Q73A", "U10B"])
        """
        self.session_service.relinkup(box_list)

    @contextmanager
    def modified_frequencies(
        self,
        target_frequencies: dict[str, float],
    ) -> Iterator[None]:
        """
        Temporarily override target frequencies within a context block.

        Parameters
        ----------
        target_frequencies : dict[str, float]
            Frequency overrides in Hz keyed by target label.

        Yields
        ------
        None
            Context where overridden frequencies are active.

        Notes
        -----
        Original frequencies are restored when exiting the context manager.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> with session.modified_frequencies({"Q00": 5.0e9}):
        ...     _ = session.measure(
        ...         {
        ...             "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...             "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ...         }
        ...     )
        """
        with self.session_service.modified_frequencies(target_frequencies):
            yield

    @contextmanager
    def apply_dc_voltages(self, targets: str | Collection[str]) -> Iterator[None]:
        """
        Temporarily apply DC voltages to selected targets.

        Parameters
        ----------
        targets : str | Collection[str]
            Target label or target labels for temporary DC bias application.

        Yields
        ------
        None
            Context where DC voltages are applied.

        Notes
        -----
        DC voltages are removed automatically when exiting the context manager.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> with session.apply_dc_voltages(["Q00", "Q01"]):
        ...     _ = session.measure(
        ...         {
        ...             "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...             "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ...         }
        ...     )
        """
        with self.amplification_service.apply_dc_voltages(targets):
            yield

    async def run_measurement(
        self,
        schedule: MeasurementSchedule,
        *,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        """
        Execute one prepared measurement schedule.

        Parameters
        ----------
        schedule : MeasurementSchedule
            Measurement schedule that includes readout instructions.
        config : MeasurementConfig
            Runtime acquisition configuration.

        Returns
        -------
        MeasurementResult
            Serialized measurement result container.
        """
        return await self.execution_service.run_measurement(
            schedule=schedule,
            config=config,
        )

    async def run_sweep_measurement(
        self,
        schedule: Callable[[SweepValue], MeasurementSchedule],
        *,
        config: MeasurementConfig,
        sweep_values: Sequence[SweepValue],
    ) -> SweepMeasurementResult:
        """
        Execute a pointwise sweep over explicit sweep values.

        Parameters
        ----------
        schedule : Callable[[SweepValue], MeasurementSchedule]
            Factory that builds one schedule from one sweep value.
        config : MeasurementConfig
            Runtime acquisition configuration.
        sweep_values : Sequence[SweepValue]
            Explicit sweep values evaluated in sequence.

        Returns
        -------
        SweepMeasurementResult
            Sweep result with one flattened entry per sweep value.
        """
        return await self.execution_service.run_sweep_measurement(
            schedule,
            config=config,
            sweep_values=sweep_values,
        )

    async def run_ndsweep_measurement(
        self,
        schedule: Callable[[SweepPoint], MeasurementSchedule],
        *,
        config: MeasurementConfig,
        sweep_points: dict[SweepKey, Sequence[SweepValue]],
        sweep_axes: SweepAxes | None = None,
    ) -> NDSweepMeasurementResult:
        """
        Execute an N-dimensional Cartesian-product sweep.

        Parameters
        ----------
        schedule : Callable[[SweepPoint], MeasurementSchedule]
            Factory that builds one schedule from one expanded sweep point.
        config : MeasurementConfig
            Runtime acquisition configuration.
        sweep_points : dict[SweepKey, Sequence[SweepValue]]
            Sweep axes and candidate values for each axis.
        sweep_axes : SweepAxes | None, optional
            Axis order used for Cartesian expansion and index mapping. If
            `None`, dictionary insertion order is used.

        Returns
        -------
        NDSweepMeasurementResult
            N-dimensional sweep result stored as flattened point results with
            shape metadata.
        """
        return await self.execution_service.run_ndsweep_measurement(
            schedule,
            config=config,
            sweep_points=sweep_points,
            sweep_axes=sweep_axes,
        )

    async def measure_noise(
        self,
        targets: Collection[str],
        *,
        duration: float,
    ) -> MeasurementResult:
        """
        Measure readout noise with no control waveform drive.

        Parameters
        ----------
        targets : Collection[str]
            Target labels to capture.
        duration : float
            Capture duration in ns.

        Returns
        -------
        MeasurementResult
            Noise measurement result.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> result = await session.measure_noise(["Q00", "Q01"], duration=2048.0)
        """
        return await self.execution_service.measure_noise(
            targets=targets,
            duration=duration,
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
        plot: bool = False,
        **deprecated_options: Any,
    ) -> MeasureResult:
        """
        Execute one measurement from target waveform mappings.

        Parameters
        ----------
        waveforms : Mapping[str, IQArray]
            Control waveforms keyed by target label. Each waveform is a complex
            I/Q array sampled at `sampling_period` ns.
        n_shots : int | None, optional
            Number of shots.
        shot_interval : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether shot averaging is applied in hardware.
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.
        frequencies : dict[str, float] | None, optional
            Channel-frequency overrides keyed by schedule label.
        readout_amplitudes : dict[str, float] | None, optional
            Readout amplitude overrides keyed by target label.
        readout_duration : float | None, optional
            Readout capture duration in ns.
        readout_pre_margin : float | None, optional
            Margin inserted before readout in ns.
        readout_post_margin : float | None, optional
            Margin inserted after readout in ns.
        readout_ramp_time : float | None, optional
            Readout ramp duration in ns.
        readout_drag_coeff : float | None, optional
            Drag coefficient for ramp shaping.
        readout_ramp_type : RampType | None, optional
            Ramp shape type.
        readout_amplification : bool | None, optional
            Whether to apply readout amplification pulses.
        classification_line_param0 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 0.
        classification_line_param1 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 1.
        plot : bool, optional
            Whether to plot readout waveforms and/or results.
        **deprecated_options : Any
            Deprecated option aliases kept for backward compatibility.

        Returns
        -------
        MeasureResult
            Single measurement result.

        Notes
        -----
        Deprecated options are normalized by the execution service.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> result = session.measure(
        ...     {
        ...         "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...         "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ...     }
        ... )
        """
        return self.execution_service.measure(
            waveforms=waveforms,
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
            time_integration=time_integration,
            state_classification=state_classification,
            classification_line_param0=classification_line_param0,
            classification_line_param1=classification_line_param1,
            plot=plot,
            **deprecated_options,
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
        plot: bool = False,
        save_result: bool = True,
        **deprecated_options: Any,
    ) -> MultipleMeasureResult:
        """
        Execute measurement for a pulse schedule or waveform mapping.

        Parameters
        ----------
        schedule : PulseSchedule | TargetMap[IQArray]
            Pulse schedule or waveform mapping to execute.
        n_shots : int | None, optional
            Number of shots.
        shot_interval : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether shot averaging is applied in hardware.
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.
        frequencies : dict[str, float] | None, optional
            Channel-frequency overrides keyed by schedule label.
        readout_amplitudes : dict[str, float] | None, optional
            Readout amplitude overrides keyed by target label.
        readout_duration : float | None, optional
            Readout capture duration in ns.
        readout_pre_margin : float | None, optional
            Margin inserted before readout in ns.
        readout_post_margin : float | None, optional
            Margin inserted after readout in ns.
        readout_ramp_time : float | None, optional
            Readout ramp duration in ns.
        readout_drag_coeff : float | None, optional
            Drag coefficient for ramp shaping.
        readout_ramp_type : RampType | None, optional
            Ramp shape type.
        readout_amplification : bool | None, optional
            Whether to apply readout amplification pulses.
        final_measurement : bool | None, optional
            Whether to append a final readout measurement.
        classification_line_param0 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 0.
        classification_line_param1 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 1.
        plot : bool, optional
            Whether to plot readout waveforms and/or results.
        save_result : bool, optional
            Whether to save execution outputs into the session result cache.
        **deprecated_options : Any
            Deprecated option aliases kept for backward compatibility.

        Returns
        -------
        MultipleMeasureResult
            Measurement results for all captured readout windows.

        Notes
        -----
        Deprecated options are normalized by the execution service.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> result = session.execute(
        ...     {
        ...         "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...         "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ...     }
        ... )
        """
        return self.execution_service.execute(
            schedule=schedule,
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
            final_measurement=final_measurement,
            time_integration=time_integration,
            state_classification=state_classification,
            classification_line_param0=classification_line_param0,
            classification_line_param1=classification_line_param1,
            plot=plot,
            save_result=save_result,
            **deprecated_options,
        )

    def capture_loopback(
        self,
        schedule: PulseSchedule | TargetMap[IQArray],
        *,
        n_shots: int | None = None,
    ) -> MeasurementResult:
        """
        Capture full-span loopback data from read-in and monitor channels.

        Parameters
        ----------
        schedule : PulseSchedule | TargetMap[IQArray]
            Pulse schedule or waveform mapping to execute.
        n_shots : int | None, optional
            Number of shots.

        Returns
        -------
        MeasurementResult
            Measurement result for loopback capture windows.
        """
        return self.execution_service.capture_loopback(
            schedule=schedule,
            n_shots=n_shots,
        )

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
        Create a measurement config from optional runtime overrides.

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
            Measurement configuration with merged defaults and overrides.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> config = session.create_measurement_config(
        ...     n_shots=1024,
        ...     shot_interval=2048.0,
        ... )
        """
        return self.execution_service.create_measurement_config(
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
        """
        Build a measurement schedule from a pulse schedule and readout options.

        Parameters
        ----------
        pulse_schedule : PulseSchedule
            Pulse schedule to augment with readout instructions.
        frequencies : dict[str, float] | None, optional
            Channel-frequency overrides keyed by schedule label.
        readout_amplitudes : dict[str, float] | None, optional
            Readout amplitude overrides keyed by target label.
        readout_duration : float | None, optional
            Readout capture duration in ns.
        readout_pre_margin : float | None, optional
            Margin inserted before readout in ns.
        readout_post_margin : float | None, optional
            Margin inserted after readout in ns.
        readout_ramp_time : float | None, optional
            Readout ramp duration in ns.
        readout_ramp_type : RampType | None, optional
            Ramp shape type.
        readout_drag_coeff : float | None, optional
            Drag coefficient for ramp shaping.
        readout_amplification : bool | None, optional
            Whether to insert readout amplification pulses.
        final_measurement : bool | None, optional
            Whether to append a final measurement at schedule tail.
        capture_placement : CapturePlacement | None, optional
            Capture-window placement (`pulse_aligned` or `entire_schedule`).
        capture_targets : list[str] | None, optional
            Explicit capture-channel labels for `entire_schedule` placement.
        plot : bool | None, optional
            Whether to plot the generated schedule.

        Returns
        -------
        MeasurementSchedule
            Built measurement schedule ready for execution.

        Examples
        --------
        >>> # `session` is an initialized `Measurement` instance.
        >>> schedule = session.build_measurement_schedule(pulse_schedule)
        """
        return self.execution_service.build_measurement_schedule(
            pulse_schedule=pulse_schedule,
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

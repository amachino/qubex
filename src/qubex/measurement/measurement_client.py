"""Client class for end-to-end measurement workflows."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from qxpulse import PulseSchedule, RampType

from qubex.backend import (
    ConfigLoader,
    ControlParams,
    ExperimentSystem,
    Mux,
    SystemManager,
    Target,
)
from qubex.backend.dc_voltage_controller import dc_voltage
from qubex.backend.quel1 import (
    SAMPLING_PERIOD,
    ExecutionMode,
    Quel1BackendController,
)
from qubex.measurement.measurement_config_factory import MeasurementConfigFactory
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import (
    MeasurementResult,
)
from qubex.typing import ConfigurationMode, IQArray, MeasurementMode, TargetMap

from .classifiers.state_classifier import StateClassifier
from .measurement_backend_manager import MeasurementBackendManager
from .measurement_pulse_factory import MeasurementPulseFactory
from .measurement_result_converter import MeasurementResultConverter
from .measurement_schedule_builder import MeasurementScheduleBuilder
from .measurement_schedule_executor import MeasurementScheduleExecutor
from .models.measure_result import (
    MeasureResult,
    MultipleMeasureResult,
)
from .models.measurement_schedule import MeasurementSchedule

logger = logging.getLogger(__name__)


class MeasurementClient:
    """
    Client class for end-to-end measurement workflows.

    `MeasurementClient` owns the high-level workflow while delegating concrete
    responsibilities to focused collaborators: configuration/backend lifecycle
    (`MeasurementBackendManager`), schedule assembly
    (`MeasurementScheduleBuilder` and `MeasurementPulseFactory`) and delegates
    schedule execution to `MeasurementScheduleExecutor`. It also keeps optional
    state classifiers used during readout post-processing.

    Notes
    -----
    For backward compatibility, `Measurement` is provided as an alias of this
    class.
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
        _execution_mode: ExecutionMode | None = None,
        _clock_health_checks: bool | None = None,
    ):
        """
        Initialize the MeasurementClient.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        qubits : Sequence[str]
            The list of qubit labels.
        config_dir : Path | str, optional
            The configuration directory.
        params_dir : Path | str, optional
            The parameters directory.
        load_configs : bool | None, optional
            Whether to load configurations. If `None`, `DEFAULT_LOAD_CONFIGS`
            is used.
        connect_devices : bool | None, optional
            Whether to connect devices. If `None`,
            `DEFAULT_CONNECT_DEVICES` is used.
        configuration_mode : ConfigurationMode | None, optional
            Configuration mode. If `None`, `DEFAULT_CONFIGURATION_MODE`
            is used.
        _execution_mode : ExecutionMode | None, optional
            Private backend execution mode override used by schedule executor.
        _clock_health_checks : bool | None, optional
            Private flag to enable clock-health checks in parallel execution.

        Examples
        --------
        >>> from qubex.measurement import MeasurementClient
        >>> cli = MeasurementClient(
        ...     chip_id="64Q",
        ...     qubits=["Q00", "Q01"],
        ... )
        """
        self._chip_id: Final = chip_id
        self._qubits: Final = list(qubits)
        self._execution_mode: Final[ExecutionMode | None] = _execution_mode
        self._clock_health_checks: Final[bool | None] = _clock_health_checks
        self._classifiers: TargetMap[StateClassifier] = {}
        self._system_manager = SystemManager.shared()
        self._backend_manager = MeasurementBackendManager(
            system_manager=self._system_manager,
            qubits=self._qubits,
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
            )
        if connect_devices:
            self.connect()

    def load(
        self,
        config_dir: Path | str | None,
        params_dir: Path | str | None,
        configuration_mode: ConfigurationMode | None = None,
    ) -> None:
        """
        Load the measurement settings.

        Parameters
        ----------
        config_dir : Path | str | None
            The configuration directory.
        params_dir : Path | str | None
            The parameters directory.
        configuration_mode : ConfigurationMode, optional
            The configuration mode, by default "ge-cr-cr".
        """
        self.backend_manager.load(
            chip_id=self._chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
        )

    def connect(
        self,
        *,
        sync_clocks: bool | None = None,
        parallel: bool | None = None,
    ) -> None:
        """
        Connect to the devices.

        Parameters
        ----------
        sync_clocks : bool | None, optional
            Whether to resync clocks, by default True.
        parallel : bool | None, optional
            Whether to fetch backend settings in parallel.
        """
        self.backend_manager.connect(sync_clocks=sync_clocks, parallel=parallel)

    def reload(
        self,
        *,
        configuration_mode: ConfigurationMode | None = None,
    ) -> None:
        """Reload the measuremnt settings."""
        self.load(
            config_dir=self.config_loader.config_path,
            params_dir=self.config_loader.params_path,
            configuration_mode=configuration_mode,
        )
        self.connect()

    @property
    def qubits(self) -> list[str]:
        """Get the list of qubit labels."""
        return self._qubits

    @property
    def box_ids(self) -> list[str]:
        """Get the list of box IDs."""
        return self.backend_manager.box_ids

    @property
    def mux_dict(self) -> dict[str, Mux]:
        """Get a dictionary of Mux objects indexed by qubit labels."""
        return self.backend_manager.mux_dict

    @property
    def system_manager(self) -> SystemManager:
        """Get the state manager."""
        return self._system_manager

    @property
    def backend_manager(self) -> MeasurementBackendManager:
        """Return the backend/config manager."""
        return self._backend_manager

    @property
    def pulse_factory(self) -> MeasurementPulseFactory:
        """Create a pulse factory from current system state."""
        return MeasurementPulseFactory(
            control_params=self.control_params,
            mux_dict=self.mux_dict,
        )

    @property
    def schedule_builder(self) -> MeasurementScheduleBuilder:
        """Create a measurement schedule builder from current system state."""
        return MeasurementScheduleBuilder(
            control_params=self.control_params,
            pulse_factory=self.pulse_factory,
            targets=self.targets,
            mux_dict=self.mux_dict,
            sampling_period=self.sampling_period,
        )

    @property
    def measurement_config_factory(self) -> MeasurementConfigFactory:
        """Create a measurement config factory from current system state."""
        return MeasurementConfigFactory(
            experiment_system=self.experiment_system,
        )

    @property
    def config_loader(self) -> ConfigLoader:
        """Get the configuration loader."""
        return self.backend_manager.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        return self.backend_manager.experiment_system

    @property
    def backend_controller(self) -> Quel1BackendController:
        """Get the backend controller."""
        return self.backend_manager.backend_controller

    @property
    def sampling_period(self) -> float:
        """Resolve sampling period (ns) from backend-controller contract."""
        try:
            sampling_period = getattr(
                self.backend_controller, "DEFAULT_SAMPLING_PERIOD", None
            )
        except Exception:
            return SAMPLING_PERIOD
        if isinstance(sampling_period, (int, float)):
            return float(sampling_period)
        return SAMPLING_PERIOD

    @property
    def measurement_schedule_executor(self) -> MeasurementScheduleExecutor:
        """Return executor implementation used by schedule execution APIs."""
        return MeasurementScheduleExecutor.create_default(
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

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        """Get the state classifiers."""
        return self._classifiers

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

    def update_classifiers(self, classifiers: TargetMap[StateClassifier]) -> None:
        """Update the state classifiers."""
        for target, classifier in classifiers.items():
            self._classifiers[target] = classifier  # type: ignore

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> npt.NDArray:
        """
        Return the combined confusion matrix for targets.

        Parameters
        ----------
        targets : Collection[str]
            Target labels to include.

        Returns
        -------
        npt.NDArray
            Kronecker-product confusion matrix.
        """
        targets = list(targets)
        confusion_matrices = []
        for target in targets:
            cm = self.classifiers[target].confusion_matrix
            n_shots = cm[0].sum()
            confusion_matrices.append(cm / n_shots)
        return reduce(np.kron, confusion_matrices)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> npt.NDArray:
        """
        Return the inverse combined confusion matrix.

        Parameters
        ----------
        targets : Collection[str]
            Target labels to include.

        Returns
        -------
        npt.NDArray
            Inverse confusion matrix.
        """
        targets = list(targets)
        confusion_matrix = self.get_confusion_matrix(targets)
        return np.linalg.inv(confusion_matrix)

    def is_connected(self) -> bool:
        """
        Check if the measurement system is connected to the devices.

        Returns
        -------
        bool
            True if connected, False otherwise.
        """
        return self.backend_manager.is_connected()

    def disconnect(self) -> None:
        """Disconnect backend resources held by the measurement backend."""
        self.backend_manager.disconnect()

    def check_link_status(self, box_list: list[str]) -> dict:
        """
        Check the link status of the boxes.

        Parameters
        ----------
        box_list : list[str]
            The list of box IDs.

        Returns
        -------
        dict
            The link status of the boxes.

        Examples
        --------
        >>> cli.check_link_status(["Q73A", "U10B"])
        """
        return self.backend_manager.check_link_status(box_list)

    def check_clock_status(self, box_list: list[str]) -> dict:
        """
        Check the clock status of the boxes.

        Parameters
        ----------
        box_list : list[str]
            The list of box IDs.

        Returns
        -------
        dict
            The clock status of the boxes.

        Examples
        --------
        >>> cli.check_clock_status(["Q73A", "U10B"])
        """
        return self.backend_manager.check_clock_status(box_list)

    def linkup(self, box_list: list[str], noise_threshold: int | None = None) -> None:
        """
        Link up the boxes and synchronize the clocks.

        Parameters
        ----------
        box_list : list[str]
            The list of box IDs.

        Examples
        --------
        >>> cli.linkup(["Q73A", "U10B"])
        """
        self.backend_manager.linkup(box_list, noise_threshold=noise_threshold)

    def relinkup(self, box_list: list[str]) -> None:
        """
        Relink up the boxes and synchronize the clocks.

        Parameters
        ----------
        box_list : list[str]
            The list of box IDs.

        Examples
        --------
        >>> cli.relinkup(["Q73A", "U10B"])
        """
        self.backend_manager.relinkup(box_list)

    @contextmanager
    def modified_frequencies(
        self,
        target_frequencies: dict[str, float],
    ) -> Iterator[None]:
        """
        Temporarily modify the target frequencies.

        Parameters
        ----------
        target_frequencies : dict[str, float]
            The target frequencies to be modified.

        Examples
        --------
        >>> with cli.modified_frequencies({"Q00": 5.0}):
        ...     result = cli.measure({
        ...         "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...         "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ...     })
        """
        with self.backend_manager.modified_frequencies(target_frequencies):
            yield

    @contextmanager
    def apply_dc_voltages(self, targets: str | Collection[str]) -> Iterator[None]:
        """
        Temporarily apply DC voltages to the specified targets.

        Parameters
        ----------
        targets : Collection[str]
            The list of target names.

        Examples
        --------
        >>> with cli.apply_dc_voltages(["Q00", "Q01"]):
        ...     result = cli.measure({
        ...         "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...         "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ...     })
        """
        if isinstance(targets, str):
            targets = [targets]
        qubits = [Target.qubit_label(target) for target in targets]
        muxes = {
            self.experiment_system.get_mux_by_qubit(qubit).index for qubit in qubits
        }
        voltages = {mux + 1: self.control_params.get_dc_voltage(mux) for mux in muxes}
        with dc_voltage(voltages):
            yield

    def execute_measurement_schedule(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
    ) -> MeasurementResult:
        """
        Run the measurement with the given schedule and configuration.

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
        result = self.measurement_schedule_executor.execute(
            schedule=schedule,
            config=config,
        )
        return result

    def measure_noise(
        self,
        targets: Collection[str],
        *,
        duration: float,
    ) -> MeasureResult:
        """
        Measure the readout noise.

        Parameters
        ----------
        targets : Collection[str]
            The list of target names.
        duration : float, optional
            The duration in ns.

        Returns
        -------
        MeasureResult
            The measurement results.

        Examples
        --------
        >>> result = cli.measure_noise()
        """
        return self.measure(
            waveforms={target: np.zeros(0) for target in targets},
            mode="avg",
            shots=1,
            readout_duration=duration,
            readout_amplitudes=dict.fromkeys(targets, 0),
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
        add_pump_pulses: bool = False,
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool = False,
        enable_dsp_classification: bool = False,
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
            Waveforms are complex I/Q arrays with the sampling period of 2 ns.
        mode : MeasurementMode, optional
            The measurement mode, by default "single".
            - "single": Measure once.
            - "avg": Measure multiple times and average the results.
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
        add_pump_pulses : bool, optional
            Whether to add pump pulses, by default False.

        Returns
        -------
        MeasureResult
            The measurement results.

        Examples
        --------
        >>> result = cli.measure({
        ...     "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...     "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ... })
        """
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
        add_pump_pulses: bool = False,
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool = False,
        enable_dsp_classification: bool = False,
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
            The measurement mode, by default "single".
            - "single": Measure once.
            - "avg": Measure multiple times and average the results.
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
            Whether to add the last measurement, by default False.
        add_pump_pulses : bool, optional
            Whether to add pump pulses, by default False.
        enable_dsp_sum : bool, optional
            Whether to enable DSP summation, by default False.
        enable_dsp_classification : bool, optional
            Whether to enable DSP classification, by default False.
        plot : bool, optional
            Whether to plot the results, by default False.

        Returns
        -------
        MultipleMeasureResult
            The measurement results.
        """
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
        if rawdata_dir is not None:
            result.save(rawdata_dir)

        return MeasurementResultConverter.to_multiple_measure_result(
            result,
            config=self.backend_controller.box_config,
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

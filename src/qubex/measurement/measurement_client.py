"""Client class for end-to-end measurement workflows."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property, reduce
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from qubex.backend import (
    BackendExecutor,
    ConfigLoader,
    ControlParams,
    DeviceController,
    ExperimentSystem,
    Mux,
    QuelBackendExecutor,
    RawResult,
    SystemManager,
    Target,
)
from qubex.backend.dc_voltage_controller import dc_voltage
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import (
    MeasurementResult,
)
from qubex.pulse import PulseSchedule, RampType
from qubex.typing import IQArray, TargetMap

from .classifiers.state_classifier import StateClassifier
from .measurement_backend_adapter import (
    MeasurementBackendAdapter,
    QuelMeasurementBackendAdapter,
)
from .measurement_device_manager import MeasurementDeviceManager
from .measurement_pulse_factory import MeasurementPulseFactory
from .measurement_schedule_builder import MeasurementScheduleBuilder
from .models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)
from .models.measurement_schedule import MeasurementSchedule

logger = logging.getLogger(__name__)

DEFAULT_SHOTS: Final = 1024
DEFAULT_INTERVAL: Final = 150 * 1024  # ns
DEFAULT_READOUT_DURATION: Final = 384  # ns
DEFAULT_READOUT_RAMPTIME: Final = 32  # ns
DEFAULT_READOUT_PRE_MARGIN: Final = 32  # ns
DEFAULT_READOUT_POST_MARGIN: Final = 128  # ns


class MeasurementClient:
    """
    Client class for end-to-end measurement workflows.

    `MeasurementClient` owns the high-level workflow while delegating concrete
    responsibilities to focused collaborators: configuration/device lifecycle
    (`MeasurementDeviceManager`), schedule assembly
    (`MeasurementScheduleBuilder` and `MeasurementPulseFactory`), and backend
    execution (`DeviceExecutor`/`QuelDeviceExecutor`). It also keeps optional
    state classifiers used during readout post-processing.

    Notes
    -----
    For backward compatibility, `Measurement` is provided as an alias of this
    class.
    """

    def __init__(
        self,
        *,
        chip_id: str,
        qubits: Collection[str],
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        load_configs: bool = True,
        connect_devices: bool = False,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
        device_executor: BackendExecutor | None = None,
        measurement_backend_adapter: MeasurementBackendAdapter | None = None,
    ):
        """
        Initialize the MeasurementClient.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        qubits : Sequence[str]
            The list of qubit labels.
        config_dir : str, optional
            The configuration directory.
        params_dir : str, optional
            The parameters directory.
        load_configs : bool, optional
            Whether to load the configurations, by default True.
        connect_devices : bool, optional
            Whether to connect the devices, by default False.
        configuration_mode : Literal["ge-ef-cr", "ge-cr-cr"], optional
            The configuration mode, by default "ge-cr-cr".

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
        self._classifiers: TargetMap[StateClassifier] = {}
        self._system_manager = SystemManager.shared()
        self._device_executor = device_executor
        self._measurement_backend_adapter = measurement_backend_adapter
        self._device_manager = MeasurementDeviceManager(
            system_manager=self._system_manager,
            qubits=self._qubits,
        )
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
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] | None = None,
    ) -> None:
        """
        Load the measurement settings.

        Parameters
        ----------
        config_dir : Path | str | None
            The configuration directory.
        params_dir : Path | str | None
            The parameters directory.
        configuration_mode : Literal["ge-ef-cr", "ge-cr-cr"], optional
            The configuration mode, by default "ge-cr-cr".
        """
        self.device_manager.load(
            chip_id=self._chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
        )

    def connect(
        self,
        *,
        sync_clocks: bool = True,
    ) -> None:
        """Connect to the devices."""
        self.device_manager.connect(sync_clocks=sync_clocks)

    def reload(
        self,
        *,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] | None = None,
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

    @cached_property
    def box_ids(self) -> list[str]:
        """Get the list of box IDs."""
        return self.device_manager.box_ids

    @cached_property
    def mux_dict(self) -> dict[str, Mux]:
        """Get a dictionary of Mux objects indexed by qubit labels."""
        return self.device_manager.mux_dict

    @property
    def system_manager(self) -> SystemManager:
        """Get the state manager."""
        return self._system_manager

    @property
    def device_manager(self) -> MeasurementDeviceManager:
        """Return the device/config manager."""
        return self._device_manager

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
        )

    @property
    def config_loader(self) -> ConfigLoader:
        """Get the configuration loader."""
        return self.device_manager.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        return self.device_manager.experiment_system

    @property
    def device_controller(self) -> DeviceController:
        """Get the device controller."""
        return self.device_manager.device_controller

    @property
    def device_executor(self) -> BackendExecutor:
        """Return the device executor implementation."""
        if self._device_executor is not None:
            return self._device_executor
        return QuelBackendExecutor(
            device_controller=self.device_controller,
        )

    @property
    def measurement_backend_adapter(self) -> MeasurementBackendAdapter:
        """Return schedule-to-backend adapter implementation."""
        if self._measurement_backend_adapter is not None:
            return self._measurement_backend_adapter
        return QuelMeasurementBackendAdapter(
            device_controller=self.device_controller,
            experiment_system=self.experiment_system,
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
        return self.device_manager.is_connected()

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
        return self.device_manager.check_link_status(box_list)

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
        return self.device_manager.check_clock_status(box_list)

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
        self.device_manager.linkup(box_list, noise_threshold=noise_threshold)

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
        self.device_manager.relinkup(box_list)

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
        with self.device_manager.modified_frequencies(target_frequencies):
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

    def run(
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
        self.measurement_backend_adapter.validate_schedule(schedule)

        measure_mode = MeasureMode(config.mode)
        request = self.measurement_backend_adapter.build_execution_request(
            schedule=schedule,
            config=config,
        )
        backend_result = self.device_executor.execute(
            request=request,
        )

        result = self._create_measurement_result(
            backend_result=backend_result,
            measure_mode=measure_mode,
            shots=config.shots,
            measurement_config=config,
        )

        rawdata_dir = self.system_manager.rawdata_dir
        if rawdata_dir is not None:
            result.save(data_dir=rawdata_dir)

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
        mode: Literal["single", "avg"] = "avg",
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
        mode : Literal["single", "avg"], optional
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
        mode: Literal["single", "avg"] = "avg",
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
    ) -> MultipleMeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        schedule : PulseSchedule | TargetMap[IQArray]
            The pulse schedule or control waveforms.
        mode : Literal["single", "avg"], optional
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

        run_config = MeasurementConfig.from_execute_args(
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
            enable_dsp_demodulation=enable_dsp_demodulation,
            enable_dsp_sum=enable_dsp_sum,
            enable_dsp_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )

        measurement_schedule = self._build_measurement_schedule(
            schedule=schedule,
            readout_amplitudes=run_config.readout.readout_amplitudes,
            readout_duration=run_config.readout.readout_duration,
            readout_pre_margin=run_config.readout.readout_pre_margin,
            readout_post_margin=run_config.readout.readout_post_margin,
            readout_ramptime=run_config.readout.readout_ramptime,
            readout_drag_coeff=run_config.readout.readout_drag_coeff,
            readout_ramp_type=run_config.readout.readout_ramp_type,
            add_last_measurement=add_last_measurement,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )

        result = self.run(
            schedule=measurement_schedule,
            config=run_config,
        )
        return self._to_multiple_measure_result(result)

    def _build_measurement_schedule(
        self,
        schedule: PulseSchedule,
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
        built_schedule = self.schedule_builder.build(
            schedule=schedule,
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
        return built_schedule

    def _create_measurement_result(
        self,
        backend_result: RawResult,
        measure_mode: MeasureMode,
        shots: int,
        measurement_config: MeasurementConfig | None = None,
    ) -> MeasurementResult:
        label_slice = slice(1, None)  # remove the resonator prefix "R"
        norm_factor = 2 ** (-32)  # normalization factor for 32-bit data

        iq_data = {}
        for target, iqs in sorted(backend_result.data.items()):
            sideband = self.experiment_system.get_target(target).sideband
            if sideband == "L":
                iq_data[target] = [np.conjugate(iq) for iq in iqs]
            else:
                iq_data[target] = iqs

        measure_data = defaultdict(list)
        if measure_mode == MeasureMode.SINGLE:
            for target, iqs in iq_data.items():
                qubit = target[label_slice]
                for idx, iq in enumerate(iqs):
                    if idx == 0:
                        # skip the first extra capture
                        continue
                    measure_data[qubit].append(iq * norm_factor)
        elif measure_mode == MeasureMode.AVG:
            for target, iqs in iq_data.items():
                qubit = target[label_slice]
                for idx, iq in enumerate(iqs):
                    if idx == 0:
                        # skip the first extra capture
                        continue
                    measure_data[qubit].append(iq.squeeze() * norm_factor / shots)
        else:
            raise ValueError(f"Invalid measure mode: {measure_mode}")

        return MeasurementResult(
            mode=measure_mode.value,
            data=dict(measure_data),
            device_config=self.device_controller.box_config,
            measurement_config=(
                measurement_config.to_dict() if measurement_config is not None else {}
            ),
        )

    def _to_multiple_measure_result(
        self,
        result: MeasurementResult,
        config: dict[str, object] | None = None,
    ) -> MultipleMeasureResult:
        """Convert canonical `MeasurementResult` to legacy multiple result."""
        measure_data = defaultdict(list)
        for qubit, captures in result.data.items():
            for raw in captures:
                measure_data[qubit].append(
                    MeasureData(
                        target=qubit,
                        mode=result.measure_mode,
                        raw=np.asarray(raw),
                        classifier=self.classifiers.get(qubit),
                    )
                )

        return MultipleMeasureResult(
            mode=result.measure_mode,
            data=dict(measure_data),
            config=self.device_controller.box_config if config is None else config,
        )

    def _create_multiple_measure_result(
        self,
        backend_result: RawResult,
        measure_mode: MeasureMode,
        shots: int,
    ) -> MultipleMeasureResult:
        """Create legacy `MultipleMeasureResult` from backend output."""
        result = self._create_measurement_result(
            backend_result=backend_result,
            measure_mode=measure_mode,
            shots=shots,
        )
        return self._to_multiple_measure_result(result)

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from contextlib import contextmanager
from functools import cached_property, reduce
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from qubex.backend import (
    SAMPLING_PERIOD,
    ConfigLoader,
    ControlParams,
    DeviceController,
    ExperimentSystem,
    Mux,
    SystemManager,
    Target,
)
from qubex.backend.dc_voltage_controller import dc_voltage
from qubex.backend.quel_instrument_executor import (
    EXTRA_SUM_SECTION_LENGTH,
    WORD_DURATION,
    WORD_LENGTH,
    QuelInstrumentExecutor,
)
from qubex.pulse import Blank, FlatTop, PulseArray, PulseSchedule, RampType
from qubex.typing import IQArray, TargetMap

from .classifiers import StateClassifier
from .models import (
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)
from .models.capture_schedule import Capture, CaptureSchedule

logger = logging.getLogger(__name__)

DEFAULT_SHOTS: Final = 1024
DEFAULT_INTERVAL: Final = 150 * 1024  # ns
DEFAULT_READOUT_DURATION: Final = 384  # ns
DEFAULT_READOUT_RAMPTIME: Final = 32  # ns
DEFAULT_READOUT_PRE_MARGIN: Final = 32  # ns
DEFAULT_READOUT_POST_MARGIN: Final = 128  # ns


class Measurement:
    def __init__(
        self,
        *,
        chip_id: str,
        qubits: Collection[str],
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        load_configs: bool = True,
        connect_devices: bool = True,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
    ):
        """
        Initialize the Measurement.

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
            Whether to connect the devices, by default True.
        configuration_mode : Literal["ge-ef-cr", "ge-cr-cr"], optional
            The configuration mode, by default "ge-cr-cr".

        Examples
        --------
        >>> from qubex import Measurement
        >>> meas = Measurement(
        ...     chip_id="64Q",
        ...     qubits=["Q00", "Q01"],
        ... )
        """
        self._chip_id: Final = chip_id
        self._qubits: Final = list(qubits)
        self._classifiers: TargetMap[StateClassifier] = {}
        self._instrument_executor: QuelInstrumentExecutor | None = None
        self._system_manager = SystemManager.shared()
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
    ):
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
        self.system_manager.load(
            chip_id=self._chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
        )
        self.system_manager.load_skew_file(self.box_ids)

    def connect(
        self,
        *,
        sync_clocks: bool = True,
    ):
        """Connect to the devices."""
        if len(self.box_ids) == 0:
            logger.warning("No boxes are selected. Please check the configuration.")
            return
        self.device_controller.connect(self.box_ids)
        self.system_manager.pull(self.box_ids)
        if sync_clocks:
            self.device_controller.resync_clocks(self.box_ids)

    def reload(
        self,
        *,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] | None = None,
    ):
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
        boxes = self.experiment_system.get_boxes_for_qubits(self._qubits)
        return [box.id for box in boxes]

    @cached_property
    def mux_dict(self) -> dict[str, Mux]:
        """Get a dictionary of Mux objects indexed by qubit labels."""
        return {
            qubit: self.experiment_system.get_mux_by_qubit(qubit)
            for qubit in self._qubits
        }

    @property
    def system_manager(self) -> SystemManager:
        """Get the state manager."""
        return self._system_manager

    @property
    def instrument_executor(self) -> QuelInstrumentExecutor:
        """Get the instrument executor."""
        if self._instrument_executor is None:
            self._instrument_executor = QuelInstrumentExecutor(
                system_manager=self.system_manager,
                device_controller=self.device_controller,
                experiment_system=self.experiment_system,
                classifiers=self._classifiers,
            )
        return self._instrument_executor

    @property
    def config_loader(self) -> ConfigLoader:
        """Get the configuration loader."""
        return self._system_manager.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        return self._system_manager.experiment_system

    @property
    def device_controller(self) -> DeviceController:
        """Get the device controller."""
        return self._system_manager.device_controller

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

    def update_classifiers(self, classifiers: TargetMap[StateClassifier]):
        """Update the state classifiers."""
        for target, classifier in classifiers.items():
            self._classifiers[target] = classifier  # type: ignore

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> npt.NDArray:
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
        return self.device_controller.is_connected

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
        >>> meas.check_link_status(["Q73A", "U10B"])
        """
        link_statuses = {
            box: self.device_controller.link_status(box) for box in box_list
        }
        is_linkedup = all([all(status.values()) for status in link_statuses.values()])
        return {
            "status": is_linkedup,
            "links": link_statuses,
        }

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
        >>> meas.check_clock_status(["Q73A", "U10B"])
        """
        clocks = self.device_controller.read_clocks(box_list)
        clock_statuses = {
            box: clock
            for box, clock in zip(
                box_list,
                clocks,
                strict=True,
            )
        }
        is_synced = self.device_controller.check_clocks(box_list)
        return {
            "status": is_synced,
            "clocks": clock_statuses,
        }

    def linkup(self, box_list: list[str], noise_threshold: int | None = None):
        """
        Link up the boxes and synchronize the clocks.

        Parameters
        ----------
        box_list : list[str]
            The list of box IDs.

        Examples
        --------
        >>> meas.linkup(["Q73A", "U10B"])
        """
        self.device_controller.linkup_boxes(box_list, noise_threshold=noise_threshold)
        self.device_controller.sync_clocks(box_list)

    def relinkup(self, box_list: list[str]):
        """
        Relink up the boxes and synchronize the clocks.

        Parameters
        ----------
        box_list : list[str]
            The list of box IDs.

        Examples
        --------
        >>> meas.relinkup(["Q73A", "U10B"])
        """
        self.device_controller.relinkup_boxes(box_list)
        self.device_controller.sync_clocks(box_list)

    @contextmanager
    def modified_frequencies(self, target_frequencies: dict[str, float]):
        """
        Temporarily modify the target frequencies.

        Parameters
        ----------
        target_frequencies : dict[str, float]
            The target frequencies to be modified.

        Examples
        --------
        >>> with meas.modified_frequencies({"Q00": 5.0}):
        ...     result = meas.measure({
        ...         "Q00": [0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j],
        ...         "Q01": [0.2 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j],
        ...     })
        """
        if target_frequencies is None:
            yield
        else:
            with self.system_manager.modified_frequencies(target_frequencies):
                yield

    @contextmanager
    def apply_dc_voltages(self, targets: str | Collection[str]):
        """
        Temporarily apply DC voltages to the specified targets.

        Parameters
        ----------
        targets : Collection[str]
            The list of target names.

        Examples
        --------
        >>> with meas.apply_dc_voltages(["Q00", "Q01"]):
        ...     result = meas.measure({
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

    def measure_noise(
        self,
        targets: Collection[str],
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
        >>> result = meas.measure_noise()
        """
        return self.measure(
            waveforms={target: np.zeros(0) for target in targets},
            mode="avg",
            shots=1,
            readout_duration=duration,
            readout_amplitudes={target: 0 for target in targets},
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
        >>> result = meas.measure({
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

        if not schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")

        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

        measure_mode = MeasureMode(mode)

        if add_last_measurement:
            self._add_readout_pulses(
                schedule=schedule,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                readout_ramptime=readout_ramptime,
                readout_drag_coeff=readout_drag_coeff,
                readout_ramp_type=readout_ramp_type,
            )

        readout_targets = [
            label for label in schedule.labels if self.targets[label].is_read
        ]
        if not readout_targets:
            raise ValueError("No readout targets in the pulse schedule.")

        self.instrument_executor.pad_schedule_for_capture(schedule)

        readout_ranges = schedule.get_pulse_ranges(readout_targets)
        if add_pump_pulses:
            self._add_pump_pulses(
                schedule=schedule,
                readout_ranges=readout_ranges,
                readout_pre_margin=readout_pre_margin,
                readout_ramptime=readout_ramptime,
                readout_ramp_type=readout_ramp_type,
            )

        if not schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")

        if plot:
            schedule.plot()

        capture_schedule = self._create_capture_schedule(
            schedule=schedule,
            readout_ranges=readout_ranges,
        )

        return self.instrument_executor.execute(
            schedule=schedule,
            capture_schedule=capture_schedule,
            measure_mode=measure_mode,
            shots=shots,
            interval=interval,
            enable_dsp_demodulation=enable_dsp_demodulation,
            enable_dsp_sum=enable_dsp_sum,
            enable_dsp_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )

    def _add_readout_pulses(
        self,
        *,
        schedule: PulseSchedule,
        readout_amplitudes: dict[str, float] | None,
        readout_duration: float | None,
        readout_pre_margin: float | None,
        readout_post_margin: float | None,
        readout_ramptime: float | None,
        readout_drag_coeff: float | None,
        readout_ramp_type: RampType | None,
    ) -> None:
        if readout_amplitudes is None:
            readout_amplitudes = self.control_params.readout_amplitude
        if readout_duration is None:
            readout_duration = DEFAULT_READOUT_DURATION
        if readout_pre_margin is None:
            readout_pre_margin = DEFAULT_READOUT_PRE_MARGIN
        if readout_post_margin is None:
            readout_post_margin = DEFAULT_READOUT_POST_MARGIN

        readout_targets = list(
            {
                Target.read_label(label)
                for label in schedule.labels
                if not self.targets[label].is_pump
            }
        )
        for target in readout_targets:
            schedule.add(
                target,
                self.readout_pulse(
                    target=target,
                    duration=readout_duration,
                    amplitude=readout_amplitudes.get(target),
                    ramptime=readout_ramptime,
                    type=readout_ramp_type,
                    drag_coeff=readout_drag_coeff,
                    pre_margin=readout_pre_margin,
                    post_margin=readout_post_margin,
                ),
            )

    def _add_pump_pulses(
        self,
        *,
        schedule: PulseSchedule,
        readout_ranges: dict[str, list[range]],
        readout_pre_margin: float | None,
        readout_ramptime: float | None,
        readout_ramp_type: RampType | None,
    ) -> None:
        if readout_pre_margin is None:
            readout_pre_margin = DEFAULT_READOUT_PRE_MARGIN

        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            mux = self.mux_dict[Target.qubit_label(target)]
            for i in range(len(ranges)):
                current_range = ranges[i]

                if i == 0:
                    blank_duration = current_range.start * SAMPLING_PERIOD
                else:
                    prev_range = ranges[i - 1]
                    blank_duration = (
                        current_range.start - prev_range.stop
                    ) * SAMPLING_PERIOD

                blank_duration -= readout_pre_margin

                pump_duration = (
                    current_range.stop - current_range.start
                ) * SAMPLING_PERIOD + readout_pre_margin

                pump_amplitude = self.control_params.get_pump_amplitude(mux.index)

                schedule.add(
                    mux.label,
                    PulseArray(
                        [
                            Blank(blank_duration),
                            self.pump_pulse(
                                target=target,
                                duration=pump_duration,
                                amplitude=pump_amplitude,
                                ramptime=readout_ramptime,
                                type=readout_ramp_type,
                            ),
                        ]
                    ),
                )

    def _create_capture_schedule(
        self,
        *,
        schedule: PulseSchedule,
        readout_ranges: dict[str, list[range]],
        capture_delays: dict[int, int] | None = None,
    ) -> CaptureSchedule:
        if capture_delays is None:
            capture_delays = self.control_params.capture_delay_word

        captures: list[Capture] = []
        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            if ranges[0].start % WORD_LENGTH != 0:
                raise ValueError(
                    f"Capture range should start at a multiple of 4 samples ({WORD_DURATION} ns)."
                )

            mux = self.mux_dict[Target.qubit_label(target)]
            capture_delay_word = capture_delays.get(mux.index)
            if capture_delay_word is None:
                capture_delay_word = 0
            delay_samples = capture_delay_word * WORD_LENGTH
            delay_time = delay_samples * SAMPLING_PERIOD

            captures.append(
                Capture(
                    channels=[target],
                    start_time=delay_time,
                    duration=EXTRA_SUM_SECTION_LENGTH * SAMPLING_PERIOD,
                )
            )

            for i, current_range in enumerate(ranges):
                capture_range_length = len(current_range)
                if current_range.start % WORD_LENGTH != 0:
                    raise ValueError(
                        f"Capture range should start at a multiple of 4 samples ({WORD_DURATION} ns)."
                    )
                if capture_range_length % WORD_LENGTH != 0:
                    raise ValueError(
                        f"Capture duration should be a multiple of 4 samples ({WORD_DURATION} ns)."
                    )

                if i < len(ranges) - 1:
                    next_range = ranges[i + 1]
                    post_blank_length = next_range.start - current_range.stop
                    if post_blank_length < WORD_LENGTH:
                        raise ValueError(
                            f"Readout pulses must have at least {WORD_DURATION} ns post-blank time."
                        )
                    if post_blank_length % WORD_LENGTH != 0:
                        raise ValueError(
                            f"Post-blank time should be a multiple of 4 samples ({WORD_DURATION} ns)."
                        )
                else:
                    last_post_blank_length = schedule.length - current_range.stop
                    if last_post_blank_length < 0:
                        raise ValueError("Invalid capture range length.")

                captures.append(
                    Capture(
                        channels=[target],
                        start_time=(current_range.start + delay_samples)
                        * SAMPLING_PERIOD,
                        duration=capture_range_length * SAMPLING_PERIOD,
                    )
                )

        return CaptureSchedule(captures=captures)

    def readout_pulse(
        self,
        target: str,
        *,
        duration: float | None = None,
        amplitude: float | None = None,
        ramptime: float | None = None,
        type: RampType | None = None,
        drag_coeff: float | None = None,
        pre_margin: float | None = None,
        post_margin: float | None = None,
    ) -> PulseArray:
        qubit = Target.qubit_label(target)
        if duration is None:
            duration = DEFAULT_READOUT_DURATION
        if amplitude is None:
            amplitude = self.control_params.get_readout_amplitude(qubit)
        if ramptime is None:
            ramptime = DEFAULT_READOUT_RAMPTIME
        if type is None:
            type = "RaisedCosine"
        if drag_coeff is None:
            drag_coeff = 0.0
        if pre_margin is None:
            pre_margin = DEFAULT_READOUT_PRE_MARGIN
        if post_margin is None:
            post_margin = DEFAULT_READOUT_POST_MARGIN
        pulse = FlatTop(
            duration=duration,
            amplitude=amplitude,
            tau=ramptime,
            beta=drag_coeff,
            type=type,
        )
        return PulseArray(
            [
                Blank(pre_margin),
                pulse.padded(
                    total_duration=duration + post_margin,
                    pad_side="right",
                ),
            ]
        )

    def pump_pulse(
        self,
        target: str,
        duration: float | None = None,
        amplitude: float | None = None,
        ramptime: float | None = None,
        type: RampType | None = None,
    ) -> FlatTop:
        qubit = Target.qubit_label(target)
        mux = self.mux_dict[qubit]
        if duration is None:
            duration = DEFAULT_READOUT_DURATION
        if amplitude is None:
            amplitude = self.control_params.get_pump_amplitude(mux.index)
        if ramptime is None:
            ramptime = DEFAULT_READOUT_RAMPTIME
        if type is None:
            type = "RaisedCosine"
        return FlatTop(
            duration=duration,
            amplitude=amplitude,
            tau=ramptime,
            type=type,
        )

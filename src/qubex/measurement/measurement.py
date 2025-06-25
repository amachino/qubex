from __future__ import annotations

import logging
import math
from collections import defaultdict
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import Collection, Final, Literal

import numpy as np
import numpy.typing as npt
from qubecalib import Sequencer
from qubecalib import neopulse as pls

from ..backend import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_PARAMS_DIR,
    SAMPLING_PERIOD,
    ControlParams,
    DeviceController,
    ExperimentSystem,
    RawResult,
    StateManager,
    Target,
)
from ..backend.dc_voltage_controller import dc_voltage
from ..backend.sequencer_mod import SequencerMod
from ..pulse import Blank, FlatTop, PulseArray, PulseSchedule, RampType
from ..typing import IQArray, TargetMap
from .measurement_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)
from .state_classifier import StateClassifier

DEFAULT_SHOTS: Final = 1024
DEFAULT_INTERVAL: Final = 150 * 1024  # ns
DEFAULT_CAPTURE_WINDOW: Final = 512  # ns
DEFAULT_CAPTURE_OFFSET: Final = 0  # ns
DEFAULT_READOUT_DURATION: Final = 384  # ns
DEFAULT_READOUT_RAMPTIME: Final = 32  # ns
DEFAULT_READOUT_PRE_MARGIN: Final = 32  # ns
DEFAULT_READOUT_POST_MARGIN: Final = 128  # ns
WORD_LENGTH: Final = 4  # samples
WORD_DURATION: Final = WORD_LENGTH * SAMPLING_PERIOD  # ns
BLOCK_LENGTH: Final = WORD_LENGTH * 16  # samples
BLOCK_DURATION: Final = BLOCK_LENGTH * SAMPLING_PERIOD  # ns

logger = logging.getLogger(__name__)


class Measurement:
    def __init__(
        self,
        *,
        chip_id: str,
        qubits: Collection[str] | None = None,
        config_dir: str = DEFAULT_CONFIG_DIR,
        params_dir: str = DEFAULT_PARAMS_DIR,
        connect_devices: bool = True,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
        skew_file_path: Path | str | None = None,
    ):
        """
        Initialize the Measurement.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        qubits : Sequence[str], optional
            The list of qubit labels, by default None.
        config_dir : str, optional
            The configuration directory, by default DEFAULT_CONFIG_DIR.
        params_dir : str, optional
            The parameters directory, by default DEFAULT_PARAMS_DIR.
        connect_devices : bool, optional
            Whether to connect the devices, by default True.
        configuration_mode : Literal["ge-ef-cr", "ge-cr-cr"], optional
            The configuration mode, by default "ge-cr-cr".
        skew_file_path : Path | str | None, optional
            The skew file path, by default None.

        Examples
        --------
        >>> from qubex import Measurement
        >>> meas = Measurement(
        ...     chip_id="64Q",
        ...     qubits=["Q00", "Q01"],
        ... )
        """
        self._chip_id = chip_id
        self._qubits = qubits
        self._config_dir = config_dir
        self._params_dir = params_dir
        self._classifiers: TargetMap[StateClassifier] = {}
        self._initialize(
            connect_devices=connect_devices,
            configuration_mode=configuration_mode,
            skew_file_path=skew_file_path,
        )

    def _initialize(
        self,
        connect_devices: bool,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
        skew_file_path: Path | str | None = None,
    ):
        self._state_manager = StateManager.shared()
        self.state_manager.load(
            chip_id=self._chip_id,
            config_dir=self._config_dir,
            params_dir=self._params_dir,
            configuration_mode=configuration_mode,
        )
        box_ids = []
        if self._qubits is not None:
            boxes = self.experiment_system.get_boxes_for_qubits(self._qubits)
            box_ids = [box.id for box in boxes]
        if len(box_ids) == 0:
            return

        if skew_file_path is None:
            skew_file_path = f"{self._config_dir}/skew.yaml"
        if not Path(skew_file_path).exists():
            print(f"Skew file not found: {skew_file_path}")
        else:
            try:
                self.device_controller.load_skew_file(box_ids, skew_file_path)
            except Exception as e:
                print(f"Failed to load the skew file: {e}")

        if connect_devices:
            try:
                self.device_controller.connect(box_ids)
                self.state_manager.pull(box_ids)
            except Exception as e:
                print(f"Failed to connect to devices: {e}")

    def reload(self):
        """Reload the measuremnt settings."""
        self._initialize(connect_devices=True)

    @property
    def state_manager(self) -> StateManager:
        """Get the state manager."""
        return self._state_manager

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        return self._state_manager.experiment_system

    @property
    def device_controller(self) -> DeviceController:
        """Get the device controller."""
        return self._state_manager.device_controller

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
        clock_statuses = {box: clock for box, clock in zip(box_list, clocks)}
        is_synced = self.device_controller.check_clocks(box_list)
        return {
            "status": is_synced,
            "clocks": clock_statuses,
        }

    def linkup(self, box_list: list[str], noise_threshold: int = 500):
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
            with self.state_manager.modified_frequencies(target_frequencies):
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
            capture_window=duration,
            readout_amplitudes={target: 0 for target in targets},
        )

    def measure(
        self,
        waveforms: TargetMap[IQArray],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int | None = None,
        interval: float | None = None,
        capture_window: float | None = None,
        capture_offset: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        add_pump_pulses: bool = False,
    ) -> MeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        waveforms : TargetMap[IQArray]
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
        capture_window : float, optional
            The capture window in ns.
        capture_offset : float, optional
            The capture offset in ns.
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
        if shots is None:
            shots = DEFAULT_SHOTS

        measure_mode = MeasureMode(mode)
        sequencer = self._create_sequencer(
            waveforms=waveforms,
            interval=interval,
            add_pump_pulses=add_pump_pulses,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_amplitudes=readout_amplitudes,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            capture_window=capture_window,
            capture_offset=capture_offset,
        )
        backend_result = self.device_controller.execute_sequencer(
            sequencer=sequencer,
            repeats=shots,
            integral_mode=measure_mode.integral_mode,
        )
        return self._create_measure_result(
            backend_result=backend_result,
            measure_mode=measure_mode,
            shots=shots,
        )

    def execute(
        self,
        schedule: PulseSchedule,
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
        capture_window: float | None = None,
        capture_offset: float | None = None,
        add_last_measurement: bool = False,
        add_pump_pulses: bool = False,
        plot: bool = False,
    ) -> MultipleMeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        schedule : PulseSchedule
            The pulse schedule.
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
        capture_window : float, optional
            The capture window in ns.
        capture_offset : float, optional
            The capture offset in ns.
        add_last_measurement : bool, optional
            Whether to add the last measurement, by default False.
        add_pump_pulses : bool, optional
            Whether to add pump pulses, by default False.
        plot : bool, optional
            Whether to plot the results, by default False.

        Returns
        -------
        MultipleMeasureResult
            The measurement results.
        """
        if not schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")

        if shots is None:
            shots = DEFAULT_SHOTS

        measure_mode = MeasureMode(mode)
        sequencer = self._create_sequencer_from_schedule(
            schedule=schedule,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            capture_window=capture_window,
            capture_offset=capture_offset,
            add_last_measurement=add_last_measurement,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )
        backend_result = self.device_controller.execute_sequencer(
            sequencer=sequencer,
            repeats=shots,
            integral_mode=measure_mode.integral_mode,
        )
        return self._create_multiple_measure_result(
            backend_result=backend_result,
            measure_mode=measure_mode,
            shots=shots,
        )

    # @cache
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

    # @cache
    def pump_pulse(
        self,
        target: str,
        duration: float | None = None,
        amplitude: float | None = None,
        ramptime: float | None = None,
        type: RampType | None = None,
    ) -> FlatTop:
        qubit = Target.qubit_label(target)
        mux = self.experiment_system.get_mux_by_qubit(qubit)
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

    def _create_sequencer(
        self,
        *,
        waveforms: TargetMap[IQArray],
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_drag_coeff: float | None = None,
        capture_window: float | None = None,
        capture_offset: float | None = None,
        add_pump_pulses: bool = False,
    ) -> Sequencer:
        if interval is None:
            interval = DEFAULT_INTERVAL
        if readout_amplitudes is None:
            readout_amplitudes = self.control_params.readout_amplitude
        if readout_duration is None:
            readout_duration = DEFAULT_READOUT_DURATION
        if readout_pre_margin is None:
            readout_pre_margin = DEFAULT_READOUT_PRE_MARGIN
        if readout_post_margin is None:
            readout_post_margin = DEFAULT_READOUT_POST_MARGIN
        if capture_window is None:
            capture_window = DEFAULT_CAPTURE_WINDOW
        if capture_offset is None:
            capture_offset = 0.0

        qubits = [Target.qubit_label(target) for target in waveforms]
        control_length = max(len(waveform) for waveform in waveforms.values())
        control_length = math.ceil(control_length / BLOCK_LENGTH) * BLOCK_LENGTH
        pre_margin_length = self._number_of_samples(readout_pre_margin)
        readout_length = self._number_of_samples(readout_duration)
        post_margin_length = self._number_of_samples(readout_post_margin)
        total_readout_length = pre_margin_length + readout_length + post_margin_length
        total_length = control_length + total_readout_length
        total_length = math.ceil(total_length / BLOCK_LENGTH) * BLOCK_LENGTH
        readout_slice = slice(control_length, control_length + total_readout_length)
        capture_length = self._number_of_samples(capture_window)
        offset_length = self._number_of_samples(capture_offset, allow_negative=True)
        capture_start = control_length + pre_margin_length + offset_length
        if capture_start < 0 or capture_start % WORD_LENGTH != 0:
            raise ValueError(
                f"Invalid capture start: {capture_start} (must be a multiple of {WORD_LENGTH})"
            )

        # zero padding for user-defined waveforms
        user_waveforms: dict[str, npt.NDArray[np.complex128]] = {}
        for target, waveform in waveforms.items():
            if waveform is None or len(waveform) == 0:
                continue
            padded_waveform = np.zeros(total_length, dtype=np.complex128)
            left_padding = control_length - len(waveform)
            control_slice = slice(left_padding, control_length)
            padded_waveform[control_slice] = waveform
            user_waveforms[target] = padded_waveform

        # add system-defined readout waveforms
        readout_waveforms: dict[str, npt.NDArray[np.complex128]] = {}
        for qubit in qubits:
            readout_target = Target.read_label(qubit)
            if readout_target in user_waveforms:
                padded_waveform = user_waveforms[readout_target]
            else:
                padded_waveform = np.zeros(total_length, dtype=np.complex128)
            readout_pulse = self.readout_pulse(
                target=qubit,
                duration=readout_duration,
                amplitude=readout_amplitudes.get(qubit),
                ramptime=readout_ramptime,
                type=readout_ramp_type,
                drag_coeff=readout_drag_coeff,
                pre_margin=readout_pre_margin,
                post_margin=readout_post_margin,
            )
            padded_waveform[readout_slice] = readout_pulse.values
            omega = 2 * np.pi * self.get_awg_frequency(readout_target)
            offset = capture_start * SAMPLING_PERIOD
            padded_waveform *= np.exp(-1j * omega * offset)
            readout_waveforms[readout_target] = padded_waveform

        # zero padding (pump)
        pump_duration = readout_pre_margin + readout_duration + readout_post_margin
        pump_waveforms: dict[str, npt.NDArray[np.complex128]] = {}
        if add_pump_pulses:
            for qubit in qubits:
                mux = self.experiment_system.get_mux_by_qubit(qubit)
                pump_pulse = self.pump_pulse(
                    target=qubit,
                    duration=pump_duration,
                    amplitude=self.control_params.get_pump_amplitude(mux.index),
                    ramptime=readout_ramptime,
                    type=readout_ramp_type,
                )
                padded_waveform = np.zeros(total_length, dtype=np.complex128)
                padded_waveform[readout_slice] = pump_pulse.values
                pump_waveforms[mux.label] = padded_waveform

        # create dict of GenSampledSequence and CapSampledSequence
        gen_sequences: dict[str, pls.GenSampledSequence] = {}
        cap_sequences: dict[str, pls.CapSampledSequence] = {}
        for target, waveform in user_waveforms.items():
            # add GenSampledSequence (control)
            gen_sequences[target] = pls.GenSampledSequence(
                target_name=target,
                prev_blank=0,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
                modulation_frequency=self.get_awg_frequency(target),
                sub_sequences=[
                    pls.GenSampledSubSequence(
                        real=np.real(waveform),
                        imag=np.imag(waveform),
                        repeats=1,
                        post_blank=None,
                        original_post_blank=None,
                    )
                ],
            )
        for target, waveform in pump_waveforms.items():
            # add GenSampledSequence (pump)
            gen_sequences[target] = pls.GenSampledSequence(
                target_name=target,
                prev_blank=0,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
                modulation_frequency=self.get_awg_frequency(target),
                sub_sequences=[
                    pls.GenSampledSubSequence(
                        real=np.real(waveform),
                        imag=np.imag(waveform),
                        repeats=1,
                        post_blank=None,
                        original_post_blank=None,
                    )
                ],
            )
        for target, waveform in readout_waveforms.items():
            # add GenSampledSequence (readout)
            modulation_frequency = self.get_awg_frequency(target)
            gen_sequences[target] = pls.GenSampledSequence(
                target_name=target,
                prev_blank=0,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
                modulation_frequency=modulation_frequency,
                sub_sequences=[
                    pls.GenSampledSubSequence(
                        real=np.real(waveform),
                        imag=np.imag(waveform),
                        repeats=1,
                        post_blank=None,
                        original_post_blank=None,
                    )
                ],
            )
            # add CapSampledSequence
            cap_sequences[target] = pls.CapSampledSequence(
                target_name=target,
                repeats=None,
                prev_blank=0,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
                modulation_frequency=modulation_frequency,
                sub_sequences=[
                    pls.CapSampledSubSequence(
                        capture_slots=[
                            pls.CaptureSlots(
                                duration=capture_length,
                                post_blank=None,
                                original_duration=None,  # type: ignore
                                original_post_blank=None,
                            )
                        ],
                        repeats=None,
                        prev_blank=capture_start,
                        post_blank=None,
                        original_prev_blank=None,  # type: ignore
                        original_post_blank=None,
                    )
                ],
            )

        # create resource map
        all_targets = (
            list(user_waveforms.keys())
            + list(readout_waveforms.keys())
            + list(pump_waveforms.keys())
        )
        resource_map = self.device_controller.get_resource_map(all_targets)

        # calculate the backend interval
        backend_interval = total_length * SAMPLING_PERIOD + interval
        backend_interval = math.ceil(backend_interval / BLOCK_DURATION) * BLOCK_DURATION
        backend_interval += BLOCK_DURATION  # TODO: remove this hack

        # return Sequencer
        return SequencerMod(
            gen_sampled_sequence=gen_sequences,
            cap_sampled_sequence=cap_sequences,
            resource_map=resource_map,  # type: ignore
            interval=backend_interval,
            sysdb=self.device_controller.qubecalib.sysdb,
        )

    def _create_sampled_sequences_from_schedule(
        self,
        schedule: PulseSchedule,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_drag_coeff: float | None = None,
        capture_window: float | None = None,
        capture_offset: float | None = None,
        add_last_measurement: bool = False,
        add_pump_pulses: bool = False,
        plot: bool = False,
    ) -> tuple[dict[str, pls.GenSampledSequence], dict[str, pls.CapSampledSequence]]:
        if readout_amplitudes is None:
            readout_amplitudes = self.control_params.readout_amplitude
        if readout_duration is None:
            readout_duration = DEFAULT_READOUT_DURATION
        if readout_pre_margin is None:
            readout_pre_margin = DEFAULT_READOUT_PRE_MARGIN
        if readout_post_margin is None:
            readout_post_margin = DEFAULT_READOUT_POST_MARGIN
        if capture_window is None:
            capture_window = DEFAULT_CAPTURE_WINDOW
        if capture_offset is None:
            capture_offset = 0.0

        if not schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")

        # add last readout pulse if necessary
        if add_last_measurement:
            # register all readout targets for the last measurement
            readout_targets = list(
                {
                    Target.read_label(label)
                    for label in schedule.labels
                    if not self.targets[label].is_pump
                }
            )
            # create a new schedule with the last readout pulse
            with PulseSchedule(schedule.labels + readout_targets) as ps:
                ps.call(schedule)
                ps.barrier()
                for target in readout_targets:
                    ps.add(
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
            # update the schedule
            schedule = ps
        else:
            # readout targets in the provided schedule
            readout_targets = [
                label for label in schedule.labels if self.targets[label].is_read
            ]

        # check the readout targets
        if not readout_targets:
            raise ValueError("No readout targets in the pulse schedule.")

        # WORKAROUND: add 2 words (8 samples) blank for the first extra capture by left padding
        extra_sum_section_duration = WORD_DURATION
        extra_post_blank_duration = WORD_DURATION
        extra_capture_duration = extra_sum_section_duration + extra_post_blank_duration
        schedule = schedule.padded(
            total_duration=schedule.duration + extra_capture_duration,
            pad_side="left",
        )

        # ensure the schedule duration is a multiple of MIN_DURATION by right padding
        sequence_duration = (
            math.ceil(schedule.duration / BLOCK_DURATION) * BLOCK_DURATION
        )
        schedule = schedule.padded(
            total_duration=sequence_duration,
            pad_side="right",
        )

        # get readout ranges
        readout_ranges = schedule.get_pulse_ranges(readout_targets)

        # add pump pulses if necessary
        if add_pump_pulses:
            with PulseSchedule() as ps_with_pumps:
                ps_with_pumps.call(schedule)
                for target, ranges in readout_ranges.items():
                    if not ranges:
                        continue
                    mux = self.experiment_system.get_mux_by_qubit(
                        Target.qubit_label(target)
                    )
                    # add pump pulses to overlap with the readout pulses
                    for i in range(len(ranges)):
                        current_range = ranges[i]

                        if i == 0:
                            blank_duration = current_range.start * SAMPLING_PERIOD
                        else:
                            prev_range = ranges[i - 1]
                            blank_duration = (
                                current_range.start - prev_range.stop
                            ) * SAMPLING_PERIOD

                        # start the pump pulse before the readout pulse if there is a pre-margin
                        blank_duration -= readout_pre_margin

                        pump_duration = (
                            current_range.stop - current_range.start
                        ) * SAMPLING_PERIOD + readout_pre_margin

                        pump_amplitude = self.control_params.get_pump_amplitude(
                            mux.index
                        )

                        ps_with_pumps.add(
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

            if not ps_with_pumps.is_valid():
                raise ValueError("Invalid pulse schedule with pump pulses.")

            # update the schedule
            schedule = ps_with_pumps

        if plot:
            schedule.plot()

        # get sampled sequences
        sampled_sequences = schedule.get_sampled_sequences()

        # adjust the phase of the readout pulses
        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            seq = sampled_sequences[target]
            omega = 2 * np.pi * self.get_awg_frequency(target)
            for rng in ranges:
                offset = rng.start * SAMPLING_PERIOD
                seq[rng] *= np.exp(-1j * omega * offset)

        # create GenSampledSequence
        gen_sequences: dict[str, pls.GenSampledSequence] = {}
        for target, waveform in sampled_sequences.items():
            gen_sequences[target] = pls.GenSampledSequence(
                target_name=target,
                prev_blank=0,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
                modulation_frequency=self.get_awg_frequency(target),
                sub_sequences=[
                    # has only one GenSampledSubSequence
                    pls.GenSampledSubSequence(
                        real=np.real(waveform),
                        imag=np.imag(waveform),
                        repeats=1,
                        post_blank=None,
                        original_post_blank=None,
                    )
                ],
            )

        # create CapSampledSequence
        cap_sequences: dict[str, pls.CapSampledSequence] = {}
        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            cap_sub_sequence = pls.CapSampledSubSequence(
                capture_slots=[],
                repeats=None,
                prev_blank=0,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
            )

            # WORKAROUND: add an extra capture to ensure the first capture begins at a multiple of 64 samples
            post_blank_to_first_readout = ranges[0].start - extra_sum_section_duration
            cap_sub_sequence.capture_slots.append(
                pls.CaptureSlots(
                    duration=extra_sum_section_duration,  # type: ignore
                    post_blank=post_blank_to_first_readout,  # type: ignore
                    original_duration=extra_sum_section_duration,  # type: ignore
                    original_post_blank=post_blank_to_first_readout,  # type: ignore
                )
            )

            for i in range(len(ranges) - 1):
                current_range = ranges[i]
                next_range = ranges[i + 1]
                capture_range_length = len(current_range)
                # post_blank_length is the number of samples to the next readout pulse
                post_blank_length = next_range.start - current_range.stop

                if current_range.start % WORD_LENGTH != 0:
                    raise ValueError(
                        f"Capture range should start at a multiple of 4 samples ({WORD_DURATION} ns)."
                    )
                if capture_range_length % WORD_LENGTH != 0:
                    raise ValueError(
                        f"Capture duration should be a multiple of 4 samples ({WORD_DURATION} ns)."
                    )
                if post_blank_length < WORD_LENGTH:
                    raise ValueError(
                        f"Readout pulses must have at least {WORD_DURATION} ns post-blank time."
                    )
                if post_blank_length % WORD_LENGTH != 0:
                    raise ValueError(
                        f"Post-blank time should be a multiple of 4 samples ({WORD_DURATION} ns)."
                    )
                cap_sub_sequence.capture_slots.append(
                    pls.CaptureSlots(
                        duration=capture_range_length,
                        post_blank=post_blank_length,
                        original_duration=None,  # type: ignore
                        original_post_blank=None,  # type: ignore
                    )
                )
            last_capture_range = ranges[-1]
            last_capture_range_length = len(last_capture_range)
            # last_post_blank_length is the number of samples to the end of the schedule
            last_post_blank_length = schedule.length - last_capture_range.stop

            cap_sub_sequence.capture_slots.append(
                pls.CaptureSlots(
                    duration=last_capture_range_length,
                    post_blank=last_post_blank_length,
                    original_duration=None,  # type: ignore
                    original_post_blank=None,  # type: ignore,
                )
            )
            cap_sequence = pls.CapSampledSequence(
                target_name=target,
                repeats=None,
                prev_blank=0,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
                modulation_frequency=self.get_awg_frequency(target),
                sub_sequences=[
                    # has only one CapSampledSubSequence
                    cap_sub_sequence,
                ],
            )
            cap_sequences[target] = cap_sequence

        return gen_sequences, cap_sequences

    def _create_sequencer_from_schedule(
        self,
        schedule: PulseSchedule,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        capture_window: float | None = None,
        capture_offset: float | None = None,
        add_last_measurement: bool = False,
        add_pump_pulses: bool = False,
        plot: bool = False,
    ) -> Sequencer:
        if interval is None:
            interval = DEFAULT_INTERVAL

        gen_sequences, cap_sequences = self._create_sampled_sequences_from_schedule(
            schedule=schedule,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            capture_window=capture_window,
            capture_offset=capture_offset,
            add_last_measurement=add_last_measurement,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )

        backend_interval = (
            math.ceil((schedule.duration + interval) / BLOCK_DURATION) * BLOCK_DURATION
        )

        resource_map = self.device_controller.get_resource_map(schedule.labels)

        return SequencerMod(
            gen_sampled_sequence=gen_sequences,
            cap_sampled_sequence=cap_sequences,
            resource_map=resource_map,  # type: ignore
            interval=backend_interval,
            sysdb=self.device_controller.qubecalib.sysdb,
        )

    def _create_measure_result(
        self,
        backend_result: RawResult,
        measure_mode: MeasureMode,
        shots: int,
    ) -> MeasureResult:
        label_slice = slice(1, None)  # remove the resonator prefix "R"
        norm_factor = 2 ** (-32)  # normalization factor for 32-bit data
        capture_index = 0  # the first capture index

        iq_data = {}
        for target, iqs in sorted(backend_result.data.items()):
            sideband = self.experiment_system.get_target(target).sideband
            if sideband == "L":
                iq_data[target] = np.conjugate(iqs)
            else:
                iq_data[target] = iqs

        if measure_mode == MeasureMode.SINGLE:
            backend_data = {
                # iqs[capture_index]: ndarray[duration, shots]
                target[label_slice]: iqs[capture_index].T * norm_factor
                for target, iqs in iq_data.items()
            }
            measure_data = {
                qubit: MeasureData(
                    target=qubit,
                    mode=measure_mode,
                    raw=iq,
                    classifier=self.classifiers.get(qubit),
                )
                for qubit, iq in backend_data.items()
            }
        elif measure_mode == MeasureMode.AVG:
            backend_data = {
                # iqs[capture_index]: ndarray[duration, 1]
                target[label_slice]: iqs[capture_index].squeeze() * norm_factor / shots
                for target, iqs in iq_data.items()
            }
            measure_data = {
                qubit: MeasureData(
                    target=qubit,
                    mode=measure_mode,
                    raw=iq,
                )
                for qubit, iq in backend_data.items()
            }
        else:
            raise ValueError(f"Invalid measure mode: {measure_mode}")

        return MeasureResult(
            mode=measure_mode,
            data=measure_data,
            config=backend_result.config,
        )

    def _create_multiple_measure_result(
        self,
        backend_result: RawResult,
        measure_mode: MeasureMode,
        shots: int,
    ) -> MultipleMeasureResult:
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
                    measure_data[qubit].append(
                        MeasureData(
                            target=qubit,
                            mode=measure_mode,
                            raw=iq.T * norm_factor,
                            classifier=self.classifiers.get(qubit),
                        )
                    )
        elif measure_mode == MeasureMode.AVG:
            for target, iqs in iq_data.items():
                qubit = target[label_slice]
                for idx, iq in enumerate(iqs):
                    if idx == 0:
                        # skip the first extra capture
                        continue
                    measure_data[qubit].append(
                        MeasureData(
                            target=qubit,
                            mode=measure_mode,
                            raw=iq.squeeze() * norm_factor / shots,
                        )
                    )
        else:
            raise ValueError(f"Invalid measure mode: {measure_mode}")

        return MultipleMeasureResult(
            mode=measure_mode,
            data=dict(measure_data),
            config=backend_result.config,
        )

    @staticmethod
    def _number_of_samples(
        duration: float,
        allow_negative: bool = False,
    ) -> int:
        """
        Returns the number of samples in the waveform.

        Parameters
        ----------
        duration : float
            Duration of the waveform in ns.
        """
        dt = SAMPLING_PERIOD
        if duration < 0 and not allow_negative:
            raise ValueError("Duration must be positive.")

        # Tolerance for floating point comparison
        tolerance = 1e-9
        frac = duration / dt
        N = round(frac)
        if abs(frac - N) > tolerance:
            raise ValueError(
                f"Duration must be a multiple of the sampling period ({dt} ns)."
            )
        return N

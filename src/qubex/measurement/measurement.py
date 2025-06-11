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
from typing_extensions import deprecated

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
DEFAULT_CONTROL_WINDOW: Final = 1024  # ns
DEFAULT_CAPTURE_WINDOW: Final = 1024  # ns
DEFAULT_CAPTURE_MARGIN: Final = 128  # ns
DEFAULT_CAPTURE_DELAY: Final = 896  # ns
DEFAULT_READOUT_DURATION: Final = 512  # ns
DEFAULT_READOUT_RAMPTIME: Final = 32  # ns
INTERVAL_STEP: Final = 10240  # ns
MIN_LENGTH: Final = 64  # samples
MIN_DURATION: Final = MIN_LENGTH * SAMPLING_PERIOD  # ns

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
        use_neopulse: bool = False,
        use_sequencer_execute: bool = True,
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
        self._use_neopulse = use_neopulse
        self._use_sequencer_execute = use_sequencer_execute
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

    def _calc_backend_interval(
        self,
        waveforms: TargetMap[IQArray],
        interval: float,
        control_window: float | None,
        capture_window: float,
    ) -> int:
        control_length = max(len(waveform) for waveform in waveforms.values())
        control_duration = int(control_length * SAMPLING_PERIOD)
        if control_window is not None:
            control_duration = max(control_duration, control_window)
        return (
            math.ceil((control_duration + capture_window + interval) / INTERVAL_STEP)
            * INTERVAL_STEP
        )

    def measure(
        self,
        waveforms: TargetMap[IQArray],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int | None = None,
        interval: float | None = None,
        control_window: float | None = None,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        capture_delay_words: int | None = None,
        _use_sequencer_execute: bool = True,
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
            The number of shots, by default DEFAULT_SHOTS.
        interval : float, optional
            The interval in ns, by default DEFAULT_INTERVAL.
        control_window : float, optional
            The control window in ns, by default None.
        capture_window : float, optional
            The capture window in ns, by default DEFAULT_CAPTURE_WINDOW.
        capture_margin : float, optional
            The capture margin in ns, by default DEFAULT_CAPTURE_MARGIN.
        readout_duration : float, optional
            The readout duration in ns, by default DEFAULT_READOUT_DURATION.
        readout_amplitudes : dict[str, float], optional
            The readout amplitude for each qubit, by default None.

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
        if interval is None:
            interval = DEFAULT_INTERVAL
        if capture_window is None:
            capture_window = DEFAULT_CAPTURE_WINDOW

        backend_interval = self._calc_backend_interval(
            waveforms=waveforms,
            interval=interval,
            control_window=control_window,
            capture_window=capture_window,
        )
        measure_mode = MeasureMode(mode)
        if self._use_neopulse:
            # deprecated
            sequence = self._create_sequence(
                waveforms=waveforms,
                control_window=control_window,
                capture_window=capture_window,
                capture_margin=capture_margin,
                readout_duration=readout_duration,
                readout_amplitudes=readout_amplitudes,
            )
            backend_result = self.device_controller.execute_sequence(
                sequence=sequence,
                repeats=shots,
                interval=backend_interval,
                integral_mode=measure_mode.integral_mode,
            )
        else:
            sequencer = self._create_sequencer(
                waveforms=waveforms,
                interval=backend_interval,
                capture_window=capture_window,
                capture_margin=capture_margin,
                readout_duration=readout_duration,
                readout_amplitudes=readout_amplitudes,
                readout_ramptime=readout_ramptime,
                readout_drag_coeff=readout_drag_coeff,
                readout_ramp_type=readout_ramp_type,
            )
            if self._use_sequencer_execute and _use_sequencer_execute:
                backend_result = self.device_controller.execute_sequencer(
                    sequencer=sequencer,
                    repeats=shots,
                    integral_mode=measure_mode.integral_mode,
                )
            else:
                backend_result = self.device_controller._execute_sequencer(
                    sequencer=sequencer,
                    repeats=shots,
                    integral_mode=measure_mode.integral_mode,
                    capture_delay_words=capture_delay_words,
                )
        return self._create_measure_result(
            backend_result=backend_result,
            measure_mode=measure_mode,
            shots=shots,
        )

    @deprecated("Use `measure` instead.")
    def measure_batch(
        self,
        waveforms_list: Collection[TargetMap[IQArray]],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int | None = None,
        interval: float | None = None,
        control_window: float | None = None,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
    ):
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        waveforms_list : Collection[TargetMap[IQArray]]
            The control waveforms for each target.
            Waveforms are complex I/Q arrays with the sampling period of 2 ns.
        mode : Literal["single", "avg"], optional
            The measurement mode, by default "single".
            - "single": Measure once.
            - "avg": Measure multiple times and average the results.
        shots : int, optional
            The number of shots, by default DEFAULT_SHOTS.
        interval : float, optional
            The interval in ns, by default DEFAULT_INTERVAL.
        control_window : float, optional
            The control window in ns, by default None.
        capture_window : float, optional
            The capture window in ns, by default DEFAULT_CAPTURE_WINDOW.
        capture_margin : float, optional
            The capture margin in ns, by default DEFAULT_CAPTURE_MARGIN.
        readout_duration : float, optional
            The readout duration in ns, by default DEFAULT_READOUT_DURATION.
        readout_amplitudes : dict[str, float], optional
            The readout amplitude for each qubit, by default None.

        Yields
        ------
        MeasureResult
            The measurement results.
        """
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if capture_window is None:
            capture_window = DEFAULT_CAPTURE_WINDOW

        measure_mode = MeasureMode(mode)
        self.device_controller.clear_command_queue()
        for waveforms in waveforms_list:
            backend_interval = self._calc_backend_interval(
                waveforms=waveforms,
                interval=interval,
                control_window=control_window,
                capture_window=capture_window,
            )
            if self._use_neopulse:
                # deprecated
                sequence = self._create_sequence(
                    waveforms=waveforms,
                    control_window=control_window,
                    capture_window=capture_window,
                    capture_margin=capture_margin,
                    readout_duration=readout_duration,
                    readout_amplitudes=readout_amplitudes,
                )
                self.device_controller.add_sequence(
                    sequence=sequence,
                    interval=backend_interval,
                )
            else:
                sequencer = self._create_sequencer(
                    waveforms=waveforms,
                    interval=backend_interval,
                    capture_window=capture_window,
                    capture_margin=capture_margin,
                    readout_duration=readout_duration,
                    readout_amplitudes=readout_amplitudes,
                )
                self.device_controller.add_sequencer(sequencer)
        backend_results = self.device_controller.execute(
            repeats=shots,
            integral_mode=measure_mode.integral_mode,
        )
        for backend_result in backend_results:
            yield self._create_measure_result(
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
        add_last_measurement: bool = False,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
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
            The number of shots, by default DEFAULT_SHOTS.
        interval : float, optional
            The interval in ns, by default DEFAULT_INTERVAL.
        add_last_measurement : bool, optional
            Whether to add the last measurement, by default False.
        capture_window : float, optional
            The capture window in ns, by default DEFAULT_CAPTURE_WINDOW.
        capture_margin : float, optional
            The capture margin in ns, by default DEFAULT_CAPTURE_MARGIN.
        readout_duration : float, optional
            The readout duration in ns, by default DEFAULT_READOUT_DURATION.
        readout_amplitudes : dict[str, float], optional
            The readout amplitude for each qubit, by default None.

        Returns
        -------
        MultipleMeasureResult
            The measurement results.
        """
        if not schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")

        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

        measure_mode = MeasureMode(mode)
        sequencer = self._create_sequencer_from_schedule(
            schedule=schedule,
            interval=interval,
            add_last_measurement=add_last_measurement,
            capture_window=capture_window,
            capture_margin=capture_margin,
            readout_duration=readout_duration,
            readout_amplitudes=readout_amplitudes,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
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

    @deprecated("Use `create_sequencer` instead.")
    def _create_sequence(
        self,
        *,
        waveforms: TargetMap[IQArray],
        control_window: float | None = None,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
    ) -> pls.Sequence:
        if control_window is None:
            control_window = DEFAULT_CONTROL_WINDOW
        if capture_window is None:
            capture_window = DEFAULT_CAPTURE_WINDOW
        if capture_margin is None:
            capture_margin = DEFAULT_CAPTURE_MARGIN
        if readout_duration is None:
            readout_duration = DEFAULT_READOUT_DURATION
        if readout_amplitudes is None:
            readout_amplitudes = self.control_params.readout_amplitude

        capture = pls.Capture(duration=capture_window)
        qubits = {Target.qubit_label(target) for target in waveforms}
        with pls.Sequence() as sequence:
            with pls.Flushright():
                pls.padding(control_window)
                for target, waveform in waveforms.items():
                    pls.Arbit(np.array(waveform)).target(target)
            with pls.Series():
                pls.padding(capture_margin)
                with pls.Flushleft():
                    for qubit in qubits:
                        readout_target = Target.read_label(qubit)
                        pls.RaisedCosFlatTop(
                            duration=readout_duration,
                            amplitude=readout_amplitudes[qubit],
                            rise_time=DEFAULT_READOUT_RAMPTIME,
                        ).target(readout_target)
                        capture.target(readout_target)
        return sequence

    def readout_pulse(
        self,
        target: str,
        duration: float | None = None,
        amplitude: float | None = None,
        tau: float | None = None,
        beta: float | None = None,
        type: RampType | None = None,
    ) -> FlatTop:
        qubit = Target.qubit_label(target)
        if duration is None:
            duration = DEFAULT_READOUT_DURATION
        if amplitude is None:
            amplitude = self.control_params.readout_amplitude[qubit]
        if tau is None:
            tau = DEFAULT_READOUT_RAMPTIME
        if beta is None:
            beta = 0.0
        if type is None:
            type = "RaisedCosine"
        return FlatTop(
            duration=duration,
            amplitude=amplitude,
            tau=tau,
            beta=beta,
            type=type,
        )

    def _create_sequencer(
        self,
        *,
        waveforms: TargetMap[IQArray],
        interval: float,
        control_window: float | None = None,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
    ) -> Sequencer:
        if capture_window is None:
            capture_window = DEFAULT_CAPTURE_WINDOW
        if capture_margin is None:
            capture_margin = DEFAULT_CAPTURE_MARGIN
        if readout_duration is None:
            readout_duration = DEFAULT_READOUT_DURATION
        if readout_amplitudes is None:
            readout_amplitudes = self.control_params.readout_amplitude

        qubits = [Target.qubit_label(target) for target in waveforms]
        control_length = max(len(waveform) for waveform in waveforms.values())
        control_length = math.ceil(control_length / MIN_LENGTH) * MIN_LENGTH
        if control_window is not None:
            control_length = max(
                control_length,
                self._number_of_samples(control_window),
            )
        margin_length = self._number_of_samples(capture_margin)
        capture_length = self._number_of_samples(capture_window)
        readout_length = self._number_of_samples(readout_duration)
        total_length = control_length + margin_length + capture_length
        readout_start = control_length + margin_length

        # zero padding (control)
        # [0, .., 0, 0, control, 0, 0, .., 0, 0, 0, 0, 0, .., 0, 0, 0]
        # |<- control_length -><- margin_length -><- capture_length ->|
        control_waveforms: dict[str, npt.NDArray[np.complex128]] = {}
        for target, waveform in waveforms.items():
            if waveform is None or len(waveform) == 0:
                continue
            padded_waveform = np.zeros(total_length, dtype=np.complex128)
            left_padding = control_length - len(waveform)
            control_slice = slice(left_padding, control_length)
            padded_waveform[control_slice] = waveform
            control_waveforms[target] = padded_waveform

        # zero padding (readout)
        # [0, .., 0, 0, 0, 0, 0, 0, 0, .., 0, 0, 0, readout, 0, ..., 0]
        # |<- control_length -><- margin_length -><- capture_length ->|
        readout_waveforms: dict[str, npt.NDArray[np.complex128]] = {}
        for qubit in qubits:
            readout_pulse = self.readout_pulse(
                target=qubit,
                duration=readout_duration,
                amplitude=readout_amplitudes.get(qubit),
                tau=readout_ramptime,
                beta=readout_drag_coeff,
                type=readout_ramp_type,
            )
            padded_waveform = np.zeros(total_length, dtype=np.complex128)
            readout_slice = slice(readout_start, readout_start + readout_length)
            padded_waveform[readout_slice] = readout_pulse.values
            readout_target = Target.read_label(qubit)
            omega = 2 * np.pi * self.get_awg_frequency(readout_target)
            offset = readout_start * SAMPLING_PERIOD
            padded_waveform *= np.exp(-1j * omega * offset)
            readout_waveforms[readout_target] = padded_waveform

        # create dict of GenSampledSequence and CapSampledSequence
        gen_sequences: dict[str, pls.GenSampledSequence] = {}
        cap_sequences: dict[str, pls.CapSampledSequence] = {}
        for target, waveform in control_waveforms.items():
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
                                original_duration=capture_window,
                                original_post_blank=None,
                            )
                        ],
                        repeats=None,
                        prev_blank=readout_start,
                        post_blank=None,
                        original_prev_blank=readout_start,
                        original_post_blank=None,
                    )
                ],
            )

        # create resource map
        all_targets = list(control_waveforms.keys()) + list(readout_waveforms.keys())
        resource_map = self.device_controller.get_resource_map(all_targets)

        # return Sequencer
        return SequencerMod(
            gen_sampled_sequence=gen_sequences,
            cap_sampled_sequence=cap_sequences,
            resource_map=resource_map,  # type: ignore
            interval=interval,
            sysdb=self.device_controller.qubecalib.sysdb,
        )

    def _create_sampled_sequences_from_schedule(
        self,
        schedule: PulseSchedule,
        add_last_measurement: bool = False,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
    ) -> tuple[dict[str, pls.GenSampledSequence], dict[str, pls.CapSampledSequence]]:
        if capture_window is None:
            capture_window = DEFAULT_CAPTURE_WINDOW
        if capture_margin is None:
            capture_margin = DEFAULT_CAPTURE_MARGIN
        if readout_duration is None:
            readout_duration = DEFAULT_READOUT_DURATION
        if readout_amplitudes is None:
            readout_amplitudes = self.control_params.readout_amplitude

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
                        PulseArray(
                            [
                                Blank(capture_margin),
                                self.readout_pulse(
                                    target=target,
                                    duration=readout_duration,
                                    amplitude=readout_amplitudes.get(target),
                                    tau=readout_ramptime,
                                    beta=readout_drag_coeff,
                                    type=readout_ramp_type,
                                ).padded(
                                    total_duration=capture_window,
                                    pad_side="right",
                                ),
                            ]
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
        word_duration = SAMPLING_PERIOD * 4
        extra_sum_section_duration = word_duration
        extra_post_blank_duration = word_duration
        extra_capture_duration = extra_sum_section_duration + extra_post_blank_duration
        schedule = schedule.padded(
            total_duration=schedule.duration + extra_capture_duration,
            pad_side="left",
        )

        # ensure the schedule duration is a multiple of MIN_DURATION by right padding
        sequence_duration = math.ceil(schedule.duration / MIN_DURATION) * MIN_DURATION
        schedule = schedule.padded(
            total_duration=sequence_duration,
            pad_side="right",
        )

        # get sampled sequences
        sampled_sequences = schedule.get_sampled_sequences()

        # get readout ranges
        readout_ranges = schedule.get_pulse_ranges(readout_targets)

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
                duration = len(current_range)
                # post_blank is the time to the next readout pulse
                post_blank = next_range.start - current_range.stop
                if post_blank <= 0:
                    raise ValueError(
                        "Readout pulses must have blank time between them."
                    )
                if post_blank % word_duration != 0:
                    raise ValueError(
                        f"Blank time between readout pulses must be a multiple of {word_duration} ns."
                    )
                cap_sub_sequence.capture_slots.append(
                    pls.CaptureSlots(
                        duration=duration,
                        post_blank=post_blank,
                        original_duration=duration,
                        original_post_blank=post_blank,
                    )
                )
            last_range = ranges[-1]
            last_duration = len(last_range)
            # last_post_blank is the time to the end of the schedul
            last_post_blank = schedule.length - last_range.stop

            cap_sub_sequence.capture_slots.append(
                pls.CaptureSlots(
                    duration=last_duration,
                    post_blank=last_post_blank,
                    original_duration=last_duration,
                    original_post_blank=last_post_blank,
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
        interval: float,
        add_last_measurement: bool = False,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
    ) -> Sequencer:
        gen_sequences, cap_sequences = self._create_sampled_sequences_from_schedule(
            schedule=schedule,
            add_last_measurement=add_last_measurement,
            capture_window=capture_window,
            capture_margin=capture_margin,
            readout_duration=readout_duration,
            readout_amplitudes=readout_amplitudes,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
        )

        backend_interval = (
            math.ceil((schedule.duration + interval) / INTERVAL_STEP) * INTERVAL_STEP
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
    ) -> int:
        """
        Returns the number of samples in the waveform.

        Parameters
        ----------
        duration : float
            Duration of the waveform in ns.
        """
        dt = SAMPLING_PERIOD
        if duration < 0:
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

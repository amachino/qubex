from __future__ import annotations

import typing
from contextlib import contextmanager
from functools import cached_property
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

try:
    from qubecalib import Sequencer
    from qubecalib import neopulse as pls
except ImportError:
    pass

from ..backend import (
    DEFAULT_CONFIG_DIR,
    SAMPLING_PERIOD,
    ControlParams,
    DeviceController,
    ExperimentSystem,
    RawResult,
    StateManager,
    Target,
)
from ..pulse import Blank, FlatTop, PulseSchedule, PulseSequence
from ..typing import IQArray, TargetMap
from .measurement_result import MeasureData, MeasureMode, MeasureResult
from .state_classifier import StateClassifier

DEFAULT_SHOTS: Final = 1024
DEFAULT_INTERVAL: Final = 150 * 1024  # ns
DEFAULT_CONTROL_WINDOW: Final = 1024  # ns
DEFAULT_CAPTURE_WINDOW: Final = 1024  # ns
DEFAULT_CAPTURE_OFFSET: Final = 128  # ns
DEFAULT_READOUT_DURATION: Final = 512  # ns
INTERVAL_STEP: Final = 10240  # ns
MIN_DURATION: Final = 128  # ns


class Measurement:
    def __init__(
        self,
        chip_id: str,
        *,
        config_dir: str = DEFAULT_CONFIG_DIR,
        use_neopulse: bool = True,
    ):
        """
        Initialize the Measurement.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        config_dir : str, optional
            The configuration directory, by default "./config".

        Examples
        --------
        >>> from qubex import Measurement
        >>> meas = Measurement("64Q")
        """
        self._state_manager = StateManager.shared()
        self._state_manager.load(
            chip_id=chip_id,
            config_dir=config_dir,
        )
        self._use_neopulse = use_neopulse
        self.classifiers: dict[str, StateClassifier] = {}

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

    @cached_property
    def targets(self) -> dict[str, Target]:
        """Get the targets."""
        return {target.label: target for target in self.experiment_system.targets}

    @cached_property
    def base_frequencies(self) -> dict[str, float]:
        """Get the base frequencies."""
        return {
            target.label: self.experiment_system.get_base_frequency(target.label)
            for target in self.experiment_system.targets
        }

    @cached_property
    def diff_frequencies(self) -> dict[str, float]:
        """Get the base frequencies."""
        return {
            target.label: self.experiment_system.get_diff_frequency(target.label)
            for target in self.experiment_system.targets
        }

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
        original_frequencies = {
            label: target.frequency
            for label, target in self.targets.items()
            if label in target_frequencies
        }
        self.device_controller.modify_target_frequencies(target_frequencies)
        try:
            yield
        finally:
            self.device_controller.modify_target_frequencies(original_frequencies)

    def measure_noise(
        self,
        targets: list[str],
        duration: int,
    ) -> MeasureResult:
        """
        Measure the readout noise.

        Parameters
        ----------
        targets : list[str]
            The list of target names.
        duration : int, optional
            The duration in ns.

        Returns
        -------
        MeasureResult
            The measurement results.

        Examples
        --------
        >>> result = meas.measure_noise()
        """
        capture = pls.Capture(duration=duration)
        readout_targets = {Target.readout_label(target) for target in targets}
        with pls.Sequence() as sequence:
            with pls.Flushleft():
                for target in readout_targets:
                    capture.target(target)
        backend_result = self.device_controller.execute_sequence(
            sequence=sequence,
            repeats=1,
            interval=DEFAULT_INTERVAL,
            integral_mode="single",
        )
        return self._create_measure_result(backend_result, MeasureMode.SINGLE)

    def measure(
        self,
        waveforms: TargetMap[IQArray],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        capture_offset: int = DEFAULT_CAPTURE_OFFSET,
        readout_duration: int = DEFAULT_READOUT_DURATION,
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
        interval : int, optional
            The interval in ns, by default DEFAULT_INTERVAL.
        control_window : int, optional
            The control window in ns, by default DEFAULT_CONTROL_WINDOW.
        capture_window : int, optional
            The capture window in ns, by default DEFAULT_CAPTURE_WINDOW.
        capture_offset : int, optional
            The capture offset in ns, by default DEFAULT_CAPTURE_OFFSET.
        readout_duration : int, optional
            The readout duration in ns, by default DEFAULT_READOUT_DURATION.

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
        backend_interval = (
            (control_window + capture_window + interval) // INTERVAL_STEP + 1
        ) * INTERVAL_STEP

        measure_mode = MeasureMode(mode)
        if self._use_neopulse:
            sequence = self._create_sequence(
                waveforms=waveforms,
                control_window=control_window,
                capture_window=capture_window,
                capture_offset=capture_offset,
                readout_duration=readout_duration,
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
                capture_window=capture_window,
                capture_offset=capture_offset,
                readout_duration=readout_duration,
            )
            backend_result = self.device_controller.execute_sequencer(
                sequencer=sequencer,
                repeats=shots,
                interval=backend_interval,
                integral_mode=measure_mode.integral_mode,
            )
        return self._create_measure_result(backend_result, measure_mode)

    def measure_batch(
        self,
        waveforms_list: typing.Sequence[TargetMap[IQArray]],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        capture_offset: int = DEFAULT_CAPTURE_OFFSET,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ):
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        waveforms_list : Sequence[TargetMap[IQArray]]
            The control waveforms for each target.
            Waveforms are complex I/Q arrays with the sampling period of 2 ns.
        mode : Literal["single", "avg"], optional
            The measurement mode, by default "single".
            - "single": Measure once.
            - "avg": Measure multiple times and average the results.
        shots : int, optional
            The number of shots, by default DEFAULT_SHOTS.
        interval : int, optional
            The interval in ns, by default DEFAULT_INTERVAL.
        control_window : int, optional
            The control window in ns, by default DEFAULT_CONTROL_WINDOW.
        capture_window : int, optional
            The capture window in ns, by default DEFAULT_CAPTURE_WINDOW.
        capture_offset : int, optional
            The capture offset in ns, by default DEFAULT_CAPTURE_OFFSET.
        readout_duration : int, optional
            The readout duration in ns, by default DEFAULT_READOUT_DURATION.

        Yields
        ------
        MeasureResult
            The measurement results.
        """
        backend_interval = (
            (control_window + capture_window + interval) // INTERVAL_STEP + 1
        ) * INTERVAL_STEP

        measure_mode = MeasureMode(mode)
        self.device_controller.clear_command_queue()
        for waveforms in waveforms_list:
            if self._use_neopulse:
                sequence = self._create_sequence(
                    waveforms=waveforms,
                    control_window=control_window,
                    capture_window=capture_window,
                    capture_offset=capture_offset,
                    readout_duration=readout_duration,
                )
                self.device_controller.add_sequence(sequence)
            else:
                sequencer = self._create_sequencer(
                    waveforms=waveforms,
                    capture_window=capture_window,
                    capture_offset=capture_offset,
                    readout_duration=readout_duration,
                )
                self.device_controller.add_sequencer(sequencer)
        backend_results = self.device_controller.execute(
            repeats=shots,
            interval=backend_interval,
            integral_mode=measure_mode.integral_mode,
        )
        for backend_result in backend_results:
            yield self._create_measure_result(backend_result, measure_mode)

    def execute(
        self,
        schedule: PulseSchedule,
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        capture_offset: int = DEFAULT_CAPTURE_OFFSET,
    ) -> MeasureResult:
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
        interval : int, optional
            The interval in ns, by default DEFAULT_INTERVAL.
        capture_offset : int, optional
            The capture offset in ns, by default DEFAULT_CAPTURE_OFFSET.

        Returns
        -------
        MeasureResult
            The measurement results.
        """
        backend_interval = (
            (int(schedule.duration) + interval) // INTERVAL_STEP + 1
        ) * INTERVAL_STEP

        measure_mode = MeasureMode(mode)
        sequencer = self._create_sequencer_from_schedule(
            schedule=schedule,
            add_last_measurement=False,
            capture_offset=capture_offset,
        )
        backend_result = self.device_controller.execute_sequencer(
            sequencer=sequencer,
            repeats=shots,
            interval=backend_interval,
            integral_mode=measure_mode.integral_mode,
        )
        return self._create_measure_result(backend_result, measure_mode)

    def _create_sequence(
        self,
        *,
        waveforms: TargetMap[IQArray],
        control_window: int = DEFAULT_CONTROL_WINDOW,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        capture_offset: int = DEFAULT_CAPTURE_OFFSET,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ) -> pls.Sequence:
        readout_amplitude = self.control_params.readout_amplitude
        capture = pls.Capture(duration=capture_window)
        qubits = {Target.qubit_label(target) for target in waveforms.keys()}
        with pls.Sequence() as sequence:
            with pls.Flushright():
                pls.padding(control_window)
                for target, waveform in waveforms.items():
                    pls.Arbit(np.array(waveform)).target(target)
            with pls.Series():
                pls.padding(capture_offset)
                with pls.Flushleft():
                    for qubit in qubits:
                        readout_target = Target.readout_label(qubit)
                        pls.RaisedCosFlatTop(
                            duration=readout_duration,
                            amplitude=readout_amplitude[qubit],
                            rise_time=32,
                        ).target(readout_target)
                        capture.target(readout_target)
        return sequence

    def _readout_pulse(
        self,
        target: str,
        duration: int = DEFAULT_READOUT_DURATION,
    ) -> FlatTop:
        qubit = Target.qubit_label(target)
        readout_amplitude = self.control_params.readout_amplitude
        return FlatTop(
            duration=duration,
            amplitude=readout_amplitude[qubit],
            tau=32,
        )

    def _create_sequencer(
        self,
        *,
        waveforms: TargetMap[IQArray],
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        capture_offset: int = DEFAULT_CAPTURE_OFFSET,
        readout_duration: int = DEFAULT_READOUT_DURATION,
    ) -> Sequencer:
        qubits = {Target.qubit_label(target) for target in waveforms}
        control_length = max(len(waveform) for waveform in waveforms.values())
        control_length = (control_length // MIN_DURATION + 1) * MIN_DURATION
        offset_length = self._number_of_samples(capture_offset)
        capture_length = self._number_of_samples(capture_window)
        readout_length = self._number_of_samples(readout_duration)
        total_length = control_length + offset_length + capture_length
        readout_start = control_length + offset_length

        # zero padding (control)
        # [0, .., 0, 0, control, 0, 0, .., 0, 0, 0, 0, 0, .., 0, 0, 0]
        # |<- control_length -><- offset_length -><- capture_length ->|
        control_waveforms: dict[str, npt.NDArray[np.complex128]] = {}
        for target, waveform in waveforms.items():
            padded_waveform = np.zeros(total_length, dtype=np.complex128)
            left_padding = control_length - len(waveform)
            control_slice = slice(left_padding, control_length)
            padded_waveform[control_slice] = waveform
            control_waveforms[target] = padded_waveform

        # zero padding (readout)
        # [0, .., 0, 0, 0, 0, 0, 0, 0, .., 0, 0, 0, readout, 0, ..., 0]
        # |<- control_length -><- offset_length -><- capture_length ->|
        readout_waveforms: dict[str, npt.NDArray[np.complex128]] = {}
        for qubit in qubits:
            readout_pulse = self._readout_pulse(qubit, readout_duration)
            padded_waveform = np.zeros(total_length, dtype=np.complex128)
            readout_slice = slice(readout_start, readout_start + readout_length)
            padded_waveform[readout_slice] = readout_pulse.values
            readout_target = Target.readout_label(qubit)
            omega = 2 * np.pi * self.diff_frequencies[readout_target]
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
                sub_sequences=[
                    pls.GenSampledSubSequence(
                        real=np.real(waveform),
                        imag=np.imag(waveform),
                        post_blank=None,
                        repeats=1,
                    )
                ],
            )
        for target, waveform in readout_waveforms.items():
            # add GenSampledSequence (readout)
            gen_sequences[target] = pls.GenSampledSequence(
                target_name=target,
                prev_blank=0,
                post_blank=None,
                sub_sequences=[
                    pls.GenSampledSubSequence(
                        real=np.real(waveform),
                        imag=np.imag(waveform),
                        post_blank=None,
                        repeats=1,
                    )
                ],
            )
            # add CapSampledSequence
            cap_sequences[target] = pls.CapSampledSequence(
                target_name=target,
                prev_blank=0,
                post_blank=None,
                repeats=None,
                sub_sequences=[
                    pls.CapSampledSubSequence(
                        capture_slots=[
                            pls.CaptureSlots(
                                duration=capture_length,
                                post_blank=None,
                            )
                        ],
                        prev_blank=readout_start,
                        post_blank=None,
                        repeats=None,
                    )
                ],
            )

        # create resource map
        all_targets = list(control_waveforms.keys()) + list(readout_waveforms.keys())
        resource_map = self.device_controller.get_resource_map(all_targets)

        # return Sequencer
        return Sequencer(
            gen_sampled_sequence=gen_sequences,
            cap_sampled_sequence=cap_sequences,
            resource_map=resource_map,  # type: ignore
        )

    def _create_sequencer_from_schedule(
        self,
        schedule: PulseSchedule,
        add_last_measurement: bool = False,
        capture_offset: int = DEFAULT_CAPTURE_OFFSET,
    ) -> Sequencer:
        if not schedule.is_valid():
            raise ValueError("Invalid pulse schedule.")

        # readout targets in the provided schedule
        readout_targets = [
            target for target in schedule.targets if Target.is_readout(target)
        ]

        # add last readout pulse if necessary
        if add_last_measurement:
            # register all readout targets for the last measurement
            readout_targets = list(
                {Target.readout_label(target) for target in schedule.targets}
            )
            # create a new schedule with the last readout pulse
            with PulseSchedule(schedule.targets + readout_targets) as ps:
                ps.call(schedule)
                ps.barrier()
                for target in readout_targets:
                    readout_pulse = PulseSequence(
                        [
                            Blank(duration=capture_offset),
                            self._readout_pulse(target),
                        ]
                    )
                    ps.add(target, readout_pulse)
            # update the schedule
            schedule = ps

        # get sampled sequences
        sampled_sequences = schedule.get_sampled_sequences()

        # create GenSampledSequence
        gen_sequences: dict[str, pls.GenSampledSequence] = {}
        for target, waveform in sampled_sequences.items():
            gen_sequences[target] = pls.GenSampledSequence(
                target_name=target,
                prev_blank=0,
                post_blank=None,
                sub_sequences=[
                    # has only one GenSampledSubSequence
                    pls.GenSampledSubSequence(
                        real=np.real(waveform),
                        imag=np.imag(waveform),
                        post_blank=None,
                        repeats=1,
                    )
                ],
            )

        # create CapSampledSequence
        cap_sequences: dict[str, pls.CapSampledSequence] = {}
        readout_ranges = schedule.get_pulse_ranges(readout_targets)
        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            cap_sub_sequence = pls.CapSampledSubSequence(
                capture_slots=[],
                # prev_blank is the time to the first readout pulse
                prev_blank=ranges[0].start,
                post_blank=None,
                repeats=None,
            )
            for i in range(len(ranges) - 1):
                current_range = ranges[i]
                next_range = ranges[i + 1]
                cap_sub_sequence.capture_slots.append(
                    pls.CaptureSlots(
                        duration=len(current_range),
                        # post_blank is the time to the next readout pulse
                        post_blank=next_range.start - current_range.stop,
                    )
                )
            last_range = ranges[-1]
            cap_sub_sequence.capture_slots.append(
                pls.CaptureSlots(
                    duration=len(last_range),
                    # post_blank is the time to the end of the schedule
                    post_blank=schedule.length - last_range.stop,
                )
            )
            cap_sequence = pls.CapSampledSequence(
                target_name=target,
                prev_blank=0,
                post_blank=None,
                repeats=None,
                sub_sequences=[
                    # has only one CapSampledSubSequence
                    cap_sub_sequence,
                ],
            )
            cap_sequences[target] = cap_sequence

        # create resource map
        resource_map = self.device_controller.get_resource_map(schedule.targets)

        # return Sequencer
        return Sequencer(
            gen_sampled_sequence=gen_sequences,
            cap_sampled_sequence=cap_sequences,
            resource_map=resource_map,  # type: ignore
        )

    def _create_measure_result(
        self,
        backend_result: RawResult,
        measure_mode: MeasureMode,
    ) -> MeasureResult:
        label_slice = slice(1, None)  # Remove the prefix "R"
        capture_index = 0

        measure_data = {}
        for target, iqs in backend_result.data.items():
            qubit = target[label_slice]

            if measure_mode == MeasureMode.SINGLE:
                # iqs: ndarray[duration, shots]
                raw = iqs[capture_index].T.squeeze()
                kerneled = np.mean(iqs[capture_index], axis=0) * 2 ** (-32)
                classifier = self.classifiers.get(qubit)
                if classifier is None:
                    classified_data = None
                else:
                    classified_data = classifier.classify(qubit, kerneled, plot=False)
            elif measure_mode == MeasureMode.AVG:
                # iqs: ndarray[duration, 1]
                raw = iqs[capture_index].squeeze()
                kerneled = np.mean(iqs) * 2 ** (-32)
                classified_data = None
            else:
                raise ValueError(f"Invalid measure mode: {measure_mode}")

            measure_data[qubit] = MeasureData(
                target=target,
                mode=measure_mode,
                raw=raw,
                kerneled=kerneled,
                classified=classified_data,
            )

        return MeasureResult(
            mode=measure_mode,
            data=measure_data,
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

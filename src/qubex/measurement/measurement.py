"""
High-level measurement orchestration API.

This module exposes `Measurement`, a facade that composes device setup,
schedule construction, and execution components for measurement runs.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping
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
    RawResult,
    SystemManager,
    Target,
)
from qubex.backend.dc_voltage_controller import dc_voltage
from qubex.pulse import Blank, PulseArray, PulseSchedule, RampType
from qubex.typing import IQArray, TargetMap

from .classifiers.state_classifier import StateClassifier
from .measurement_device_manager import MeasurementDeviceManager
from .measurement_pulse_factory import MeasurementPulseFactory
from .models.measurement_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)

logger = logging.getLogger(__name__)

try:
    from qubecalib import Sequencer
    from qubecalib import neopulse as pls

    from qubex.backend.sequencer_mod import SequencerMod
except ImportError as e:
    logger.info(e)

DEFAULT_SHOTS: Final = 1024
DEFAULT_INTERVAL: Final = 150 * 1024  # ns
DEFAULT_READOUT_DURATION: Final = 384  # ns
DEFAULT_READOUT_RAMPTIME: Final = 32  # ns
DEFAULT_READOUT_PRE_MARGIN: Final = 32  # ns
DEFAULT_READOUT_POST_MARGIN: Final = 128  # ns
WORD_LENGTH: Final = 4  # samples
WORD_DURATION: Final = WORD_LENGTH * SAMPLING_PERIOD  # ns
BLOCK_LENGTH: Final = WORD_LENGTH * 16  # samples
BLOCK_DURATION: Final = BLOCK_LENGTH * SAMPLING_PERIOD  # ns

EXTRA_SUM_SECTION_LENGTH = WORD_LENGTH * 4  # samples
EXTRA_POST_BLANK_LENGTH = WORD_LENGTH  # samples
EXTRA_CAPTURE_LENGTH = EXTRA_SUM_SECTION_LENGTH + EXTRA_POST_BLANK_LENGTH  # samples
EXTRA_CAPTURE_DURATION = EXTRA_CAPTURE_LENGTH * SAMPLING_PERIOD  # ns


class Measurement:
    """
    Coordinate end-to-end measurement workflows.

    `Measurement` owns the high-level workflow while delegating concrete
    responsibilities to focused collaborators: configuration/device lifecycle
    (`MeasurementDeviceManager`), schedule assembly
    (`MeasurementScheduleBuilder` and `MeasurementPulseFactory`), and backend
    execution (`MeasurementRunner`/`QuelDeviceExecutor`). It also keeps optional
    state classifiers used during readout post-processing.
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
            Whether to connect the devices, by default False.
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
        self._system_manager = SystemManager.shared()
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
        >>> meas.check_link_status(["Q73A", "U10B"])
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
        >>> meas.check_clock_status(["Q73A", "U10B"])
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
        >>> meas.linkup(["Q73A", "U10B"])
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
        >>> meas.relinkup(["Q73A", "U10B"])
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
        >>> with meas.modified_frequencies({"Q00": 5.0}):
        ...     result = meas.measure({
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
        >>> result = meas.measure_noise()
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
            add_last_measurement=add_last_measurement,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )
        backend_result = self.device_controller.execute_sequencer(
            sequencer=sequencer,
            repeats=shots,
            integral_mode=measure_mode.integral_mode,
            dsp_demodulation=enable_dsp_demodulation,
            enable_sum=enable_dsp_sum,
            enable_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )
        result = self._create_multiple_measure_result(
            backend_result=backend_result,
            measure_mode=measure_mode,
            shots=shots,
        )
        rawdata_dir = self.system_manager.rawdata_dir
        if rawdata_dir is not None:
            result.save(data_dir=rawdata_dir)
        return result

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
        capture_delays: dict[int, int] | None = None,
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
        if capture_delays is None:
            capture_delays = self.control_params.capture_delay_word

        # add last readout pulse if necessary
        if add_last_measurement:
            # ensure the schedule duration is a multiple of WORD_DURATION by right padding
            sequence_duration = (
                math.ceil(schedule.duration / WORD_DURATION) * WORD_DURATION
            )
            schedule.pad(
                total_duration=sequence_duration,
                pad_side="right",
            )

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
                        self.pulse_factory.readout_pulse(
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

        # WORKAROUND: add some blank for the first extra capture by left padding
        schedule.pad(
            total_duration=schedule.duration + EXTRA_CAPTURE_DURATION,
            pad_side="left",
        )

        # ensure the schedule duration is a multiple of MIN_DURATION by right padding
        sequence_duration = (
            math.ceil(schedule.duration / BLOCK_DURATION + 1) * BLOCK_DURATION
        )
        schedule.pad(
            total_duration=sequence_duration,
            pad_side="right",
        )

        # get readout ranges
        readout_ranges = schedule.get_pulse_ranges(readout_targets)

        capture_delay_sample = {}
        for target in readout_targets:
            mux = self.mux_dict[Target.qubit_label(target)]
            capture_delay_word = capture_delays.get(mux.index)
            if capture_delay_word is None:
                capture_delay_word = 0
            capture_delay_sample[target] = capture_delay_word * WORD_LENGTH

        # add pump pulses if necessary
        if add_pump_pulses:
            # TODO: There is a bug where, when pumping while simultaneously reading out multiple resonators in the same MUX, multiple pump pulses are placed in series.
            with PulseSchedule() as ps_with_pumps:
                ps_with_pumps.call(schedule)
                for target, ranges in readout_ranges.items():
                    if not ranges:
                        continue
                    mux = self.mux_dict[Target.qubit_label(target)]
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
                                    self.pulse_factory.pump_pulse(
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
        sampled_sequences = schedule.get_sampled_sequences(copy=False)

        # adjust the phase of the readout pulses
        for target, ranges in readout_ranges.items():
            if not ranges:
                continue
            seq = sampled_sequences[target]
            # use diff_frequency instead of awg_frequency since the envelope will be adjusted by conjugation later
            omega = 2 * np.pi * self.get_diff_frequency(target)
            delay = capture_delay_sample[target]
            for rng in ranges:
                offset = (rng.start + delay) * SAMPLING_PERIOD
                seq[rng] *= np.exp(1j * omega * offset)

        # create GenSampledSequence
        gen_sequences: dict[str, pls.GenSampledSequence] = {}
        for target, waveform in sampled_sequences.items():
            if self.experiment_system.get_target(target).sideband != "L":
                waveform = np.conj(waveform)
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

            if ranges[0].start % WORD_LENGTH != 0:
                raise ValueError(
                    f"Capture range should start at a multiple of 4 samples ({WORD_DURATION} ns)."
                )

            delay = capture_delay_sample[target]

            cap_sub_sequence = pls.CapSampledSubSequence(
                capture_slots=[],
                repeats=None,
                prev_blank=delay,
                post_blank=None,
                original_prev_blank=0,
                original_post_blank=None,
            )

            # WORKAROUND: add an extra capture to ensure the first capture begins at a multiple of 64 samples
            post_blank_to_first_readout = ranges[0].start - EXTRA_SUM_SECTION_LENGTH
            cap_sub_sequence.capture_slots.append(
                pls.CaptureSlots(
                    duration=EXTRA_SUM_SECTION_LENGTH,
                    post_blank=post_blank_to_first_readout,
                    original_duration=None,  # type: ignore
                    original_post_blank=None,  # type: ignore
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
            add_last_measurement=add_last_measurement,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )

        backend_interval = (
            math.ceil((schedule.duration + interval) / BLOCK_DURATION) * BLOCK_DURATION
        )

        targets = list(gen_sequences.keys() | cap_sequences.keys())
        resource_map = self.device_controller.get_resource_map(targets)

        return SequencerMod(
            gen_sampled_sequence=gen_sequences,
            cap_sampled_sequence=cap_sequences,
            resource_map=resource_map,  # type: ignore
            interval=backend_interval,
            sysdb=self.device_controller.qubecalib.sysdb,
            driver=self.device_controller.quel1system,
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
                            raw=iq * norm_factor,
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
            config=self.device_controller.box_config,
        )

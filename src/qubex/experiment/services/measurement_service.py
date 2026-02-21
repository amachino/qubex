"""Measurement service for experiment execution and analysis."""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from collections.abc import Collection, Sequence
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import qctrlvisualizer as qcv
from numpy.typing import ArrayLike, NDArray
from qxpulse import (
    Arbitrary,
    Blank,
    FlatTop,
    PhaseShift,
    Pulse,
    PulseArray,
    PulseSchedule,
    RampType,
    Waveform,
)
from rich.console import Console
from tqdm import tqdm

import qubex.visualization as viz
from qubex.analysis import IQPlotter, fitting
from qubex.analysis.state_tomography import (
    mle_fit_density_matrix,
    plot_ghz_state_tomography,
)
from qubex.backend import TargetRegistry
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    CLASSIFIER_DIR,
    DEFAULT_INTERVAL,
    DEFAULT_RABI_TIME_RANGE,
    DEFAULT_SHOTS,
    HPI_DURATION,
    HPI_RAMPTIME,
)
from qubex.experiment.experiment_context import ExperimentContext
from qubex.experiment.models.experiment_result import (
    ExperimentResult,
    RabiData,
    SweepData,
)
from qubex.experiment.models.rabi_param import RabiParam
from qubex.experiment.models.result import Result
from qubex.measurement import (
    MeasureResult,
    MultipleMeasureResult,
    StateClassifier,
    StateClassifierGMM,
    StateClassifierKMeans,
)
from qubex.typing import (
    IQArray,
    MeasurementMode,
    ParametricPulseSchedule,
    ParametricWaveformDict,
    TargetMap,
)

from .pulse_service import PulseService

logger = logging.getLogger(__name__)

console = Console()


class MeasurementService:
    """Service for running measurement routines."""

    def __init__(
        self,
        *,
        experiment_context: ExperimentContext,
        pulse_service: PulseService,
    ):
        self._ctx = experiment_context
        self._pulse_service = pulse_service

    @property
    def ctx(self) -> ExperimentContext:
        """Return the experiment context."""
        return self._ctx

    @property
    def pulse(self) -> PulseService:
        """Return the pulse service."""
        return self._pulse_service

    @staticmethod
    def unique_in_order(labels: Sequence[str]) -> list[str]:
        """Return labels de-duplicated while preserving first appearance order."""
        return list(dict.fromkeys(labels))

    @staticmethod
    def ordered_qubit_labels(labels: Sequence[str]) -> list[str]:
        """Return qubit labels in first appearance order from target labels."""
        fallback_registry = TargetRegistry()
        return MeasurementService.unique_in_order(
            [
                fallback_registry.resolve_qubit_label(label, allow_legacy=True)
                for label in labels
            ]
        )

    def check_noise(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: int | None = None,
        plot: bool | None = None,
    ) -> MeasureResult:
        """Measure noise for the specified targets."""
        if duration is None:
            duration = 10240
        if plot is None:
            plot = True

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        result = self.ctx.measurement.measure_noise(
            targets,
            duration=duration,
            enable_dsp_sum=False,
        )
        for data in result.data.values():
            if plot:
                data.plot()
        return result

    def execute(
        self,
        schedule: PulseSchedule,
        *,
        frequencies: dict[str, float] | None = None,
        mode: MeasurementMode | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        add_last_measurement: bool | None = None,
        add_pump_pulses: bool | None = None,
        enable_dsp_demodulation: bool | None = None,
        enable_dsp_sum: bool | None = None,
        enable_dsp_classification: bool | None = None,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
        reset_awg_and_capunits: bool | None = None,
        plot: bool | None = None,
    ) -> MultipleMeasureResult:
        """Execute a schedule and return multiple measurement results."""
        if mode is None:
            mode = "avg"
        if add_last_measurement is None:
            add_last_measurement = False
        if add_pump_pulses is None:
            add_pump_pulses = False
        if enable_dsp_demodulation is None:
            enable_dsp_demodulation = True
        if enable_dsp_classification is None:
            enable_dsp_classification = False
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if plot is None:
            plot = False

        if readout_duration is None:
            readout_duration = self.pulse.readout_duration
        if readout_pre_margin is None:
            readout_pre_margin = self.pulse.readout_pre_margin
        if readout_post_margin is None:
            readout_post_margin = self.pulse.readout_post_margin
        if enable_dsp_sum is None:
            enable_dsp_sum = mode == "single"

        if reset_awg_and_capunits:
            qubits = {
                self.ctx.resolve_qubit_label(target) for target in schedule.labels
            }
            self.ctx.reset_awg_and_capunits(qubits=qubits)

        with self.ctx.modified_frequencies(frequencies):
            result = self.ctx.measurement.execute(
                schedule=schedule,
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
                add_last_measurement=add_last_measurement,
                add_pump_pulses=add_pump_pulses,
                enable_dsp_demodulation=enable_dsp_demodulation,
                enable_dsp_sum=enable_dsp_sum,
                enable_dsp_classification=enable_dsp_classification,
                line_param0=line_param0,
                line_param1=line_param1,
                plot=plot,
            )

        if plot:
            result.plot()
        return result

    def measure(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        frequencies: dict[str, float] | None = None,
        initial_states: dict[str, str] | None = None,
        mode: MeasurementMode | None = None,
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
        enable_dsp_sum: bool | None = None,
        enable_dsp_classification: bool | None = None,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
        reset_awg_and_capunits: bool | None = None,
        plot: bool | None = None,
    ) -> MeasureResult:
        """Measure a sequence or schedule and return results."""
        if mode is None:
            mode = "avg"
        if add_pump_pulses is None:
            add_pump_pulses = False
        if enable_dsp_demodulation is None:
            enable_dsp_demodulation = True
        if enable_dsp_classification is None:
            enable_dsp_classification = False
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if plot is None:
            plot = False

        if readout_duration is None:
            readout_duration = self.pulse.readout_duration
        if readout_pre_margin is None:
            readout_pre_margin = self.pulse.readout_pre_margin
        if readout_post_margin is None:
            readout_post_margin = self.pulse.readout_post_margin

        waveforms: dict[str, NDArray[np.complex128]] = {}

        if enable_dsp_sum is None:
            enable_dsp_sum = mode == "single"

        if isinstance(sequence, PulseSchedule):
            if not sequence.is_valid():
                raise ValueError("Invalid pulse schedule.")

            if initial_states is not None:
                labels = self.unique_in_order(
                    [*sequence.labels, *initial_states.keys()]
                )
                with PulseSchedule(labels) as ps:
                    for target, state in initial_states.items():
                        if target in self.ctx.qubit_labels:
                            ps.add(
                                target, self.pulse.get_pulse_for_state(target, state)
                            )
                        else:
                            raise ValueError(f"Invalid init target: {target}")
                    ps.barrier()
                    ps.call(sequence)
                waveforms = ps.get_sampled_sequences()
            else:
                waveforms = sequence.get_sampled_sequences()
        else:
            if initial_states is not None:
                labels = self.unique_in_order(
                    [*sequence.keys(), *initial_states.keys()]
                )
                with PulseSchedule(labels) as ps:
                    for target, state in initial_states.items():
                        if target in self.ctx.qubit_labels:
                            ps.add(
                                target, self.pulse.get_pulse_for_state(target, state)
                            )
                        else:
                            raise ValueError(f"Invalid init target: {target}")
                    ps.barrier()
                    for target, waveform in sequence.items():
                        if isinstance(waveform, Waveform):
                            ps.add(target, waveform)
                        else:
                            ps.add(target, Arbitrary(waveform))
                waveforms = ps.get_sampled_sequences()
            else:
                for target, waveform in sequence.items():
                    if isinstance(waveform, Waveform):
                        waveforms[target] = waveform.values
                    else:
                        waveforms[target] = np.array(waveform, dtype=np.complex128)

        if reset_awg_and_capunits:
            qubits = {self.ctx.resolve_qubit_label(target) for target in waveforms}
            self.ctx.reset_awg_and_capunits(qubits=qubits)

        with self.ctx.modified_frequencies(frequencies):
            result = self.ctx.measurement.measure(
                waveforms=waveforms,
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
                add_pump_pulses=add_pump_pulses,
                enable_dsp_demodulation=enable_dsp_demodulation,
                enable_dsp_sum=enable_dsp_sum,
                enable_dsp_classification=enable_dsp_classification,
                line_param0=line_param0,
                line_param1=line_param1,
            )
        if plot:
            result.plot()
        return result

    def measure_state(
        self,
        states: dict[
            str, Literal["0", "1", "+", "-", "+i", "-i"] | Literal["g", "e", "f"]
        ],
        *,
        mode: MeasurementMode | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool | None = None,
        plot: bool | None = None,
    ) -> MeasureResult:
        """Prepare given states and measure readout results."""
        if mode is None:
            mode = "single"
        if add_pump_pulses is None:
            add_pump_pulses = False
        if plot is None:
            plot = False

        targets = []

        for target, state in states.items():
            targets.append(target)
            if state == "f":
                targets.append(self.ctx.resolve_ef_label(target))

        with PulseSchedule(targets) as ps:
            for target, state in states.items():
                if state in ["0", "1", "+", "-", "+i", "-i"]:
                    ps.add(target, self.pulse.get_pulse_for_state(target, state))
                elif state == "g":
                    ps.add(target, Blank(0))
                elif state == "e":
                    ps.add(target, self.pulse.get_hpi_pulse(target).repeated(2))
                elif state == "f":
                    ps.add(target, self.pulse.get_hpi_pulse(target).repeated(2))
                    ps.barrier()
                    ef_label = self.ctx.resolve_ef_label(target)
                    ps.add(ef_label, self.pulse.get_hpi_pulse(ef_label).repeated(2))

        return self.measure(
            sequence=ps,
            mode=mode,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )

    def measure_idle_states(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Measure idle states for targets."""
        if add_pump_pulses is None:
            add_pump_pulses = False
        if plot is None:
            plot = True

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        result = self.measure_state(
            states=dict.fromkeys(targets, "g"),
            mode="single",
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            add_pump_pulses=add_pump_pulses,
            plot=False,
        )
        data = {target: result.data[target].kerneled for target in targets}
        counts = {
            target: self.ctx.classifiers[target].classify(
                target,
                data[target],
                plot=plot,
            )
            for target in targets
        }

        return Result(
            data={
                "data": data,
                "counts": counts,
            }
        )

    def obtain_reference_points(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        interval: float | None = None,
        store_reference_points: bool | None = None,
    ) -> Result:
        """Obtain and optionally store reference IQ points."""
        if store_reference_points is None:
            store_reference_points = True

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if shots is None:
            shots = 10000

        result = self.measure_state(
            dict.fromkeys(targets, "g"),
            mode="avg",
            shots=shots,
            interval=interval,
        )

        iq = {
            target: complex(measure_data.kerneled)
            for target, measure_data in result.data.items()
        }
        phase = {target: float(np.angle(v)) for target, v in iq.items()}
        amplitude = {target: float(np.abs(v)) for target, v in iq.items()}

        if store_reference_points:
            self.ctx.calib_note.reference_phases.update(phase)

        return Result(
            data={
                "iq": iq,
                "phase": phase,
                "amplitude": amplitude,
            }
        )

    def sweep_parameter(
        self,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        *,
        sweep_range: ArrayLike,
        repetitions: int | None = None,
        frequencies: dict[str, float] | None = None,
        initial_states: dict[str, str] | None = None,
        rabi_level: Literal["ge", "ef"] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        plot: bool | None = None,
        enable_tqdm: bool | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        yaxis_type: Literal["linear", "log"] | None = None,
    ) -> ExperimentResult[SweepData]:
        """
        Sweep a parameter and measure results.

        Parameters
        ----------
        sequence
            Parametric schedule or waveform map to evaluate.
        sweep_range
            Values to sweep over.
        repetitions
            Number of repetitions for each sweep point.
        """
        if repetitions is None:
            repetitions = 1
        if rabi_level is None:
            rabi_level = "ge"
        if plot is None:
            plot = True
        if enable_tqdm is None:
            enable_tqdm = False
        if title is None:
            title = "Sweep result"
        if xlabel is None:
            xlabel = "Sweep value"
        if ylabel is None:
            ylabel = "Measured value"
        if xaxis_type is None:
            xaxis_type = "linear"
        if yaxis_type is None:
            yaxis_type = "linear"

        sweep_range = np.array(sweep_range)

        if rabi_level == "ge":
            rabi_params = self.pulse.ge_rabi_params
        elif rabi_level == "ef":
            rabi_params = self.pulse.ef_rabi_params
        else:
            raise ValueError("Invalid Rabi level.")

        if callable(sequence):
            initial_sequence = sequence(sweep_range[0])
            if isinstance(initial_sequence, PulseSchedule):
                sequences = [
                    sequence(param)
                    .repeated(repetitions)  # type: ignore
                    .get_sampled_sequences(copy=False)
                    for param in sweep_range
                ]
                ordered_qubits = self.ctx.ordered_qubit_labels(initial_sequence.labels)
            elif isinstance(initial_sequence, dict):
                sequences = [
                    {
                        target: waveform.repeated(repetitions).values
                        for target, waveform in sequence(param).items()  # type: ignore
                    }
                    for param in sweep_range
                ]
                ordered_qubits = self.ctx.ordered_qubit_labels(list(initial_sequence))
            else:
                raise TypeError("Invalid sequence.")
        else:
            raise TypeError("Invalid sequence.")

        signals: dict[str, list[object]] = {qubit: [] for qubit in ordered_qubits}
        plotter = IQPlotter(
            {
                qubit: self.ctx.state_centers[qubit]
                for qubit in ordered_qubits
                if qubit in self.ctx.state_centers
            }
        )

        # initialize awgs and capture units
        self.ctx.reset_awg_and_capunits(qubits=set(ordered_qubits))

        with self.ctx.modified_frequencies(frequencies):
            for seq in tqdm(
                sequences,
                desc="Sweeping parameters",
                disable=not enable_tqdm,
            ):
                result = self.measure(
                    seq,
                    initial_states=initial_states,
                    mode="avg",
                    shots=shots,
                    interval=interval,
                    readout_amplitudes=readout_amplitudes,
                    readout_duration=readout_duration,
                    readout_pre_margin=readout_pre_margin,
                    readout_post_margin=readout_post_margin,
                    reset_awg_and_capunits=False,
                )
                for target in ordered_qubits:
                    if target in result.data:
                        signals[target].append(result.data[target].kerneled)
                for target, data in result.data.items():
                    if target in signals:
                        continue
                    signals[target] = [data.kerneled]
                if plot:
                    plotter.update(
                        {
                            target: np.asarray(values)
                            for target, values in signals.items()
                            if values
                        }
                    )

        if plot:
            plotter.show()

        sweep_data = {
            target: SweepData(
                target=target,
                data=np.array(values),
                sweep_range=sweep_range,
                rabi_param=rabi_params.get(target),
                state_centers=self.ctx.state_centers.get(target),
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                xaxis_type=xaxis_type,
                yaxis_type=yaxis_type,
            )
            for target, values in signals.items()
            if values
        }
        result = ExperimentResult(
            data=sweep_data,
            rabi_params=self.pulse.rabi_params,
        )
        return result

    def sweep_measurement(
        self,
        sequence: ParametricPulseSchedule,
        *,
        sweep_range: ArrayLike,
        frequencies: dict[str, float] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_last_measurement: bool | None = None,
        plot: bool | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        yaxis_type: Literal["linear", "log"] | None = None,
    ) -> ExperimentResult[SweepData]:
        """
        Run a sweep measurement for the provided sequence.

        Parameters
        ----------
        sequence
            Parametric pulse schedule to sweep.
        sweep_range
            Values to sweep over.
        add_last_measurement
            Whether to append a measurement at the end.
        """
        # TODO: Support ParametricWaveformDict and replace the sweep_parameter method
        if add_last_measurement is None:
            add_last_measurement = True
        if plot is None:
            plot = True
        if title is None:
            title = "Sweep result"
        if xlabel is None:
            xlabel = "Sweep value"
        if ylabel is None:
            ylabel = "Measured value"
        if xaxis_type is None:
            xaxis_type = "linear"
        if yaxis_type is None:
            yaxis_type = "linear"

        sweep_range = np.array(sweep_range)

        rabi_params = self.pulse.ge_rabi_params

        initial_sequence = sequence(sweep_range[0])
        ordered_targets = self.unique_in_order(list(initial_sequence.labels))
        ordered_qubits = self.ctx.ordered_qubit_labels(ordered_targets)
        signals: dict[str, list[object]] = {target: [] for target in ordered_targets}
        plotter = IQPlotter(
            {
                qubit: self.ctx.state_centers[qubit]
                for qubit in ordered_qubits
                if qubit in self.ctx.state_centers
            }
        )

        # initialize awgs and capture units
        self.ctx.reset_awg_and_capunits(qubits=set(ordered_qubits))

        with self.ctx.modified_frequencies(frequencies):
            for param in sweep_range:
                result = self.execute(
                    sequence(param),
                    mode="avg",
                    shots=shots,
                    interval=interval,
                    readout_amplitudes=readout_amplitudes,
                    readout_duration=readout_duration,
                    readout_pre_margin=readout_pre_margin,
                    readout_post_margin=readout_post_margin,
                    reset_awg_and_capunits=False,
                    add_last_measurement=add_last_measurement,
                )
                for target in ordered_targets:
                    if target in result.data:
                        signals[target].append(result.data[target][-1].kerneled)
                for target, data in result.data.items():
                    if target in signals:
                        continue
                    signals[target] = [data[-1].kerneled]
                if plot:
                    plotter.update(
                        {
                            target: np.asarray(values)
                            for target, values in signals.items()
                            if values
                        }
                    )

        if plot:
            plotter.show()

        sweep_data = {
            target: SweepData(
                target=target,
                data=np.array(values),
                sweep_range=sweep_range,
                rabi_param=rabi_params.get(target),
                state_centers=self.ctx.state_centers.get(target),
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                xaxis_type=xaxis_type,
                yaxis_type=yaxis_type,
            )
            for target, values in signals.items()
            if values
        }
        result = ExperimentResult(
            data=sweep_data,
            rabi_params=self.pulse.rabi_params,
        )
        return result

    def repeat_sequence(
        self,
        sequence: TargetMap[Waveform] | PulseSchedule,
        *,
        initial_states: dict[str, str] | None = None,
        repetitions: int | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
    ) -> ExperimentResult[SweepData]:
        """
        Measure repeated sequences across repetition counts.

        Parameters
        ----------
        sequence
            Sequence to repeat.
        repetitions
            Maximum repetition count used for the sweep.
        """
        if repetitions is None:
            repetitions = 20
        if plot is None:
            plot = True

        def repeated_sequence(N: int) -> PulseSchedule:
            if isinstance(sequence, dict):
                with PulseSchedule() as ps:
                    for target, pulse in sequence.items():
                        ps.add(target, pulse.repeated(N))
            elif isinstance(sequence, PulseSchedule):
                ps = sequence.repeated(N)
            else:
                raise TypeError("Invalid sequence.")
            return ps

        result = self.sweep_parameter(
            sequence=repeated_sequence,
            sweep_range=np.arange(repetitions + 1),
            initial_states=initial_states,
            shots=shots,
            interval=interval,
            plot=plot,
            xlabel="Number of repetitions",
        )

        if plot:
            result.plot(normalize=True)

        return result

    def obtain_rabi_params(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        amplitudes: dict[str, float] | None = None,
        frequencies: dict[str, float] | None = None,
        is_damped: bool | None = None,
        fit_threshold: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
        store_params: bool | None = None,
        simultaneous: bool | None = None,
    ) -> ExperimentResult[RabiData]:
        """
        Estimate Rabi parameters for the specified targets.

        Parameters
        ----------
        targets
            Target qubits to characterize.
        time_range
            Time sweep range for the Rabi experiment.
        simultaneous
            Whether to perform a simultaneous experiment.
        """
        if time_range is None:
            time_range = DEFAULT_RABI_TIME_RANGE
        if is_damped is None:
            is_damped = True
        if fit_threshold is None:
            fit_threshold = 0.5
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True
        if store_params is None:
            store_params = True
        if simultaneous is None:
            simultaneous = False

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        time_range = np.asarray(time_range)

        if ramptime is None:
            ramptime = HPI_DURATION - HPI_RAMPTIME  # π/2

        if amplitudes is None:
            ampl = self.ctx.params.control_amplitude
            amplitudes = {target: ampl[target] for target in targets}

        if simultaneous:
            result = self.rabi_experiment(
                amplitudes=amplitudes,
                time_range=time_range,
                ramptime=ramptime,
                frequencies=frequencies,
                is_damped=is_damped,
                fit_threshold=fit_threshold,
                shots=shots,
                interval=interval,
                plot=plot,
                store_params=store_params,
            )
        else:
            rabi_data = {}
            rabi_params = {}
            for target in targets:
                data = self.rabi_experiment(
                    amplitudes={target: amplitudes[target]},
                    time_range=time_range,
                    ramptime=ramptime,
                    frequencies=frequencies,
                    is_damped=is_damped,
                    fit_threshold=fit_threshold,
                    shots=shots,
                    interval=interval,
                    store_params=store_params,
                    plot=plot,
                ).data[target]
                rabi_data[target] = data
                rabi_params[target] = data.rabi_param
            result = ExperimentResult(
                data=rabi_data,
                rabi_params=rabi_params,
            )
        return result

    def obtain_ef_rabi_params(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        is_damped: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
    ) -> ExperimentResult[RabiData]:
        """
        Estimate EF Rabi parameters for the specified targets.

        Parameters
        ----------
        targets
            Target qubits to characterize.
        time_range
            Time sweep range for the EF Rabi experiment.
        """
        # TODO: Integrate with obtain_rabi_params
        if time_range is None:
            time_range = DEFAULT_RABI_TIME_RANGE
        if is_damped is None:
            is_damped = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        time_range = np.asarray(time_range)

        if ramptime is None:
            ramptime = HPI_DURATION - HPI_RAMPTIME

        ef_labels = [self.ctx.resolve_ef_label(target) for target in targets]
        ef_targets = [self.ctx.targets[ef] for ef in ef_labels]

        amplitudes = {
            ef.label: self.ctx.params.get_ef_control_amplitude(ef.qubit)
            for ef in ef_targets
        }

        rabi_data = {}
        rabi_params = {}
        for label in ef_labels:
            data = self.ef_rabi_experiment(
                amplitudes={label: amplitudes[label]},
                time_range=time_range,
                ramptime=ramptime,
                is_damped=is_damped,
                shots=shots,
                interval=interval,
                store_params=True,
                plot=plot,
            ).data[label]
            rabi_data[label] = data
            rabi_params[label] = data.rabi_param

        result = ExperimentResult(
            data=rabi_data,
            rabi_params=rabi_params,
        )
        return result

    def check_waveform(
        self,
        targets: Collection[str] | str | None = None,
        *,
        method: Literal["measure", "execute"] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitude: float | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool | None = None,
        plot: bool | None = None,
    ) -> MeasureResult | MultipleMeasureResult:
        """
        Check the readout waveforms of the given targets.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the waveforms.
        method : Literal["measure", "execute"] | None, optional
            Deprecated selector for waveform-check execution path.
            Passing this argument emits `DeprecationWarning`.
        shots : int, optional
            Number of shots.
        interval : int, optional
            Interval between shots.
        readout_amplitude : float, optional
            Amplitude of the readout pulse.
        readout_duration : float, optional
            Duration of the readout pulse in ns.
        readout_pre_margin : float, optional
            Pre-margin of the readout pulse in ns.
        readout_post_margin : float, optional
            Post-margin of the readout pulse in ns.
        add_pump_pulses : bool, optional
            Whether to add pump pulses to the readout sequence. Defaults to False.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.check_waveform(["Q00", "Q01"])
        """
        if method is not None:
            warnings.warn(
                "`check_waveform(..., method=...)` is deprecated and will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        resolved_method = "measure" if method is None else method
        if add_pump_pulses is None:
            add_pump_pulses = False
        if plot is None:
            plot = True

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if readout_amplitude is not None:
            readout_targets = [
                self.ctx.resolve_read_label(target) for target in targets
            ]
            readout_amplitudes = dict.fromkeys(readout_targets, readout_amplitude)
        else:
            readout_amplitudes = None

        with PulseSchedule() as ps:
            for target in targets:
                ps.add(target, Blank(0))

        if resolved_method == "measure":
            result = self.measure(
                ps,
                shots=shots,
                interval=interval,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                add_pump_pulses=add_pump_pulses,
                enable_dsp_sum=False,
            )
        else:
            result = self.execute(
                ps,
                shots=shots,
                interval=interval,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                add_pump_pulses=add_pump_pulses,
                add_last_measurement=True,
                enable_dsp_sum=False,
            )
        if plot:
            result.plot()
        return result

    def check_rabi(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        shots: int | None = None,
        interval: float | None = None,
        store_params: bool | None = None,
        rabi_level: Literal["ge", "ef"] | None = None,
        plot: bool | None = None,
    ) -> ExperimentResult[RabiData]:
        """
        Check the Rabi oscillation of the given targets.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the Rabi oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns. Defaults to RABI_TIME_RANGE.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        store_params : bool, optional
            Whether to store the Rabi parameters. Defaults to False.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[RabiData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.check_rabi(["Q00", "Q01"])
        """
        if time_range is None:
            time_range = DEFAULT_RABI_TIME_RANGE
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if store_params is None:
            store_params = False
        if rabi_level is None:
            rabi_level = "ge"
        if plot is None:
            plot = True

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)
        time_range = np.asarray(time_range)
        amplitudes = {
            target: self.ctx.params.get_control_amplitude(target) for target in targets
        }
        if rabi_level == "ge":
            result = self.rabi_experiment(
                amplitudes=amplitudes,
                time_range=time_range,
                shots=shots,
                interval=interval,
                store_params=store_params,
                plot=plot,
            )
        elif rabi_level == "ef":
            result = self.ef_rabi_experiment(
                amplitudes=amplitudes,
                time_range=time_range,
                shots=shots,
                interval=interval,
                store_params=store_params,
                plot=plot,
            )
        return result

    def rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool | None = None,
        fit_threshold: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
        store_params: bool | None = None,
    ) -> ExperimentResult[RabiData]:
        """
        Run a GE Rabi experiment and fit parameters.

        Parameters
        ----------
        amplitudes
            Drive amplitudes per target.
        time_range
            Drive durations used for the sweep.
        detuning
            Optional detuning applied to target frequencies.
        """
        if time_range is None:
            time_range = DEFAULT_RABI_TIME_RANGE
        if is_damped is None:
            is_damped = True
        if fit_threshold is None:
            fit_threshold = 0.5
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True
        if store_params is None:
            store_params = False

        # target labels
        targets = list(amplitudes.keys())

        # drive time range
        time_range = np.array(time_range, dtype=np.float64)

        if ramptime is None:
            ramptime = 0.0

        effective_time_range = time_range + ramptime

        # measure ground states as reference points
        reference_points = self.obtain_reference_points(targets)["iq"]

        # target frequencies
        if frequencies is None:
            frequencies = {
                target: self.ctx.targets[target].frequency for target in amplitudes
            }

        # rabi sequence with rect pulses of duration T
        def rabi_sequence(T: float) -> PulseSchedule:
            with PulseSchedule(targets) as ps:
                for target in targets:
                    ps.add(
                        target,
                        FlatTop(
                            duration=T + 2 * ramptime,
                            amplitude=amplitudes[target],
                            tau=ramptime,
                        ),
                    )
            return ps

        # detune target frequencies if necessary
        if detuning is not None:
            frequencies = {
                target: frequencies[target] + detuning for target in amplitudes
            }

        # run the Rabi experiment by sweeping the drive time
        sweep_result = self.sweep_parameter(
            sequence=rabi_sequence,
            sweep_range=time_range,
            frequencies=frequencies,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        # sweep data with the target labels
        sweep_data = sweep_result.data

        # fit the Rabi oscillation
        rabi_params = {}
        for target, data in sweep_data.items():
            fit_result = fitting.fit_rabi(
                target=data.target,
                times=effective_time_range,
                data=data.data,
                reference_point=reference_points.get(target),
                plot=plot,
                is_damped=is_damped,
            )
            if fit_result["status"] == "error" or fit_result["r2"] < fit_threshold:
                rabi_params[target] = RabiParam.nan(target=target)
            else:
                rabi_params[target] = RabiParam(
                    target=target,
                    amplitude=fit_result["amplitude"],
                    frequency=fit_result["frequency"],
                    phase=fit_result["phase"],
                    offset=fit_result["offset"],
                    noise=fit_result["noise"],
                    angle=fit_result["angle"],
                    distance=fit_result["distance"],
                    r2=fit_result["r2"],
                    reference_phase=fit_result["reference_phase"],
                )

        # store the Rabi parameters if necessary
        if store_params:
            self.ctx.store_rabi_params(rabi_params)

        # create the Rabi data for each target
        rabi_data = {
            target: RabiData(
                target=target,
                data=data.data,
                time_range=effective_time_range,
                rabi_param=rabi_params[target],
                state_centers=self.ctx.state_centers.get(target),
            )
            for target, data in sweep_data.items()
        }

        # create the experiment result
        result = ExperimentResult(
            data=rabi_data,
            rabi_params=rabi_params,
        )

        # return the result
        return result

    def ef_rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike,
        ramptime: float | None = None,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
        store_params: bool | None = None,
    ) -> ExperimentResult[RabiData]:
        """
        Run an EF Rabi experiment and fit parameters.

        Parameters
        ----------
        amplitudes
            EF drive amplitudes per target.
        time_range
            Drive durations used for the sweep.
        detuning
            Optional detuning applied to target frequencies.
        """
        # TODO: Integrate with rabi_experiment
        if is_damped is None:
            is_damped = True
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True
        if store_params is None:
            store_params = False

        amplitudes = {
            self.ctx.resolve_ef_label(label): amplitude
            for label, amplitude in amplitudes.items()
        }
        ge_labels = [self.ctx.resolve_ge_label(label) for label in amplitudes]
        ef_labels = [self.ctx.resolve_ef_label(label) for label in amplitudes]

        # drive time range
        time_range = np.array(time_range, dtype=np.float64)

        if ramptime is None:
            ramptime = 0.0

        effective_time_range = time_range + ramptime

        # target frequencies
        if frequencies is None:
            frequencies = {
                target: self.ctx.targets[target].frequency for target in amplitudes
            }

        # ef rabi sequence with rect pulses of duration T
        def ef_rabi_sequence(T: int) -> PulseSchedule:
            with PulseSchedule() as ps:
                # prepare qubits to the excited state
                for ge in ge_labels:
                    ps.add(ge, self.pulse.x180(ge))
                ps.barrier()
                # apply the ef drive to induce the ef Rabi oscillation
                for ef in ef_labels:
                    ps.add(
                        ef,
                        FlatTop(
                            duration=T + 2 * ramptime,
                            amplitude=amplitudes[ef],
                            tau=ramptime,
                        ),
                    )
            return ps

        # detune target frequencies if necessary
        if detuning is not None:
            frequencies = {
                target: frequencies[target] + detuning for target in amplitudes
            }

        # run the Rabi experiment by sweeping the drive time
        sweep_result = self.sweep_parameter(
            sequence=ef_rabi_sequence,
            sweep_range=time_range,
            frequencies=frequencies,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        # fit the Rabi oscillation
        ef_rabi_params = {}
        ef_rabi_data = {}
        for qubit, data in sweep_result.data.items():
            ef_label = self.ctx.resolve_ef_label(qubit)
            ge_rabi_param = self.pulse.ge_rabi_params[qubit]
            iq_e = ge_rabi_param.endpoints[1]
            fit_result = fitting.fit_rabi(
                target=qubit,
                times=effective_time_range,
                data=data.data,
                reference_point=iq_e,
                plot=plot,
                is_damped=is_damped,
            )
            if fit_result["status"] != "success":
                ef_rabi_params[ef_label] = RabiParam.nan(target=ef_label)
            else:
                ef_rabi_params[ef_label] = RabiParam(
                    target=ef_label,
                    amplitude=fit_result["amplitude"],
                    frequency=fit_result["frequency"],
                    phase=fit_result["phase"],
                    offset=fit_result["offset"],
                    noise=fit_result["noise"],
                    angle=fit_result["angle"],
                    distance=fit_result["distance"],
                    r2=fit_result["r2"],
                    reference_phase=fit_result["reference_phase"],
                )
            ef_rabi_data[ef_label] = RabiData(
                target=ef_label,
                data=data.data,
                time_range=effective_time_range,
                rabi_param=ef_rabi_params[ef_label],
            )

        # store the Rabi parameters if necessary
        if store_params:
            self.ctx.store_rabi_params(ef_rabi_params)

        # create the experiment result
        result = ExperimentResult(
            data=ef_rabi_data,
            rabi_params=ef_rabi_params,
        )

        # return the result
        return result

    def measure_state_distribution(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: Literal[2, 3] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool | None = None,
        plot: bool | None = None,
    ) -> list[MeasureResult]:
        """
        Measure distributions for prepared basis states.

        Parameters
        ----------
        targets
            Target qubits to measure.
        n_states
            Number of states to prepare (2 or 3).
        plot
            Whether to plot IQ distributions.
        """
        if n_states is None:
            n_states = 2
        if add_pump_pulses is None:
            add_pump_pulses = False
        if plot is None:
            plot = True
        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        states = ["g", "e", "f"][:n_states]
        result = {
            state: self.measure_state(
                dict.fromkeys(targets, state),  # type: ignore
                shots=shots,
                interval=interval,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                add_pump_pulses=add_pump_pulses,
            )
            for state in states
        }
        for target in targets:
            data = {
                f"|{state}⟩": result[state].data[target].kerneled for state in states
            }
            if plot:
                viz.scatter_iq_data(
                    data=data,
                    title=f"State distribution : {target}",
                )
        return list(result.values())

    def build_classifier(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: Literal[2, 3] | None = None,
        save_classifier: bool | None = None,
        save_dir: Path | str | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool | None = None,
        simultaneous: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """
        Build state classifiers from measured distributions.

        Parameters
        ----------
        targets
            Target qubits used for classifier training.
        n_states
            Number of basis states to use.
        save_classifier
            Whether to save trained classifiers.
        """
        if save_classifier is None:
            save_classifier = True
        if add_pump_pulses is None:
            add_pump_pulses = False
        if simultaneous is None:
            simultaneous = False
        if plot is None:
            plot = True
        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if simultaneous:
            return self._build_classifier(
                targets=targets,
                n_states=n_states,
                save_classifier=save_classifier,
                save_dir=save_dir,
                shots=shots,
                interval=interval,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                add_pump_pulses=add_pump_pulses,
                plot=plot,
            )
        else:
            fidelities = {}
            average_fidelities = {}
            data = {}
            classifiers = {}
            for target in targets:
                result = self._build_classifier(
                    targets=target,
                    n_states=n_states,
                    save_classifier=save_classifier,
                    save_dir=save_dir,
                    shots=shots,
                    interval=interval,
                    readout_amplitudes=readout_amplitudes,
                    readout_duration=readout_duration,
                    readout_pre_margin=readout_pre_margin,
                    readout_post_margin=readout_post_margin,
                    add_pump_pulses=add_pump_pulses,
                    plot=plot,
                )
                fidelities[target] = result["readout_fidelities"][target]
                average_fidelities[target] = result["average_readout_fidelity"][target]
                data[target] = result["data"]
                classifiers[target] = result["classifiers"][target]

            return Result(
                data={
                    "readout_fidelities": fidelities,
                    "average_readout_fidelity": average_fidelities,
                    "data": data,
                    "classifiers": classifiers,
                }
            )

    def _build_classifier(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: Literal[2, 3] | None = None,
        save_classifier: bool | None = None,
        save_dir: Path | str | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        if save_classifier is None:
            save_classifier = True
        if add_pump_pulses is None:
            add_pump_pulses = False
        if plot is None:
            plot = True
        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)
        if n_states is None:
            n_states = 2
        if shots is None:
            shots = 10000

        self.obtain_reference_points(targets)

        results = self.measure_state_distribution(
            targets=targets,
            n_states=n_states,
            shots=shots,
            interval=interval,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_duration=readout_duration,
            readout_amplitudes=readout_amplitudes,
            add_pump_pulses=add_pump_pulses,
            plot=False,
        )

        data = {
            target: {
                state: result.data[target].kerneled
                for state, result in enumerate(results)
            }
            for target in targets
        }

        classifiers: TargetMap[StateClassifier]
        if self.ctx.classifier_type == "kmeans":
            classifiers = {
                target: StateClassifierKMeans.fit(data[target]) for target in targets
            }
        elif self.ctx.classifier_type == "gmm":
            classifiers = {
                target: StateClassifierGMM.fit(
                    data[target],
                    phase=self.ctx.reference_phases[target],
                )
                for target in targets
            }
        else:
            raise ValueError("Invalid classifier type.")
        self.ctx.measurement.update_classifiers(classifiers)

        if save_classifier:
            for label, classifier in classifiers.items():
                if save_dir is not None:
                    path = Path(save_dir) / self.ctx.chip_id / f"{label}.pkl"
                else:
                    path = Path(CLASSIFIER_DIR) / self.ctx.chip_id / f"{label}.pkl"
                if not path.parent.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                classifier.save(path)

        fidelities = {}
        average_fidelities = {}
        for target in targets:
            clf = classifiers[target]
            classified = []
            for state in range(n_states):
                if plot:
                    print(f"{target} prepared as |{state}⟩:")
                result = clf.classify(
                    target,
                    data[target][state],
                    plot=plot,
                )
                classified.append(result)
            fidelity = [
                classified[state][state] / sum(classified[state].values())
                for state in range(n_states)
            ]
            fidelities[target] = fidelity
            average_fidelities[target] = np.mean(fidelity)

            if plot:
                print(f"{target}:")
                print(f"  Total shots: {shots}")
                for state in range(n_states):
                    print(
                        f"  |{state}⟩ → {classified[state]}, f_{state}: {fidelity[state] * 100:.2f}%"
                    )
                print(
                    f"  Average readout fidelity : {average_fidelities[target] * 100:.2f}%\n\n"
                )

        self.ctx.calib_note.state_params = {
            target: {
                "target": target,
                "centers": {
                    str(state): [center.real, center.imag]
                    for state, center in classifiers[target].centers.items()
                },
                "reference_phase": self.ctx.calib_note.reference_phases[target],
            }
            for target in targets
        }

        return Result(
            data={
                "readout_fidelities": fidelities,
                "average_readout_fidelity": average_fidelities,
                "data": data,
                "classifiers": classifiers,
            }
        )

    def measure_1q_state_fidelity(
        self,
        target: str,
        *,
        target_state: str | None = None,
        waveform: Waveform | None = None,
        shots: int | None = None,
        interval: float | None = None,
        reset_awg_and_capunits: bool | None = None,
        use_zvalues: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """
        Measure fidelity of a prepared single-qubit state.

        Parameters
        ----------
        target
            Target qubit label.
        target_state
            Ideal state label for comparison.
        waveform
            Optional waveform used for state preparation.
        """
        if target_state is None:
            target_state = "+"
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if use_zvalues is None:
            use_zvalues = False
        if plot is None:
            plot = False
        if waveform is None:
            measure_result = self.state_tomography(
                sequence={target: []},
                initial_state={target: target_state},
                shots=shots,
                interval=interval,
                reset_awg_and_capunits=reset_awg_and_capunits,
                use_zvalues=use_zvalues,
                plot=plot,
            )
        else:
            measure_result = self.state_tomography(
                sequence={target: waveform},
                shots=shots,
                interval=interval,
                reset_awg_and_capunits=reset_awg_and_capunits,
                use_zvalues=use_zvalues,
                plot=plot,
            )

        state_vector = measure_result.get(target)
        if state_vector is None:
            raise ValueError(f"No measurement data for target {target}.")

        if target_state == "+":
            target_state_vector = np.array([1.0, 0.0, 0.0])
        elif target_state == "-":
            target_state_vector = np.array([-1.0, 0.0, 0.0])
        elif target_state == "+i":
            target_state_vector = np.array([0.0, 1.0, 0.0])
        elif target_state == "-i":
            target_state_vector = np.array([0.0, -1.0, 0.0])
        elif target_state == "0":
            target_state_vector = np.array([0.0, 0.0, 1.0])
        elif target_state == "1":
            target_state_vector = np.array([0.0, 0.0, -1.0])
        else:
            raise ValueError(f"Invalid target state: {target_state}")

        fidelity = np.abs(np.dot(state_vector, target_state_vector)) ** 2

        print(f"{target}: |{target_state}〉")
        print(f"  Fidelity: {fidelity:.4f}")
        print(f"  Absolute infidelity: {np.abs(1 - fidelity):.4f}")
        print(
            f"  State vector: ({state_vector[0]:.4f}, {state_vector[1]:.4f}, {state_vector[2]:.4f})"
        )
        print(f"  Target state vector: {target_state_vector}")

        return Result(
            data={
                "fidelity": fidelity,
                "absolute_infidelity": np.abs(1 - fidelity),
                "state_vector": state_vector,
                "target_state_vector": target_state_vector,
                "target_state": target_state,
                "target": target,
                "shots": shots,
                "interval": interval,
            }
        )

    def state_tomography(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        reset_awg_and_capunits: bool | None = None,
        method: Literal["measure", "execute"] | None = None,
        use_zvalues: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """
        Perform single-qubit state tomography.

        Parameters
        ----------
        sequence
            Preparation sequence or schedule.
        initial_state
            Optional state preparation per target.
        method
            Measurement method to use.
        """
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if method is None:
            method = "measure"
        if use_zvalues is None:
            use_zvalues = False
        if plot is None:
            plot = False
        if isinstance(sequence, PulseSchedule):
            sequence = sequence.get_sequences()
        else:
            sequence = {
                target: (
                    Arbitrary(waveform)
                    if not isinstance(waveform, Waveform)
                    else waveform
                )
                for target, waveform in sequence.items()
            }

        x90 = x90 or self.pulse.hpi_pulse

        buffer: dict[str, list[float]] = defaultdict(list)

        ordered_qubits = self.ctx.ordered_qubit_labels(list(sequence))
        targets = self.unique_in_order([*sequence.keys(), *ordered_qubits])

        if reset_awg_and_capunits:
            self.ctx.reset_awg_and_capunits(qubits=set(ordered_qubits))

        for basis in ["X", "Y", "Z"]:
            with PulseSchedule(targets) as ps:
                # Initialization pulses
                if initial_state is not None:
                    for qubit in ordered_qubits:
                        if qubit in initial_state:
                            init_pulse = self.pulse.get_pulse_for_state(
                                target=qubit,
                                state=initial_state[qubit],
                            )
                            ps.add(qubit, init_pulse)
                    ps.barrier()

                # Pulse sequences
                for target, waveform in sequence.items():
                    ps.add(target, waveform)
                ps.barrier()

                # Basis transformation pulses
                for qubit in ordered_qubits:
                    x90p = x90[qubit]
                    y90m = x90p.shifted(-np.pi / 2)
                    if basis == "X":
                        ps.add(qubit, y90m)
                    elif basis == "Y":
                        ps.add(qubit, x90p)

            if method == "execute":
                measure_result = self.execute(
                    ps,
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                    add_last_measurement=True,
                    plot=plot,
                )
                for qubit, data in measure_result.data.items():
                    rabi_param = self.pulse.rabi_params[qubit]
                    if rabi_param is None:
                        raise ValueError("Rabi parameters are not stored.")
                    values = data[-1].kerneled
                    values_rotated = values * np.exp(-1j * rabi_param.angle)
                    values_normalized = (
                        np.imag(values_rotated) - rabi_param.offset
                    ) / rabi_param.amplitude
                    buffer[qubit] += [values_normalized]
            else:
                measure_result = self.measure(
                    ps,
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                    plot=plot,
                )
                for qubit, data in measure_result.data.items():
                    rabi_param = self.pulse.rabi_params[qubit]
                    if rabi_param is None:
                        raise ValueError("Rabi parameters are not stored.")

                    if use_zvalues:
                        p = data.kerneled
                        g, e = (
                            self.ctx.state_centers[qubit][0],
                            self.ctx.state_centers[qubit][1],
                        )
                        v_ge = e - g
                        v_gp = p - g
                        v_gp_proj = np.real(v_gp * np.conj(v_ge)) / np.abs(v_ge)
                        values_normalized = 1 - 2 * np.abs(v_gp_proj) / np.abs(v_ge)
                    else:
                        values_normalized = float(rabi_param.normalize(data.kerneled))

                    buffer[qubit] += [values_normalized]

        result = {
            qubit: (
                values[0],  # X
                values[1],  # Y
                values[2],  # Z
            )
            for qubit, values in buffer.items()
        }
        return Result(data=result)

    def state_evolution_tomography(
        self,
        *,
        sequences: (
            Sequence[PulseSchedule]
            | Sequence[TargetMap[Waveform]]
            | Sequence[TargetMap[IQArray]]
        ),
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        reset_awg_and_capunits: bool | None = None,
        method: Literal["measure", "execute"] | None = None,
        plot: bool | None = None,
    ) -> Result:
        """
        Perform tomography over a sequence of states.

        Parameters
        ----------
        sequences
            Sequences to measure in order.
        initial_state
            Optional initial state preparation per target.
        method
            Measurement method to use.
        """
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if method is None:
            method = "measure"
        if plot is None:
            plot = True
        buffer: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

        if reset_awg_and_capunits:
            initial_sequence = sequences[0]
            if isinstance(initial_sequence, PulseSchedule):
                ordered_qubits = self.ctx.ordered_qubit_labels(initial_sequence.labels)
            else:
                ordered_qubits = self.ctx.ordered_qubit_labels(list(initial_sequence))
            self.ctx.reset_awg_and_capunits(qubits=set(ordered_qubits))

        for sequence in tqdm(sequences):
            state_vectors = self.state_tomography(
                sequence=sequence,
                x90=x90,
                initial_state=initial_state,
                shots=shots,
                interval=interval,
                reset_awg_and_capunits=False,
                method=method,
                plot=False,
            )
            for target, state_vector in state_vectors.items():
                buffer[target].append(state_vector)

        result = {target: np.array(states) for target, states in buffer.items()}

        if plot:
            for target, states in result.items():
                print(f"State evolution : {target}")
                qcv.display_bloch_sphere_from_bloch_vectors(states)

        return Result(data=result)

    def partial_waveform(self, waveform: Waveform, index: int) -> Waveform:
        """Return a partial waveform up to the given index."""
        # If the index is 0, return an empty Pulse as the initial state.
        if index == 0:
            return Arbitrary([])

        elif isinstance(waveform, Pulse):
            # If the index is greater than the waveform length, return the waveform itself.
            if index >= waveform.length:
                return waveform
            # If the index is less than the waveform length, return a partial waveform.
            else:
                return Arbitrary(waveform.values[0 : index - 1])

        # If the waveform is a PulseArray, we need to extract the partial sequence.
        elif isinstance(waveform, PulseArray):
            offset = 0
            pulse_array = PulseArray([])

            # Iterate over the objects in the PulseArray.
            for obj in waveform.elements:
                # If the object is a PhaseShift gate, we can simply add it to the array.
                if isinstance(obj, PhaseShift):
                    pulse_array.add(obj)
                    continue
                elif isinstance(obj, Pulse):
                    # If the index become equal to the offset, we can stop iterating.
                    if index == offset:
                        break
                    # If the endpoint of obj is greater than the index, add the partial Pulse and break.
                    elif obj.length > index - offset:
                        pulse = Arbitrary(obj.values[0 : index - offset])
                        pulse_array.add(pulse)
                        break
                    # If the endpoint of obj is less than or equal to the index, add the whole Pulse.
                    else:
                        pulse_array.add(obj)
                        offset += obj.length
                        # NOTE: Don't break here even offset become equal to index,
                        # because we may have PhaseShift gates after the Pulse.
                else:
                    # NOTE: PulseArray should be flattened before calling this function.
                    logger.error(f"Invalid type: {type(obj)}")
            return pulse_array
        else:
            logger.error(f"Invalid type: {type(waveform)}")
            return waveform

    def pulse_tomography(
        self,
        sequence: PulseSchedule | TargetMap[Waveform] | TargetMap[IQArray],
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        n_samples: int | None = None,
        shots: int | None = None,
        interval: float | None = None,
        method: Literal["measure", "execute"] | None = None,
        plot: bool | None = None,
    ) -> Result:
        """
        Reconstruct state evolution across a pulse waveform.

        Parameters
        ----------
        sequence
            Pulse schedule or waveform map to analyze.
        n_samples
            Number of samples taken along the waveform.
        method
            Measurement method to use.
        """
        if n_samples is None:
            n_samples = 100
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if method is None:
            method = "measure"
        if plot is None:
            plot = True
        self.pulse.validate_rabi_params()

        if isinstance(sequence, PulseSchedule):
            pulses = sequence.get_sequences()
        else:
            pulses = {}
            pulse_length_set = set()
            for target, waveform in sequence.items():
                if isinstance(waveform, Waveform):
                    pulse = waveform
                elif isinstance(waveform, Sequence):
                    pulse = Arbitrary(waveform)
                else:
                    raise TypeError("Invalid waveform.")
                pulses[target] = pulse
                pulse_length_set.add(pulse.length)
            if len(pulse_length_set) != 1:
                raise ValueError("The lengths of the waveforms must be the same.")

        pulse_length = next(iter(pulses.values())).length

        if plot:
            if isinstance(sequence, PulseSchedule):
                sequence.plot(title="Pulse sequence")
            else:
                for target in pulses:
                    pulses[target].plot(title=f"Waveform : {target}")

        if n_samples is None or pulse_length < n_samples:
            indices = np.arange(pulse_length + 1)
        else:
            indices = np.linspace(0, pulse_length, n_samples).astype(int)

        flattened_pulses = {
            target: pulse.flattened() if isinstance(pulse, PulseArray) else pulse
            for target, pulse in pulses.items()
        }

        sequences = [
            {
                target: self.partial_waveform(pulse, i)
                for target, pulse in flattened_pulses.items()
            }
            for i in indices
        ]

        result = self.state_evolution_tomography(
            sequences=sequences,
            x90=x90,
            initial_state=initial_state,
            shots=shots,
            interval=interval,
            method=method,
            plot=plot,
        )

        if plot:
            times = indices * self.ctx.measurement.sampling_period
            for target, states in result.items():
                viz.plot_bloch_vectors(
                    times=times,
                    bloch_vectors=states,
                    title=f"State evolution : {target}",
                )

        return result

    def measure_population(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        fit_gmm: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """
        Measure population probabilities for each target.

        Parameters
        ----------
        sequence
            Sequence to measure.
        fit_gmm
            Whether to estimate probabilities using GMM.
        """
        if fit_gmm is None:
            fit_gmm = False
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if self.ctx.classifiers is None:
            raise ValueError("Classifiers are not built. Run `build_classifier` first.")

        result = self.measure(
            sequence,
            mode="single",
            shots=shots,
            interval=interval,
        )
        if fit_gmm:
            probabilities = {
                target: self.ctx.classifiers[target].estimate_weights(
                    result.data[target].kerneled
                )
                for target in result.data
            }
            standard_deviations = {
                target: np.zeros_like(probabilities[target]) for target in probabilities
            }
        else:
            probabilities = {
                target: data.probabilities for target, data in result.data.items()
            }
            standard_deviations = {
                target: data.standard_deviations for target, data in result.data.items()
            }
        return probabilities, standard_deviations

    def measure_population_dynamics(
        self,
        *,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        params_list: Sequence | NDArray,
        fit_gmm: bool | None = None,
        xlabel: str | None = None,
        scatter_mode: str | None = None,
        show_error: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """
        Measure population dynamics over a parameter sweep.

        Parameters
        ----------
        sequence
            Parametric sequence evaluated for each parameter.
        params_list
            Parameter values to sweep.
        show_error
            Whether to display error bars.
        """
        if fit_gmm is None:
            fit_gmm = False
        if xlabel is None:
            xlabel = "Index"
        if scatter_mode is None:
            scatter_mode = "lines+markers"
        if show_error is None:
            show_error = True
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if isinstance(params_list[0], int):
            x = params_list
        else:
            try:
                float(params_list[0])
                x = params_list
            except ValueError:
                x = np.arange(len(params_list))

        buffer_pops = defaultdict(list)
        buffer_errs = defaultdict(list)

        for params in tqdm(params_list):
            prob_dict, err_dict = self.measure_population(
                sequence=sequence(params),
                fit_gmm=fit_gmm,
                shots=shots,
                interval=interval,
            )
            for target, probs in prob_dict.items():
                buffer_pops[target].append(probs)
            for target, errors in err_dict.items():
                buffer_errs[target].append(errors)

        result_pops = {
            target: np.array(buffer_pops[target]).T for target in buffer_pops
        }
        result_errs = {
            target: np.array(buffer_errs[target]).T for target in buffer_errs
        }

        for target in result_pops:
            fig = viz.make_figure()
            for state, probs in enumerate(result_pops[target]):
                fig.add_scatter(
                    name=f"|{state}⟩",
                    mode=scatter_mode,
                    x=x,
                    y=probs,
                    error_y=(
                        dict(
                            type="data",
                            array=result_errs[target][state],
                            visible=True,
                            thickness=1.5,
                            width=3,
                        )
                        if show_error
                        else None
                    ),
                    marker=dict(size=5),
                )
            fig.update_layout(
                title=f"Population dynamics : {target}",
                xaxis_title=xlabel,
                yaxis_title="Probability",
                yaxis_range=[-0.1, 1.1],
            )
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )

        return result_pops, result_errs

    def measure_bell_state(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        control_basis: str | None = None,
        target_basis: str | None = None,
        zx90: PulseSchedule | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
        plot_sequence: bool | None = None,
        plot_raw: bool | None = None,
        plot_mitigated: bool | None = None,
        save_image: bool | None = None,
        reset_awg_and_capunits: bool | None = None,
    ) -> Result:
        """
        Measure Bell-state probabilities in a specified basis.

        Parameters
        ----------
        control_qubit
            Control qubit label.
        target_qubit
            Target qubit label.
        control_basis
            Measurement basis for the control qubit.
        target_basis
            Measurement basis for the target qubit.
        """
        if control_basis is None:
            control_basis = "Z"
        if target_basis is None:
            target_basis = "Z"
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True
        if plot_sequence is None:
            plot_sequence = False
        if plot_raw is None:
            plot_raw = True
        if plot_mitigated is None:
            plot_mitigated = True
        if save_image is None:
            save_image = True
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if self.ctx.state_centers is None:
            self.build_classifier(plot=False)

        pair = [control_qubit, target_qubit]

        with PulseSchedule(pair) as ps:
            # prepare |+⟩|0⟩
            ps.add(control_qubit, self.pulse.y90(control_qubit))

            # create |0⟩|0⟩ + |1⟩|1⟩
            ps.call(
                self.pulse.cnot(
                    control_qubit,
                    target_qubit,
                    zx90=zx90,
                    only_low_to_high=True,
                )
            )

            # apply the control basis transformation
            if control_basis == "X":
                ps.add(control_qubit, self.pulse.y90m(control_qubit))
            elif control_basis == "Y":
                ps.add(control_qubit, self.pulse.x90(control_qubit))

            # apply the target basis transformation
            if target_basis == "X":
                ps.add(target_qubit, self.pulse.y90m(target_qubit))
            elif target_basis == "Y":
                ps.add(target_qubit, self.pulse.x90(target_qubit))

        result = self.measure(
            ps,
            mode="single",
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=reset_awg_and_capunits,
        )

        basis_labels = result.get_basis_labels(pair)
        prob_dict_raw = result.get_probabilities(pair)
        # Ensure all basis labels are present in the raw probabilities
        prob_dict_raw = {label: prob_dict_raw.get(label, 0) for label in basis_labels}
        prob_dict_mitigated = result.get_mitigated_probabilities(pair)

        labels = [f"|{i}⟩" for i in prob_dict_raw]
        prob_arr_raw = np.array(list(prob_dict_raw.values()))
        prob_arr_mitigated = np.array(list(prob_dict_mitigated.values()))

        fig = viz.make_figure()
        if plot_raw:
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=prob_arr_raw,
                    name="Raw",
                )
            )
        if plot_mitigated:
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=prob_arr_mitigated,
                    name="Mitigated",
                )
            )
        fig.update_layout(
            title=f"Bell state measurement: {control_qubit}-{target_qubit}",
            xaxis_title=f"State ({control_basis}{target_basis} basis)",
            yaxis_title="Probability",
            barmode="group",
            yaxis_range=[0, 1],
        )
        if plot:
            if plot_sequence:
                ps.plot(
                    title=f"Bell state measurement: {control_basis}{target_basis} basis"
                )
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )

            for label, p, mp in zip(
                labels,
                prob_arr_raw,
                prob_arr_mitigated,
                strict=True,
            ):
                print(f"{label} : {p:.2%} -> {mp:.2%}")

        if save_image:
            viz.save_figure(
                fig,
                f"bell_state_measurement_{control_qubit}-{target_qubit}",
            )

        return Result(
            data={
                "raw": prob_arr_raw,
                "mitigated": prob_arr_mitigated,
                "result": result,
                "figure": fig,
            }
        )

    def bell_state_tomography(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        readout_mitigation: bool | None = None,
        zx90: PulseSchedule | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
        save_image: bool | None = None,
        mle_fit: bool | None = None,
    ) -> Result:
        """
        Perform two-qubit state tomography for a Bell state.

        Parameters
        ----------
        control_qubit
            Control qubit label.
        target_qubit
            Target qubit label.
        readout_mitigation
            Whether to apply readout mitigation.
        """
        if readout_mitigation is None:
            readout_mitigation = True
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True
        if save_image is None:
            save_image = True
        if mle_fit is None:
            mle_fit = True

        """Performs full state tomography on a n-qubit GHZ state.

        This involves:
        1. Measuring the GHZ state in all 3^n Pauli bases.
        2. Calculating the expectation values for all 4^n Pauli strings.
        3. Reconstructing the 2^n x 2^n density matrix using linear inversion or MLE.
        4. Calculating the fidelity with the ideal GHZ state.
        5. Plotting the resulting density matrix.

        Parameters
        ----------
        entangle_steps : list[tuple[str, str]]
            List of tuples representing the entanglement steps, e.g., [("Q00", "Q01"), ("Q01", "Q02")].
        readout_mitigation : bool
            Whether to apply readout error mitigation.
        shots : int
            Number of shots for each measurement.
        interval : float
            Time interval between measurements.
        plot : bool
            Whether to plot the resulting density matrix.
        save_image : bool
            Whether to save the plot as an image.
        mle_fit : bool
            Whether to use Maximum Likelihood Estimation (MLE) for density matrix reconstruction.

        Returns
        -------
        dict
            A dictionary containing:
            - "probabilities": Measured probabilities in all bases.
            - "expected_values": Calculated expectation values for all Pauli strings.
            - "density_matrix": Reconstructed density matrix.
            - "fidelity": Fidelity with the ideal GHZ state.
            - "figure": Plotly figure of the density matrix.
        """
        n_qubits = 2
        dim = 2**n_qubits
        probabilities = {}
        for control_basis, target_basis in tqdm(
            product(["X", "Y", "Z"], repeat=n_qubits),
            desc="Measuring Bell state",
        ):
            result = self.measure_bell_state(
                control_qubit,
                target_qubit,
                control_basis=control_basis,
                target_basis=target_basis,
                zx90=zx90,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )
            basis = f"{control_basis}{target_basis}"
            if readout_mitigation:
                probabilities[basis] = result["mitigated"]
            else:
                probabilities[basis] = result["raw"]

        expected_values = {}
        paulis = {
            "I": np.array([[1, 0], [0, 1]]),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
        }
        rho = np.zeros((dim, dim), dtype=np.complex128)
        for control_basis, control_pauli in paulis.items():
            for target_basis, target_pauli in paulis.items():
                basis = f"{control_basis}{target_basis}"
                # calculate the expectation values
                if basis == "II":
                    # II is always 1
                    # 00: +1, 01: +1, 10: +1, 11: +1
                    p = probabilities["ZZ"]
                    e = p[0b00] + p[0b01] + p[0b10] + p[0b11]
                elif basis in ["IX", "IY", "IZ"]:
                    # ignore the first qubit
                    # 00: +1, 01: -1, 10: +1, 11: -1
                    p = probabilities[f"Z{target_basis}"]
                    e = p[0b00] - p[0b01] + p[0b10] - p[0b11]
                elif basis in ["XI", "YI", "ZI"]:
                    # ignore the second qubit
                    # 00: +1, 01: +1, 10: -1, 11: -1
                    p = probabilities[f"{control_basis}Z"]
                    e = p[0b00] + p[0b01] - p[0b10] - p[0b11]
                else:
                    # two-qubit basis
                    # 00: +1, 01: -1, 10: -1, 11: +1
                    p = probabilities[basis]
                    e = p[0b00] - p[0b01] - p[0b10] + p[0b11]
                pauli = np.kron(control_pauli, target_pauli)
                rho += e * pauli
                expected_values[basis] = e

        if mle_fit:
            rho = mle_fit_density_matrix(expected_values)
        else:
            rho = rho / dim

        bell_state = np.zeros((dim, 1), dtype=np.complex128)
        bell_state[0, 0] = 1 / np.sqrt(2)
        bell_state[-1, 0] = 1 / np.sqrt(2)
        fidelity = float(np.real(bell_state.T.conj() @ rho @ bell_state))

        fig = plot_ghz_state_tomography(
            rho=rho,
            qubits=[control_qubit, target_qubit],
            fidelity=fidelity,
            plot=plot,
            save_image=save_image,
            width=600,
            height=366,
            file_name=f"bell_state_tomography_{control_qubit}-{target_qubit}",
        )["figure"]

        return Result(
            data={
                "probabilities": probabilities,
                "expected_values": expected_values,
                "density_matrix": rho,
                "fidelity": fidelity,
                "figure": fig,
            }
        )

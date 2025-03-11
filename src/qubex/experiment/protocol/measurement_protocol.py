from __future__ import annotations

from pathlib import Path
from typing import Collection, Literal, Optional, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...measurement import MeasureResult
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import PulseSchedule, Waveform
from ...typing import (
    IQArray,
    ParametricPulseSchedule,
    ParametricWaveformDict,
    TargetMap,
)
from ..experiment_constants import CALIBRATION_SHOTS, RABI_TIME_RANGE
from ..experiment_result import ExperimentResult, RabiData, SweepData


class MeasurementProtocol(Protocol):
    def execute(
        self,
        schedule: PulseSchedule,
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> MeasureResult:
        """
        Execute the given schedule.

        Parameters
        ----------
        schedule : PulseSchedule
            Schedule to execute.
        mode : Literal["single", "avg"], optional
            Measurement mode. Defaults to "avg".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> with pulse.PulseSchedule(["Q00", "Q01"]) as ps:
        ...     ps.add("Q00", pulse.Rect(...))
        ...     ps.add("Q01", pulse.Gaussian(...))
        >>> result = ex.execute(
        ...     schedule=ps,
        ...     mode="avg",
        ...     shots=1024,
        ...     interval=150 * 1024,
        ... )
        """
        ...

    def measure(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
        initial_states: dict[str, str] | None = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        plot: bool = False,
    ) -> MeasureResult:
        """
        Measures the signals using the given sequence.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Sequence of the experiment.
        frequencies : Optional[dict[str, float]]
            Frequencies of the qubits.
        initial_states : dict[str, str], optional
            Initial states of the qubits. Defaults to None.
        mode : Literal["single", "avg"], optional
            Measurement mode. Defaults to "avg".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        control_window : int, optional
            Control window. Defaults to None.
        capture_window : int, optional
            Capture window. Defaults to None.
        capture_margin : int, optional
            Capture margin. Defaults to None.
        readout_duration : int, optional
            Readout duration. Defaults to None.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each target. Defaults to None.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.measure(
        ...     sequence={"Q00": [0.1+0.0j, 0.3+0.0j, 0.1+0.0j]},
        ...     mode="avg",
        ...     shots=1024,
        ...     interval=150 * 1024,
        ...     plot=True,
        ... )
        """
        ...

    def measure_state(
        self,
        states: dict[
            str, Literal["0", "1", "+", "-", "+i", "-i"] | Literal["g", "e", "f"]
        ],
        *,
        mode: Literal["single", "avg"] = "single",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        plot: bool = False,
    ) -> MeasureResult:
        """
        Measures the signals using the given states.

        Parameters
        ----------
        states : dict[str, Literal["0", "1", "+", "-", "+i", "-i"] | Literal["g", "e", "f"]]
            States to prepare.
        mode : Literal["single", "avg"], optional
            Measurement mode. Defaults to "single".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        control_window : int, optional
            Control window. Defaults to None.
        capture_window : int, optional
            Capture window. Defaults to None.
        capture_margin : int, optional
            Capture margin. Defaults to None.
        readout_duration : int, optional
            Readout duration. Defaults to None.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.measure_state(
        ...     states={"Q00": "0", "Q01": "1"},
        ...     mode="single",
        ...     shots=1024,
        ...     interval=150 * 1024,
        ...     plot=True,
        ... )
        """
        ...

    def sweep_parameter(
        self,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        *,
        sweep_range: ArrayLike,
        repetitions: int = 1,
        frequencies: dict[str, float] | None = None,
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        plot: bool = True,
        title: str = "Sweep result",
        xlabel: str = "Sweep value",
        ylabel: str = "Measured value",
        xaxis_type: Literal["linear", "log"] = "linear",
        yaxis_type: Literal["linear", "log"] = "linear",
    ) -> ExperimentResult[SweepData]:
        """
        Sweeps a parameter and measures the signals.

        Parameters
        ----------
        sequence : ParametricPulseSchedule | ParametricWaveformMap
            Parametric sequence to sweep.
        sweep_range : ArrayLike
            Range of the parameter to sweep.
        repetitions : int, optional
            Number of repetitions. Defaults to 1.
        frequencies : dict[str, float], optional
            Frequencies of the qubits. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        control_window : int, optional
            Control window. Defaults to None.
        capture_window : int, optional
            Capture window. Defaults to None.
        capture_margin : int, optional
            Capture margin. Defaults to None.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        title : str, optional
            Title of the plot. Defaults to "Sweep result".
        xlabel : str, optional
            Title of the x-axis. Defaults to "Sweep value".
        ylabel : str, optional
            Title of the y-axis. Defaults to "Measured value".
        xaxis_type : Literal["linear", "log"], optional
            Type of the x-axis. Defaults to "linear".
        yaxis_type : Literal["linear", "log"], optional
            Type of the y-axis. Defaults to "linear".

        Returns
        -------
        ExperimentResult[SweepData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.sweep_parameter(
        ...     sequence=lambda x: {"Q00": Rect(duration=30, amplitude=x)},
        ...     sweep_range=np.arange(0, 101, 4),
        ...     repetitions=4,
        ...     shots=1024,
        ...     plot=True,
        ... )
        """
        ...

    def repeat_sequence(
        self,
        sequence: TargetMap[Waveform] | PulseSchedule,
        *,
        repetitions: int = 20,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[SweepData]:
        """
        Repeats the pulse sequence n times.

        Parameters
        ----------
        sequence : TargetMap[Waveform] | PulseSchedule
            Pulse sequence to repeat.
        repetitions : int, optional
            Number of repetitions. Defaults to 20.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[SweepData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.repeat_sequence(
        ...     sequence={"Q00": Rect(duration=64, amplitude=0.1)},
        ...     repetitions=4,
        ... )
        """
        ...

    def obtain_rabi_params(
        self,
        targets: Collection[str] | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        amplitudes: dict[str, float] | None = None,
        frequencies: dict[str, float] | None = None,
        is_damped: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = True,
        simultaneous: bool = False,
    ) -> ExperimentResult[RabiData]: ...

    def obtain_ef_rabi_params(
        self,
        targets: Collection[str] | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        is_damped: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]: ...

    def rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike = RABI_TIME_RANGE,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]: ...

    def ef_rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]: ...

    def measure_state_distribution(
        self,
        targets: Collection[str] | None = None,
        *,
        n_states: Literal[2, 3] = 2,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> list[MeasureResult]: ...

    def build_classifier(
        self,
        targets: str | Collection[str] | None = None,
        *,
        n_states: Literal[2, 3] = 2,
        save_classifier: bool = True,
        save_dir: Path | str | None = None,
        shots: int = 8192,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict: ...

    def state_tomography(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict[str, tuple[float, float, float]]:
        """
        Conducts a state tomography experiment.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Sequence to measure for each target.
        x90 : TargetMap[Waveform], optional
            π/2 pulse. Defaults to None.
        initial_state : TargetMap[str], optional
            Initial state of each target. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Results of the experiment.
        """
        ...

    def state_evolution_tomography(
        self,
        *,
        sequences: (
            Sequence[TargetMap[IQArray]]
            | Sequence[TargetMap[Waveform]]
            | Sequence[PulseSchedule]
        ),
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        """
        Conducts a state evolution tomography experiment.

        Parameters
        ----------
        sequences : Sequence[TargetMap[IQArray]] | Sequence[TargetMap[Waveform]] | Sequence[PulseSchedule]
            Sequences to measure for each target.
        x90 : TargetMap[Waveform], optional
            π/2 pulse. Defaults to None.
        initial_state : TargetMap[str], optional
            Initial state of each target. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Results of the experiment.
        """
        ...

    def pulse_tomography(
        self,
        waveforms: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        n_samples: int = 100,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> TargetMap[NDArray[np.float64]]:
        """
        Conducts a pulse tomography experiment.

        Parameters
        ----------
        waveforms : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Waveforms to measure for each target.
        x90 : TargetMap[Waveform], optional
            π/2 pulse. Defaults to None.
        initial_state : TargetMap[str], optional
            Initial state of each target. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        """
        ...

    def measure_population(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        fit_gmm: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """
        Measures the state populations of the target qubits.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Sequence to measure for each target.
        fit_gmm : bool, optional
            Whether to fit the data with a Gaussian mixture model. Defaults to False
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]
            State probabilities and standard deviations.

        Examples
        --------
        >>> sequence = {
        ...     "Q00": ex.hpi_pulse["Q00"],
        ...     "Q01": ex.hpi_pulse["Q01"],
        ... }
        >>> result = ex.measure_population(sequence)
        """
        ...

    def measure_population_dynamics(
        self,
        *,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        params_list: Sequence | NDArray,
        fit_gmm: bool = False,
        xlabel: str = "Index",
        scatter_mode: str = "lines+markers",
        show_error: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """
        Measures the population dynamics of the target qubits.

        Parameters
        ----------
        sequence : ParametricPulseSchedule | ParametricWaveformDict
            Parametric sequence to measure.
        params_list : Sequence | NDArray
            List of parameters.
        fit_gmm : bool, optional
            Whether to fit the data with a Gaussian mixture model. Defaults to False.
        xlabel : str, optional
            Label of the x-axis. Defaults to "Index".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]
            State probabilities and standard deviations.

        Examples
        --------
        >>> sequence = lambda x: {
        ...     "Q00": ex.hpi_pulse["Q00"].scaled(x),
        ...     "Q01": ex.hpi_pulse["Q01"].scaled(x),
        >>> params_list = np.linspace(0.5, 1.5, 100)
        >>> result = ex.measure_popultion_dynamics(sequence, params_list)
        """
        ...

    def measure_bell_state(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: PulseSchedule | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict: ...

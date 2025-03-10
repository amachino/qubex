from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike

from ...clifford import Clifford
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import PulseArray, PulseSchedule, Waveform
from ...typing import TargetMap
from ..experiment_result import ExperimentResult, RBData


class BenchmarkingProtocol(Protocol):
    def rb_sequence(
        self,
        *,
        target: str,
        n: int,
        x90: dict[str, Waveform] | None = None,
        zx90: PulseSchedule | dict[str, Waveform] | None = None,
        interleaved_waveform: (
            PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_clifford: Clifford | None = None,
        seed: int | None = None,
    ) -> PulseSchedule: ...

    def rb_sequence_1q(
        self,
        *,
        target: str,
        n: int,
        x90: Waveform | dict[str, Waveform] | None = None,
        interleaved_waveform: (
            Waveform | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        seed: int | None = None,
    ) -> PulseArray:
        """
        Generates a randomized benchmarking sequence.

        Parameters
        ----------
        target : str
            Target qubit.
        n : int
            Number of Clifford gates.
        x90 : Waveform | dict[str, Waveform], optional
            π/2 pulse used for the experiment. Defaults to None.
        interleaved_waveform : Waveform | dict[str, PulseArray] | dict[str, Waveform], optional
            Waveform of the interleaved gate. Defaults to None.
        interleaved_clifford : Clifford | dict[str, tuple[complex, str]], optional
            Clifford map of the interleaved gate. Defaults to None.
        seed : int, optional
            Random seed.

        Returns
        -------
        PulseArray
            Randomized benchmarking sequence.

        Examples
        --------
        >>> sequence = ex.rb_sequence(
        ...     target="Q00",
        ...     n=100,
        ...     x90=Rect(duration=30, amplitude=0.1),
        ... )

        >>> sequence = ex.rb_sequence(
        ...     target="Q00",
        ...     n=100,
        ...     x90=Rect(duration=30, amplitude=0.1),
        ...     interleaved_waveform=Rect(duration=30, amplitude=0.1),
        ...     interleaved_clifford={
        ...         "I": (1, "I"),
        ...         "X": (1, "X"),
        ...         "Y": (-1, "Y"),
        ...         "Z": (-1, "Z"),
        ...     },
        ... )
        """
        ...

    def rb_sequence_2q(
        self,
        *,
        target: str,
        n: int,
        x90: TargetMap[Waveform] | None = None,
        zx90: (
            PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_waveform: (
            PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        seed: int | None = None,
    ) -> PulseSchedule:
        """
        Generates a 2Q randomized benchmarking sequence.

        Parameters
        ----------
        target : str
            Target qubit.
        n : int
            Number of Clifford gates.
        x90 : Waveform | TargetMap[Waveform], optional
            π/2 pulse used for 1Q gates. Defaults to None.
        zx90 : PulseSchedule | dict[str, Waveform], optional
            ZX90 pulses used for 2Q gates. Defaults to None.
        interleaved_waveform : PulseSchedule | dict[str, PulseArray] | dict[str, Waveform], optional
            Waveform of the interleaved gate. Defaults to None.
        interleaved_clifford : Clifford | dict[str, tuple[complex, str]], optional
            Clifford map of the interleaved gate. Defaults to None.
        seed : int, optional
            Random seed.

        Returns
        -------
        PulseSchedule
            Randomized benchmarking sequence.

        Examples
        --------
        >>> sequence = ex.rb_sequence_2q(
        ...     target="Q00-Q01",
        ...     n=100,
        ...     x90={
        ...         "Q00": Rect(duration=30, amplitude=0.1),
        ...         "Q01": Rect(duration=30, amplitude=0.1),
        ...     },
        ... )

        >>> sequence = ex.rb_sequence_2q(
        ...     target="Q00-Q01",
        ...     n=100,
        ...     x90={
        ...         "Q00": Rect(duration=30, amplitude=0.1),
        ...         "Q01": Rect(duration=30, amplitude=0.1),
        ...     },
        ...     interleaved_waveform=Rect(duration=30, amplitude=0.1),
        ...     interleaved_clifford=Clifford.CNOT(),
        ... )
        """
        ...

    def rb_experiment_1q(
        self,
        *,
        target: str,
        n_cliffords_range: ArrayLike | None = None,
        x90: Waveform | dict[str, Waveform] | None = None,
        interleaved_waveform: Waveform | None = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        seed: int | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> ExperimentResult[RBData]:
        """
        Conducts a randomized benchmarking experiment.

        Parameters
        ----------
        target : str
            Target qubit.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords. Defaults to range(0, 1001, 50).
        x90 : Waveform, optional
            π/2 pulse used for the experiment. Defaults to None.
        interleaved_waveform : Waveform, optional
            Waveform of the interleaved gate. Defaults to None.
        interleaved_clifford : Clifford | dict[str, tuple[complex, str]], optional
            Clifford map of the interleaved gate. Defaults to None.
        spectator_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Spectator state. Defaults to "0".
        seed : int, optional
            Random seed.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the images. Defaults to True.

        Returns
        -------
        ExperimentResult[RBData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.rb_experiment(
        ...     target="Q00",
        ...     n_cliffords_range=range(0, 1001, 50),
        ...     x90=Rect(duration=30, amplitude=0.1),
        ... )

        >>> result = ex.rb_experiment(
        ...     target="Q00",
        ...     n_cliffords_range=range(0, 1001, 50),
        ...     x90=Rect(duration=30, amplitude=0.1),
        ...     interleaved_waveform=Rect(duration=30, amplitude=0.1),
        ...     interleaved_clifford={
        ...         "I": (1, "I"),
        ...         "X": (1, "X"),
        ...         "Y": (-1, "Y"),
        ...         "Z": (-1, "Z"),
        ...     },
        ... )
        """
        ...

    def rb_experiment_2q(
        self,
        *,
        target: str,
        n_cliffords_range: ArrayLike = np.arange(0, 21, 2),
        x90: TargetMap[Waveform] | None = None,
        zx90: (
            PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_waveform: (
            PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        seed: int | None = None,
        mitigate_readout: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ): ...

    def randomized_benchmarking(
        self,
        target: str,
        *,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int = 30,
        x90: Waveform | dict[str, Waveform] | None = None,
        zx90: (
            PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        seeds: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        """
        Conducts a randomized benchmarking experiment with multiple trials.

        Parameters
        ----------
        target : str
            Target qubit.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords. Defaults to None.
        n_trials : int, optional
            Number of trials for different random seeds. Defaults to 30.
        x90 : Waveform | dict[str, Waveform], optional
            π/2 pulse used for the experiment. Defaults to None.
        zx90 : PulseSchedule | dict[str, PulseArray] | dict[str, Waveform], optional
            ZX90 pulses used for 2Q gates. Defaults to None.
        spectator_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Spectator state. Defaults to "0".
        seeds : ArrayLike, optional
            Random seeds. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the images. Defaults to True.

        Returns
        -------
        dict
            Results of the experiment.
        """
        ...

    def interleaved_randomized_benchmarking(
        self,
        *,
        target: str,
        interleaved_waveform: Waveform | PulseSchedule,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]],
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int = 30,
        x90: TargetMap[Waveform] | Waveform | None = None,
        zx90: (
            PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        seeds: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        """
        Conducts a randomized benchmarking experiment with multiple trials.

        Parameters
        ----------
        target : str
            Target qubit.
        interleaved_waveform : Waveform
            Waveform of the interleaved gate.
        interleaved_clifford : str | Clifford | dict[str, tuple[complex, str]]
            Clifford map of the interleaved gate.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords. Defaults to range(0, 1001, 100).
        n_trials : int, optional
            Number of trials for different random seeds. Defaults to 30.
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        zx90 : PulseSchedule | dict[str, PulseArray] | dict[str, Waveform], optional
            ZX90 pulses used for 2Q gates. Defaults to None.
        spectator_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Spectator state. Defaults to "0".
        seeds : ArrayLike, optional
            Random seeds. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the images. Defaults to True.

        Returns
        -------
        dict
            Results of the experiment.

        Examples
        --------
        >>> result = ex.interleaved_randomized_benchmarking(
        ...     target="Q00",
        ...     interleaved_waveform=Rect(duration=30, amplitude=0.1),
        ...     interleaved_clifford={
        ...         "I": (1, "I"),
        ...         "X": (1, "X"),
        ...         "Y": (1, "Z"),
        ...         "Z": (-1, "Y"),
        ...     },
        ...     n_cliffords_range=range(0, 1001, 100),
        ...     n_trials=30,
        ...     x90=Rect(duration=30, amplitude=0.1),
        ...     spectator_state="0",
        ...     show_ref=True,
        ...     shots=1024,
        ...     interval=1024,
        ...     plot=True,
        ... )
        """
        ...

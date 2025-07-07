from __future__ import annotations

from typing import Collection, Protocol

from numpy.typing import ArrayLike

from ...clifford import Clifford
from ...pulse import PulseArray, PulseSchedule, Waveform
from ...typing import TargetMap
from ..experiment_result import ExperimentResult, RBData


class BenchmarkingProtocol(Protocol):
    def rb_sequence(
        self,
        target: str,
        *,
        n: int,
        x90: Waveform | TargetMap[Waveform] | None = None,
        zx90: PulseSchedule | None = None,
        interleaved_waveform: Waveform | PulseSchedule | None = None,
        interleaved_clifford: Clifford | None = None,
        seed: int | None = None,
    ) -> PulseSchedule: ...

    def rb_sequence_1q(
        self,
        target: str,
        *,
        n: int,
        x90: Waveform | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: Waveform | None = None,
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
        x90 : Waveform , optional
            π/2 pulse used for the experiment.
        interleaved_waveform : Waveform | None, optional
            Waveform of the interleaved gate.
        interleaved_clifford : Clifford | None, optional
            Clifford map of the interleaved gate.
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
        ...     interleaved_clifford="X90",
        ... )
        """
        ...

    def rb_sequence_2q(
        self,
        target: str,
        *,
        n: int,
        x90: TargetMap[Waveform] | None = None,
        zx90: PulseSchedule | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: PulseSchedule | None = None,
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
        x90 : TargetMap[Waveform], optional
            π/2 pulse used for 1Q gates.
        zx90 : PulseSchedule | None, optional
            ZX90 pulses used for 2Q gates.
        interleaved_waveform : PulseSchedule | None, optional
            Waveform of the interleaved gate.
        interleaved_clifford : Clifford | None, optional
            Clifford map of the interleaved gate.
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
        ...     interleaved_clifford=Clifford.ZX90(),
        ... )
        """
        ...

    def rb_experiment_1q(
        self,
        targets: Collection[str] | str,
        *,
        in_parallel: bool = False,
        n_cliffords_range: ArrayLike | None = None,
        x90: TargetMap[Waveform] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[Waveform] | None = None,
        seed: int | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> ExperimentResult[RBData]:
        """
        Conducts a randomized benchmarking experiment.

        Parameters
        ----------
        targets : Collection[str] | str
            Target qubits.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords.
        x90 : TargetMap[Waveform], optional
            π/2 pulse used for the experiment.
        interleaved_clifford : Clifford | None, optional
            Clifford map of the interleaved gate.
        interleaved_waveform : TargetMap[Waveform], optional
            Waveform of the interleaved gate.
        seed : int | None, optional
            Random seed.
        shots : int | None, optional
            Number of shots.
        interval : float | None, optional
            Interval between shots.
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
        ...     interleaved_clifford=Clifford.X90(),
        ... )
        """
        ...

    def rb_experiment_2q(
        self,
        targets: Collection[str] | str,
        *,
        in_parallel: bool = False,
        n_cliffords_range: ArrayLike | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[PulseSchedule] | None = None,
        seed: int | None = None,
        mitigate_readout: bool = True,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
    ): ...

    def randomized_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        in_parallel: bool = False,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        seeds: ArrayLike | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        """
        Conducts a randomized benchmarking experiment with multiple trials.

        Parameters
        ----------
        targets : Collection[str] | str
            Target qubits.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords.
        n_trials : int, optional
            Number of trials for different random seeds.
        x90 : Waveform | None, optional
            π/2 pulse used for the experiment.
        zx90 : TargetMap[PulseSchedule] | None, optional
            ZX90 pulses used for 2Q gates.
        seeds : ArrayLike, optional
            Random seeds.
        shots : int, optional
            Number of shots.
        interval : float, optional
            Interval between shots.
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
        targets: Collection[str] | str,
        *,
        in_parallel: bool = False,
        interleaved_clifford: str | Clifford,
        interleaved_waveform: TargetMap[PulseSchedule]
        | TargetMap[Waveform]
        | None = None,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        seeds: ArrayLike | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        """
        Conducts a randomized benchmarking experiment with multiple trials.

        Parameters
        ----------
        target : str
            Target qubit.
        interleaved_clifford : str | Clifford
            Clifford map of the interleaved gate.
        interleaved_waveform : TargetMap[PulseSchedule] | TargetMap[Waveform] | None, optional
            Waveform of the interleaved gate.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords.
        n_trials : int, optional
            Number of trials for different random seeds.
        x90 : TargetMap[Waveform] | None, optional
            π/2 pulse used for the experiment.
        zx90 : TargetMap[PulseSchedule] | None, optional
            ZX90 pulses used for 2Q gates.
        seeds : ArrayLike, optional
            Random seeds.
        shots : int, optional
            Number of shots.
        interval : float, optional
            Interval between shots.
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
        ...     interleaved_clifford="X90",
        ...     interleaved_waveform=Rect(duration=30, amplitude=0.1),
        ...     n_cliffords_range=range(0, 1001, 100),
        ...     n_trials=30,
        ...     x90=Rect(duration=30, amplitude=0.1),
        ...     shots=1024,
        ...     plot=True,
        ... )
        """
        ...

from __future__ import annotations

from typing import Collection, Literal, Protocol

from numpy.typing import ArrayLike

from ...clifford import Clifford
from ...pulse import PulseArray, PulseSchedule, Waveform
from ...typing import TargetMap


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
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[Waveform] | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        """
        Conducts a randomized benchmarking experiment.

        Parameters
        ----------
        targets : Collection[str] | str
            Target qubits.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords.
        n_trials : int, optional
            Number of trials for different random seeds.
        seeds : ArrayLike, optional
            Random seeds.
        max_n_cliffords : int, optional
            Maximum number of Cliffords to use in the experiment.
        x90 : TargetMap[Waveform] | None, optional
            π/2 pulse used for the experiment.
        interleaved_clifford : Clifford | None, optional
            Clifford map of the interleaved gate.
        interleaved_waveform : TargetMap[Waveform] | None, optional
            Waveform of the interleaved gate.
        in_parallel : bool, optional
            Whether to run the experiment in parallel for multiple targets.
            Defaults to False.
        shots : int, optional
            Number of shots for the experiment.
        interval : float, optional
            Interval between shots.
        xaxis_type : Literal["linear", "log"] | None, optional
            Type of x-axis for the plot. If None, defaults to "linear".
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the images. Defaults to True.

        Returns
        -------
        dict
            Results of the experiment, including the measured signals and fit parameters.

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
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        mitigate_readout: bool = True,
        shots: int | None = None,
        interval: float | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        plot: bool = True,
        save_image: bool = True,
    ): ...

    def irb_experiment(
        self,
        targets: Collection[str] | str,
        *,
        interleaved_clifford: str | Clifford,
        interleaved_waveform: TargetMap[PulseSchedule]
        | TargetMap[Waveform]
        | None = None,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict: ...

    def randomized_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        xaxis_type: Literal["linear", "log"] | None = None,
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
        seeds : ArrayLike, optional
            Random seeds.
        max_n_cliffords : int, optional
            Maximum number of Cliffords to use in the experiment.
        x90 : TargetMap[Waveform] | None, optional
            π/2 pulse used for the experiment.
        zx90 : TargetMap[PulseSchedule] | None, optional
            ZX90 pulses used for 2Q gates.
        in_parallel : bool, optional
            Whether to run the experiment in parallel for multiple targets.
            Defaults to False.
        xaxis_type : Literal["linear", "log"] | None, optional
            Type of x-axis for the plot. If None, defaults to "linear".
        shots : int, optional
            Number of shots for the experiment.
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
        interleaved_clifford: str | Clifford,
        interleaved_waveform: TargetMap[PulseSchedule]
        | TargetMap[Waveform]
        | None = None,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
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
        interleaved_clifford : str | Clifford
            Clifford map of the interleaved gate.
        interleaved_waveform : TargetMap[PulseSchedule] | TargetMap[Waveform] | None, optional
            Waveform of the interleaved gate.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords.
        n_trials : int, optional
            Number of trials for different random seeds.
        seeds : ArrayLike, optional
            Random seeds.
        max_n_cliffords : int, optional
            Maximum number of Cliffords to use in the experiment.
        x90 : TargetMap[Waveform] | None, optional
            π/2 pulse used for the experiment.
        zx90 : TargetMap[PulseSchedule] | None, optional
            ZX90 pulses used for 2Q gates.
        in_parallel : bool, optional
            Whether to run the experiment in parallel for multiple targets.
            Defaults to False.
        shots : int, optional
            Number of shots for the experiment.
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

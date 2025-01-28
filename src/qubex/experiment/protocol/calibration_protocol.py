from __future__ import annotations

from typing import Collection, Literal

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Protocol

from ...clifford import Clifford
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS, MeasureResult
from ...pulse import PulseSchedule, PulseSequence, Waveform
from ...typing import TargetMap
from ..experiment_constants import CALIBRATION_SHOTS, DRAG_COEFF
from ..experiment_result import AmplCalibData, ExperimentResult, RBData


class CalibrationProtocol(Protocol):
    def calibrate_default_pulse(
        self,
        targets: Collection[str],
        pulse_type: Literal["pi", "hpi"],
        n_rotations: int = 1,
        plot: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the default pulse.

        Parameters
        ----------
        targets : Collection[str]
            Target qubits to calibrate.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_ef_pulse(
        self,
        targets: Collection[str],
        pulse_type: Literal["pi", "hpi"],
        n_rotations: int = 1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the default pulse.

        Parameters
        ----------
        targets : Collection[str]
            Target qubits to calibrate.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_drag_amplitude(
        self,
        targets: Collection[str],
        *,
        spectator_state: str = "+",
        pulse_type: Literal["pi", "hpi"],
        n_rotations: int = 4,
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        use_stored_amplitude: bool = False,
        use_stored_beta: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict[str, float]:
        """
        Calibrates the DRAG amplitude.

        Parameters
        ----------
        targets : Collection[str]
            Target qubits to calibrate.
        spectator_state : str, optional
            Spectator state. Defaults to "+".
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        n_rotations : int, optional
            Number of rotations to |0> state. Defaults to 4.
        duration : float, optional
            Duration of the pulse. Defaults to None.
        drag_coeff : float, optional
            DRAG coefficient. Defaults to DRAG_COEFF.
        use_stored_amplitude : bool, optional
            Whether to use the stored amplitude. Defaults to False.
        use_stored_beta : bool, optional
            Whether to use the stored beta. Defaults to False.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        dict[str, float]
            Result of the calibration.
        """
        ...

    def calibrate_drag_beta(
        self,
        targets: Collection[str] | None = None,
        *,
        spectator_state: str = "+",
        pulse_type: Literal["pi", "hpi"] = "hpi",
        beta_range: ArrayLike = np.linspace(-0.5, 1.5, 41),
        n_turns: int = 1,
        duration: float | None = None,
        degree: int = 3,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict[str, float]:
        """
        Calibrates the DRAG beta.

        Parameters
        ----------
        targets : Collection[str]
            Target qubits to calibrate.
        spectator_state : str, optional
            Spectator state. Defaults to "+".
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        beta_range : ArrayLike, optional
            Range of the beta to sweep. Defaults to np.linspace(-0.5, 1.5, 41).
        n_turns : int, optional
            Number of turns to |0> state. Defaults to 1.
        duration : float, optional
            Duration of the pulse. Defaults to None.
        degree : int, optional
            Degree of the polynomial to fit. Defaults to 3.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        dict[str, float]
            Result of the calibration.
        """
        ...

    def calibrate_hpi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 1,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π/2 pulse.

        Parameters
        ----------
        targets : Collection[str], optional
            Target qubits to calibrate.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_pi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 1,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π pulse.

        Parameters
        ----------
        targes : Collection[str], optional
            Target qubits to calibrate.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_ef_hpi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 1,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the ef π/2 pulse.

        Parameters
        ----------
        targets : Collection[str], optional
            Target qubits to calibrate.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_ef_pi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 1,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the ef π pulse.

        Parameters
        ----------
        targets : Collection[str], optional
            Target qubits to calibrate.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_drag_hpi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-0.5, 1.5, 41),
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict:
        """
        Calibrates the DRAG π/2 pulse.

        Parameters
        ----------
        targets : Collection[str], optional
            Target qubits to calibrate.
        n_rotations : int, optional
            Number of rotations to |0> state. Defaults to 4.
        n_turns : int, optional
            Number of turns to |0> state. Defaults to 1.
        n_iterations : int, optional
            Number of iterations. Defaults to 2.
        calibrate_beta : bool, optional
            Whether to calibrate the DRAG beta. Defaults to True.
        beta_range : ArrayLike, optional
            Range of the beta to sweep. Defaults to np.linspace(-0.5, 1.5, 41).
        duration : float, optional
            Duration of the pulse. Defaults to None.
        drag_coeff : float, optional
            DRAG coefficient. Defaults to DRAG_COEFF.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        dict
            Result of the calibration.
        """
        ...

    def calibrate_drag_pi_pulse(
        self,
        targets: Collection[str] | None = None,
        *,
        spectator_state: str = "+",
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-0.5, 1.5, 41),
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        degree: int = 3,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict:
        """
        Calibrates the DRAG π pulse.

        Parameters
        ----------
        targets : Collection[str], optional
            Target qubits to calibrate.
        n_rotations : int, optional
            Number of rotations to |0> state. Defaults to 4.
        n_turns : int, optional
            Number of turns to |0> state. Defaults to 1.
        n_iterations : int, optional
            Number of iterations. Defaults to 2.
        calibrate_beta : bool, optional
            Whether to calibrate the DRAG beta. Defaults to False.
        beta_range : ArrayLike, optional
            Range of the beta to sweep. Defaults to np.linspace(-0.5, 1.5, 41).
        duration : float, optional
            Duration of the pulse. Defaults to None.
        drag_coeff : float, optional
            DRAG coefficient. Defaults to DRAG_COEFF.
        degree : int, optional
            Degree of the polynomial to fit. Defaults to 3.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        dict
            Result of the calibration.
        """
        ...

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
        targets: Collection[str] | None = None,
        *,
        n_states: Literal[2, 3] = 2,
        shots: int = 10000,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict: ...

    def rb_sequence(
        self,
        *,
        target: str,
        n: int,
        x90: dict[str, Waveform] | None = None,
        zx90: PulseSchedule | dict[str, Waveform] | None = None,
        interleaved_waveform: (
            PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform] | None
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
            Waveform | dict[str, PulseSequence] | dict[str, Waveform] | None
        ) = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        seed: int | None = None,
    ) -> PulseSequence:
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
        interleaved_waveform : Waveform | dict[str, PulseSequence] | dict[str, Waveform], optional
            Waveform of the interleaved gate. Defaults to None.
        interleaved_clifford : Clifford | dict[str, tuple[complex, str]], optional
            Clifford map of the interleaved gate. Defaults to None.
        seed : int, optional
            Random seed.

        Returns
        -------
        PulseSequence
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
            PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform] | None
        ) = None,
        interleaved_waveform: (
            PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform] | None
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
        interleaved_waveform : PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform], optional
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
            PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform] | None
        ) = None,
        interleaved_waveform: (
            PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform] | None
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
            PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform] | None
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
        zx90 : PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform], optional
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
            PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform] | None
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
        interleaved_clifford : Clifford | dict[str, tuple[complex, str]]
            Clifford map of the interleaved gate.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords. Defaults to range(0, 1001, 100).
        n_trials : int, optional
            Number of trials for different random seeds. Defaults to 30.
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        zx90 : PulseSchedule | dict[str, PulseSequence] | dict[str, Waveform], optional
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

    def measure_cr_dynamics(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        time_range: ArrayLike = np.arange(100, 401, 10),
        cr_amplitude: float = 1.0,
        cr_ramptime: float = 50,
        cr_phase: float = 0.0,
        cancel_amplitude: float = 0.0,
        cancel_phase: float = 0.0,
        echo: bool = False,
        control_state: str = "0",
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict: ...

    def cr_hamiltonian_tomography(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        flattop_range: ArrayLike = np.arange(0, 301, 10),
        cr_amplitude: float = 1.0,
        cr_ramptime: float = 50,
        cr_phase: float = 0.0,
        cancel_amplitude: float = 0.0,
        cancel_phase: float = 0.0,
        x90: TargetMap[Waveform] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict: ...

    def obtain_cr_params(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        flattop_range: ArrayLike = np.arange(0, 401, 20),
        cr_amplitude: float = 1.0,
        cr_ramptime: float = 50,
        n_iterations: int = 2,
        x90: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict: ...

    def calibrate_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        amplitude_range: ArrayLike | None = None,
        duration_range: ArrayLike | None = None,
        amplitude: float = 0.5,
        duration: float = 200,
        ramptime: float = 50,
        n_repetitions: int = 1,
        degree: int = 3,
        x180: TargetMap[Waveform] | Waveform | None = None,
        use_zvalues: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ): ...

    def calibrate_zx90_by_amplitude(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        duration: float = 100,
        ramptime: float = 20,
        amplitude_range: ArrayLike = np.linspace(0.0, 1.0, 51),
        initial_state: str = "0",
        n_repetitions: int = 1,
        degree: int = 3,
        x180: TargetMap[Waveform] | Waveform | None = None,
        use_zvalues: bool = False,
        store_params: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ): ...

    def calibrate_zx90_by_duration(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        amplitude: float = 0.5,
        duration_range: ArrayLike = np.arange(100, 201, 2),
        ramptime: float = 20,
        initial_state: str = "0",
        n_repetitions: int = 1,
        degree: int = 3,
        x180: TargetMap[Waveform] | Waveform | None = None,
        use_zvalues: bool = False,
        store_params: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ): ...

    def zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        cr_duration: float | None = None,
        cr_ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        echo: bool = True,
        x180: TargetMap[Waveform] | Waveform | None = None,
    ) -> PulseSchedule: ...

    def cnot(
        self,
        control_qubit: str,
        target_qubit: str,
        zx90: PulseSchedule | None = None,
        x90: Waveform | None = None,
    ) -> PulseSchedule: ...

    def measure_bell_state(
        self,
        control_qubit: str,
        target_qubit: str,
        zx90: PulseSchedule | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict: ...

    def optimize_x90(
        self,
        qubit: str,
        *,
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform: ...

    def optimize_drag_x90(
        self,
        qubit: str,
        *,
        duration: float = 16,
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform: ...

    def optimize_pulse(
        self,
        qubit: str,
        *,
        pulse: Waveform,
        x90: Waveform,
        target_state: tuple[float, float, float],
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform: ...

    def optimize_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        duration: float = 100,
        ramptime: float = 20,
        x180: TargetMap[Waveform] | Waveform | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ): ...

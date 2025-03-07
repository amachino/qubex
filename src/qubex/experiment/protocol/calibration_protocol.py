from __future__ import annotations

from typing import Collection, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike

from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import Waveform
from ...typing import TargetMap
from ..experiment_constants import CALIBRATION_SHOTS, DRAG_COEFF
from ..experiment_result import AmplCalibData, ExperimentResult


class CalibrationProtocol(Protocol):
    def calibrate_default_pulse(
        self,
        targets: Collection[str],
        *,
        pulse_type: Literal["pi", "hpi"],
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
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
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
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

    def calibrate_hpi_pulse(
        self,
        targets: Collection[str] | None = None,
        *,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π/2 pulse.

        Parameters
        ----------
        targets : Collection[str], optional
            Target qubits to calibrate.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
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
        *,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π pulse.

        Parameters
        ----------
        targes : Collection[str], optional
            Target qubits to calibrate.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
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

    def calibrate_ef_pulse(
        self,
        targets: Collection[str],
        *,
        pulse_type: Literal["pi", "hpi"],
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        shots: int = CALIBRATION_SHOTS,
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
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
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

    def calibrate_ef_hpi_pulse(
        self,
        targets: Collection[str] | None = None,
        *,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the ef π/2 pulse.

        Parameters
        ----------
        targets : Collection[str], optional
            Target qubits to calibrate.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
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
        *,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the ef π pulse.

        Parameters
        ----------
        targets : Collection[str], optional
            Target qubits to calibrate.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
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

    def calibrate_drag_amplitude(
        self,
        targets: Collection[str],
        *,
        spectator_state: str = "+",
        pulse_type: Literal["pi", "hpi"],
        n_points: int = 20,
        n_rotations: int = 4,
        r2_threshold: float = 0.5,
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
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations to |0> state. Defaults to 4.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
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
        beta_range: ArrayLike = np.linspace(-1.5, 1.5, 20),
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
            Range of the beta to sweep. Defaults to np.linspace(-1.5, 1.5, 20).
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

    def calibrate_drag_hpi_pulse(
        self,
        targets: Collection[str] | None = None,
        *,
        spectator_state: str = "+",
        n_points: int = 20,
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        r2_threshold: float = 0.5,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-1.5, 1.5, 20),
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
        spectator_state : str, optional
            Spectator state. Defaults to "+".
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations to |0> state. Defaults to 4.
        n_turns : int, optional
            Number of turns to |0> state. Defaults to 1.
        n_iterations : int, optional
            Number of iterations. Defaults to 2.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        calibrate_beta : bool, optional
            Whether to calibrate the DRAG beta. Defaults to True.
        beta_range : ArrayLike, optional
            Range of the beta to sweep. Defaults to np.linspace(-1.5, 1.5, 20).
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
        n_points: int = 20,
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        r2_threshold: float = 0.5,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-1.5, 1.5, 20),
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
        spectator_state : str, optional
            Spectator state. Defaults to "+".
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations to |0> state. Defaults to 4.
        n_turns : int, optional
            Number of turns to |0> state. Defaults to 1.
        n_iterations : int, optional
            Number of iterations. Defaults to 2.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        calibrate_beta : bool, optional
            Whether to calibrate the DRAG beta. Defaults to False.
        beta_range : ArrayLike, optional
            Range of the beta to sweep. Defaults to np.linspace(-1.5, 1.5, 20).
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

    def measure_cr_dynamics(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        time_range: ArrayLike | None = None,
        cr_amplitude: float = 1.0,
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
        time_range: ArrayLike | None = None,
        cr_amplitude: float = 1.0,
        cr_phase: float = 0.0,
        cancel_amplitude: float = 0.0,
        cancel_phase: float = 0.0,
        x90: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict: ...

    def update_cr_params(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        time_range: ArrayLike | None = None,
        cr_amplitude: float = 1.0,
        cr_ramptime: float = 20.0,
        cr_phase: float = 0.0,
        cancel_amplitude: float = 0.0,
        cancel_phase: float = 0.0,
        x90: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict: ...

    def obtain_cr_params(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        cr_amplitude: float = 1.0,
        cr_ramptime: float = 20.0,
        n_iterations: int = 4,
        time_range: ArrayLike = np.arange(0, 401, 20),
        use_stored_params: bool = True,
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
        duration: float = 200,
        ramptime: float = 50,
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

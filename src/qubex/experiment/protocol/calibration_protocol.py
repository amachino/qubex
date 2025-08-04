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
        targets: Collection[str] | str | None = None,
        *,
        pulse_type: Literal["pi", "hpi"],
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        update_params: bool = True,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the default pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
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
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π/2 pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
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
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π pulse.

        Parameters
        ----------
        targes : Collection[str] | str, optional
            Target qubits to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
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
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_ef_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        pulse_type: Literal["pi", "hpi"],
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the default pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_ef_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the ef π/2 pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
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
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_ef_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the ef π pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
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
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        ...

    def calibrate_drag_amplitude(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
        pulse_type: Literal["pi", "hpi"],
        duration: float | None = None,
        n_points: int = 20,
        n_rotations: int = 4,
        r2_threshold: float = 0.5,
        drag_coeff: float = DRAG_COEFF,
        use_stored_amplitude: bool = False,
        use_stored_beta: bool = False,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> dict[str, dict]:
        """
        Calibrates the DRAG amplitude.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        spectator_state : str, optional
            Spectator state.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        duration : float, optional
            Duration of the pulse.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations to |0> state. Defaults to 4.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        drag_coeff : float, optional
            DRAG coefficient. Defaults to DRAG_COEFF.
        use_stored_amplitude : bool, optional
            Whether to use the stored amplitude. Defaults to False.
        use_stored_beta : bool, optional
            Whether to use the stored beta. Defaults to False.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        dict[str, dict]
            Result of the calibration.
        """
        ...

    def calibrate_drag_beta(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
        pulse_type: Literal["pi", "hpi"] = "hpi",
        beta_range: ArrayLike = np.linspace(-2.0, 2.0, 20),
        duration: float | None = None,
        n_turns: int = 1,
        degree: int = 3,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> dict[str, float]:
        """
        Calibrates the DRAG beta.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        spectator_state : str, optional
            Spectator state.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        beta_range : ArrayLike, optional
            Range of the beta to sweep. Defaults to np.linspace(-2.0, 2.0, 20).
        duration : float, optional
            Duration of the pulse.
        n_turns : int, optional
            Number of turns to |0> state. Defaults to 1.
        degree : int, optional
            Degree of the polynomial to fit. Defaults to 3.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        dict[str, float]
            Result of the calibration.
        """
        ...

    def calibrate_drag_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
        n_points: int = 20,
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        degree: int = 3,
        r2_threshold: float = 0.5,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-2.0, 2.0, 20),
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> dict:
        """
        Calibrates the DRAG π/2 pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        spectator_state : str, optional
            Spectator state.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations to |0> state. Defaults to 4.
        n_turns : int, optional
            Number of turns to |0> state. Defaults to 1.
        n_iterations : int, optional
            Number of iterations. Defaults to 2.
        degree : int, optional
            Degree of the polynomial to fit. Defaults to 3.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        calibrate_beta : bool, optional
            Whether to calibrate the DRAG beta. Defaults to True.
        beta_range : ArrayLike, optional
            Range of the beta to sweep. Defaults to np.linspace(-2.0, 2.0, 20).
        duration : float, optional
            Duration of the pulse.
        drag_coeff : float, optional
            DRAG coefficient. Defaults to DRAG_COEFF.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        dict
            Result of the calibration.
        """
        ...

    def calibrate_drag_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
        n_points: int = 20,
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        degree: int = 3,
        r2_threshold: float = 0.5,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-2.0, 2.0, 20),
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> dict:
        """
        Calibrates the DRAG π pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        spectator_state : str, optional
            Spectator state.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations to |0> state. Defaults to 4.
        n_turns : int, optional
            Number of turns to |0> state. Defaults to 1.
        n_iterations : int, optional
            Number of iterations. Defaults to 2.
        degree : int, optional
            Degree of the polynomial to fit. Defaults to 3.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        calibrate_beta : bool, optional
            Whether to calibrate the DRAG beta. Defaults to False.
        beta_range : ArrayLike, optional
            Range of the beta to sweep. Defaults to np.linspace(-2.0, 2.0, 20).
        duration : float, optional
            Duration of the pulse.
        drag_coeff : float, optional
            DRAG coefficient. Defaults to DRAG_COEFF.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : float, optional
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
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        echo: bool = False,
        control_state: str = "0",
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        ramp_type: Literal[
            "Gaussian",
            "RaisedCosine",
            "Sintegral",
            "Bump",
        ] = "RaisedCosine",
        x180_margin: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        plot: bool = True,
    ) -> dict: ...

    def cr_hamiltonian_tomography(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        x90: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        plot: bool = True,
    ) -> dict: ...

    def update_cr_params(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        update_cr_phase: bool = True,
        update_cancel_pulse: bool = True,
        x90: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        plot: bool = True,
    ) -> dict: ...

    def obtain_cr_params(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        n_iterations: int = 4,
        n_cycles: int = 2,
        n_points_per_cycle: int = 6,
        use_stored_params: bool = False,
        tolerance: float = 0.005e-3,
        adiabatic_safe_factor: float | None = None,
        max_amplitude: float = 1.0,
        max_time_range: float = 4096.0,
        x90: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        plot: bool = True,
    ) -> dict: ...

    def calibrate_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        ramptime: float | None = None,
        duration: float | None = None,
        amplitude_range: ArrayLike | None = None,
        initial_state: str = "0",
        degree: int = 3,
        adiabatic_safe_factor: float | None = None,
        max_amplitude: float = 1.0,
        rotary_multiple: float = 9.0,
        use_drag: bool = True,
        duration_unit: float = 16.0,
        duration_buffer: float = 1.05,
        n_repetitions: int = 1,
        x180: TargetMap[Waveform] | Waveform | None = None,
        x180_margin: float = 0.0,
        use_zvalues: bool = False,
        store_params: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict: ...

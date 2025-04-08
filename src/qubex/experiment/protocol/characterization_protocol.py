from __future__ import annotations

from typing import Collection, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...analysis.fitting import RabiParam
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import Waveform
from ...typing import TargetMap
from ..experiment_constants import CALIBRATION_SHOTS, RABI_FREQUENCY, RABI_TIME_RANGE
from ..experiment_result import (
    AmplRabiData,
    ExperimentResult,
    FreqRabiData,
    RamseyData,
    T1Data,
    T2Data,
)


class CharacterizationProtocol(Protocol):
    def measure_readout_snr(
        self,
        targets: Collection[str] | str | None = None,
        *,
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict:
        """
        Measures the readout SNR of the given targets.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to measure the readout SNR.
        initial_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Initial state of the qubits. Defaults to None.
        capture_window : int, optional
            Capture window. Defaults to None.
        capture_margin : int, optional
            Capture margin. Defaults to None.
        readout_duration : int, optional
            Readout duration. Defaults to None.
        readout_amplitudes : dict[str, float], optional
            Readout amplitudes for each target.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict
            Readout SNR of the targets.

        Examples
        --------
        >>> result = ex.measure_readout_snr(["Q00", "Q01"])
        """
        ...

    def sweep_readout_amplitude(
        self,
        targets: Collection[str] | str | None = None,
        *,
        amplitude_range: ArrayLike = np.linspace(0.0, 0.1, 21),
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        """
        Sweeps the readout amplitude of the given targets.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to sweep the readout amplitude. Defaults to None.
        amplitude_range : ArrayLike, optional
            Range of the readout amplitude to sweep. Defaults to np.linspace(0.0, 1.0, 21).
        initial_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Initial state of the qubits. Defaults to None.
        capture_window : int, optional
            Capture window. Defaults to None.
        capture_margin : int, optional
            Capture margin. Defaults to None.
        readout_duration : int, optional
            Readout duration. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict
            Readout SNR of the targets.
        """
        ...

    def sweep_readout_duration(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = np.arange(128, 2048, 128),
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        capture_margin: int | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        """
        Sweeps the readout duration of the given targets.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to sweep the readout duration. Defaults to None.
        time_range : ArrayLike, optional
            Time range of the readout duration to sweep. Defaults to np.arange(0, 2048, 128).
        initial_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Initial state of the qubits. Defaults to None.
        capture_margin : int, optional
            Capture margin. Defaults to None.
        readout_amplitudes : dict[str, float], optional
            Readout amplitudes for each target. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict
            Readout SNR of the targets.
        """
        ...

    def chevron_pattern(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.05, 0.05, 51),
        time_range: ArrayLike = RABI_TIME_RANGE,
        frequencies: dict[str, float] | None = None,
        amplitudes: dict[str, float] | None = None,
        rabi_params: dict[str, RabiParam] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict: ...

    def obtain_freq_rabi_relation(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = np.arange(0, 101, 4),
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[FreqRabiData]:
        """
        Obtains the relation between the detuning and the Rabi frequency.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the Rabi oscillation.
        detuning_range : ArrayLike, optional
            Range of the detuning to sweep in GHz.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        rabi_level : Literal["ge", "ef"], optional
            Rabi level to use. Defaults to "ge".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[FreqRabiData]
            Result of the experiment.

        Raises
        ------
        ValueError
            If the Rabi parameters are not stored.

        Examples
        --------
        >>> result = ex.obtain_freq_rabi_relation(
        ...     targets=["Q00", "Q01"],
        ...     detuning_range=np.linspace(-0.01, 0.01, 11),
        ...     time_range=range(0, 101, 4),
        ... )
        """
        ...

    def obtain_ampl_rabi_relation(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        amplitude_range: ArrayLike = np.linspace(0.01, 0.1, 10),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[AmplRabiData]:
        """
        Obtains the relation between the control amplitude and the Rabi frequency.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the Rabi oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns. Defaults to RABI_TIME_RANGE.
        amplitude_range : ArrayLike, optional
            Range of the control amplitude to sweep.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[AmplRabiData]
            Result of the experiment.

        Raises
        ------
        ValueError
            If the Rabi parameters are not stored.

        Examples
        --------
        >>> result = ex.obtain_ampl_rabi_relation(
        ...     targets=["Q00", "Q01"],
        ...     amplitude_range=np.linspace(0.01, 0.1, 10),
        ...     time_range=range(0, 201, 4),
        ... )
        """
        ...

    def calibrate_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = range(0, 101, 4),
        frequencies: dict[str, float] | None = None,
        amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]: ...

    def calibrate_ef_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = np.arange(0, 101, 4),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]: ...

    def calibrate_readout_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = range(0, 101, 4),
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]: ...

    def t1_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
        xaxis_type: Literal["linear", "log"] = "log",
    ) -> ExperimentResult[T1Data]:
        """
        Conducts a T1 experiment in parallel.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Collection of qubits to check the T1 decay.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the images. Defaults to False.

        Returns
        -------
        ExperimentResult[T1Data]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.t1_experiment(
        ...     targets=["Q00", "Q01", "Q02", "Q03"]
        ...     time_range=2 ** np.arange(1, 19),
        ...     shots=1024,
        ... )
        """
        ...

    def t2_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        n_cpmg: int = 1,
        pi_cpmg: Waveform | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> ExperimentResult[T2Data]:
        """
        Conducts a T2 experiment in series.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the T2 decay.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        n_cpmg : int, optional
            Number of CPMG pulses. Defaults to 1.
        pi_cpmg : Waveform, optional
            π pulse for the CPMG sequence. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the images. Defaults to False.

        Returns
        -------
        ExperimentResult[T2Data]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.t2_experiment(
        ...     targets=["Q00", "Q01", "Q02", "Q03"]
        ...     time_range=2 ** np.arange(1, 19),
        ...     shots=1024,
        ... )
        """
        ...

    def ramsey_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = np.arange(0, 10_001, 100),
        detuning: float = 0.001,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> ExperimentResult[RamseyData]:
        """
        Conducts a Ramsey experiment in series.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the Ramsey oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns. Defaults to np.arange(0, 10001, 100).
        detuning : float, optional
            Detuning of the control frequency. Defaults to 0.001 GHz.
        spectator_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Spectator state. Defaults to "0".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the images. Defaults to False.

        Returns
        -------
        ExperimentResult[RamseyData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.ramsey_experiment(
        ...     targets=["Q00", "Q01", "Q02", "Q03"]
        ...     time_range=range(0, 10_000, 100),
        ...     shots=1024,
        ... )
        """
        ...

    def obtain_effective_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = np.arange(0, 10001, 100),
        detuning: float = 0.001,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        """
        Obtains the effective control frequency of the qubit.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to check the Ramsey oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        detuning : float, optional
            Detuning of the control frequency. Defaults to 0.001 GHz.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict
            Effective control frequency.

        Examples
        --------
        >>> result = ex.obtain_true_control_frequency(
        ...     targets=["Q00", "Q01", "Q02", "Q03"]
        ...     time_range=range(0, 10001, 100),
        ...     shots=1024,
        ... )
        """
        ...

    def jazz_experiment(
        self,
        target_qubit: str,
        spectator_qubit: str,
        *,
        time_range: ArrayLike = np.arange(0, 2001, 100),
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ):
        """
        Conducts a JAZZ experiment.

        Parameters
        ----------
        target_qubit : str
            Target qubit.
        spectator_qubit : str
            Spectator qubit.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        x90 : TargetMap[Waveform], optional
            X90 pulse for each qubit.
        x180 : TargetMap[Waveform], optional
            X180 pulse for each qubit.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict
            - "xi" : float
                Coefficient of ZZ/2 in GHz.
            - "zeta" : float
                Coefficient of ZZ/4 in GHz.
        """
        ...

    def obtain_coupling_strength(
        self,
        target_qubit: str,
        spectator_qubit: str,
        *,
        time_range: ArrayLike = np.arange(0, 5001, 200),
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        """
        Obtains the coupling strength between the target and spectator qubits.

        Parameters
        ----------
        target_qubit : str
            Target qubit.
        spectator_qubit : str
            Spectator qubit.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        x90 : TargetMap[Waveform], optional
            X90 pulse for each qubit.
        x180 : TargetMap[Waveform], optional
            X180 pulse for each qubit.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict
            - "xi" : float
                Coefficient of ZZ/2 in GHz.
            - "zeta" : float
                Coefficient of ZZ/4 in GHz.
            - "g" : float
                Coupling strength in GHz.
        """
        ...

    def measure_phase_shift(
        self,
        target: str,
        *,
        frequency_range: ArrayLike = np.arange(10.05, 10.1, 0.002),
        amplitude: float = 0.01,
        subrange_width: float = 0.3,
        shots: int = 128,
        interval: int = 0,
        plot: bool = True,
    ) -> float:
        """
        Measures the phase shift caused by the length of the transmission line.

        Parameters
        ----------
        target : str
            Target qubit connected to the resonator of interest.
        frequency_range : ArrayLike, optional
            Frequency range of the scan in GHz.
        amplitude : float, optional
            Amplitude of the readout pulse. Defaults to 0.01.
        shots : int, optional
            Number of shots. Defaults to 128.
        interval : int, optional
            Interval between shots. Defaults to 0.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        float
            Phase shift in rad/GHz.
        """
        ...

    def scan_resonator_frequencies(
        self,
        target: str,
        *,
        frequency_range: ArrayLike = np.arange(9.75, 10.75, 0.002),
        amplitude: float = 0.01,
        phase_shift: float | None = None,
        subrange_width: float = 0.3,
        shots: int = DEFAULT_SHOTS,
        interval: int = 0,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict:
        """
        Scans the readout frequencies to find the resonator frequencies.

        Parameters
        ----------
        target : str
            Target qubit connected to the resonator of interest.
        frequency_range : ArrayLike, optional
            Frequency range of the scan in GHz.
        amplitude : float, optional
            Amplitude of the readout pulse. Defaults to 0.01.
        phase_shift : float, optional
            Phase shift in rad/GHz. If None, it will be measured.
        subrange_width : float, optional
            Width of the frequency subrange in GHz. Defaults to 0.3.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to 0.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the plot as an image. Defaults to False.

        Returns
        -------
        dict
            Results of the experiment.
        """
        ...

    def resonator_spectroscopy(
        self,
        target: str,
        *,
        frequency_range: ArrayLike = np.arange(9.75, 10.75, 0.002),
        power_range: ArrayLike = np.arange(-60, 5, 5),
        phase_shift: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = 0,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        """
        Conducts a resonator spectroscopy experiment.

        Parameters
        ----------
        target : str
            Target qubit connected to the resonator of interest.
        frequency_range : ArrayLike, optional
            Frequency range of the scan in GHz. Defaults to np.arange(9.75, 10.75, 0.002).
        power_range : ArrayLike, optional
            Power range in dB. Defaults to np.arange(-60, 5, 5).
        phase_shift : float, optional
            Phase shift in rad/GHz.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to 0.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the image. Defaults to True.

        Returns
        -------
        dict
            Results of the experiment.
        """
        ...

    def measure_reflection_coefficient(
        self,
        target: str,
        *,
        frequency_range: ArrayLike,
        amplitude: float = 0.01,
        phase_shift: float,
        shots: int = DEFAULT_SHOTS,
        interval: int = 0,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        """
        Scans the readout frequencies to find the resonator frequencies.

        Parameters
        ----------
        target : str
            Target qubit connected to the resonator of interest.
        frequency_range : ArrayLike
            Frequency range of the scan in GHz.
        amplitude : float, optional
            Amplitude of the readout pulse. Defaults to 0.01.
        phase_shift : float
            Phase shift in rad/GHz.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to 0.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the image. Defaults to True.

        Returns
        -------
        dict
        """
        ...

    def scan_qubit_frequencies(
        self,
        target: str,
        *,
        frequency_range: ArrayLike = np.arange(6.5, 9.5, 0.002),
        control_amplitude: float = 0.1,
        readout_amplitude: float = 0.01,
        readout_frequency: float | None = None,
        subrange_width: float = 0.3,
        shots: int = DEFAULT_SHOTS,
        interval: int = 0,
        plot: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Scans the control frequencies to find the qubit frequencies.

        Parameters
        ----------
        target : str
            Target qubit.
        frequency_range : ArrayLike, optional
            Frequency range of the scan in GHz.
        control_amplitude : float, optional
            Amplitude of the control pulse. Defaults to 0.1.
        readout_amplitude : float, optional
            Amplitude of the readout pulse. Defaults to 0.01.
        subrange_width : float, optional
            Width of the frequency subrange in GHz. Defaults to 0.3.
        shots : int, optional
            Number of shots.
        interval : int, optional
            Interval between shots.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
            Frequency range, unwrapped phase and amplitude.
        """
        ...

    def estimate_control_amplitude(
        self,
        target: str,
        *,
        frequency_range: ArrayLike,
        control_amplitude: float = 0.01,
        target_rabi_rate: float = RABI_FREQUENCY,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ): ...

    def qubit_spectroscopy(
        self,
        target: str,
        frequency_range: ArrayLike = np.arange(6.5, 9.5, 0.002),
        power_range: ArrayLike = np.arange(-60, 5, 5),
        readout_amplitude: float = 0.01,
        readout_frequency: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = 0,
        plot: bool = True,
        save_image: bool = True,
    ) -> NDArray[np.float64]:
        """
        Conducts a qubit spectroscopy experiment.

        Parameters
        ----------
        target : str
            Target qubit.
        frequency_range : ArrayLike, optional
            Frequency range of the scan in GHz. Defaults to np.arange(6.5, 9.5, 0.002).
        power_range : ArrayLike, optional
            Power range in dB. Defaults to np.arange(-60, 5, 5).
        readout_amplitude : float, optional
            Amplitude of the readout pulse. Defaults to 0.01.
        readout_frequency : float, optional
            Readout frequency. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to 0.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the image. Defaults to True.

        Returns
        -------
        NDArray[np.float64]
            Phase in rad.
        """
        ...

from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Final

import numpy as np
from numpy.typing import NDArray

from .analysis import fit_damped_rabi, fit_rabi
from .config import Config, Params, Qubit
from .measurement import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONTROL_WINDOW,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    Measurement,
    MeasurementResult,
)
from .pulse import Rect, Waveform


@dataclass
class RabiParams:
    """
    Data class representing the parameters of Rabi oscillation.

    This class is used to store the parameters of Rabi oscillation, which is
    obtained by fitting the measured data. The parameters are used to normalize
    the measured data.

    Attributes
    ----------
    qubit : str
        Identifier of the qubit.
    phase_shift : float
        Phase shift of the I/Q signal.
    fluctuation : float
        Fluctuation of the I/Q signal.
    amplitude : float
        Amplitude of the Rabi oscillation.
    omega : float
        Angular frequency of the Rabi oscillation.
    phi : float
        Phase of the Rabi oscillation.
    offset : float
        Offset of the Rabi oscillation.
    """

    qubit: str
    phase_shift: float
    fluctuation: float
    amplitude: float
    omega: float
    phi: float
    offset: float


@dataclass
class SweepResult:
    """
    Data class representing the result of a sweep experiment.

    This class is used to store the result of a sweep experiment. The result
    includes the sweep range, the measured signals, and the time when the
    experiment is conducted.

    Attributes
    ----------
    qubit : str
        Identifier of the qubit.
    sweep_range : NDArray
        Sweep range of the experiment.
    signals : NDArray
        Measured signals.
    created_at : str
        Time when the experiment is conducted.
    """

    qubit: str
    sweep_range: NDArray
    signals: NDArray
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def rotated(self, rabi_params: RabiParams) -> NDArray:
        """Returns the measured signals after rotating them by the phase shift."""
        return self.signals * np.exp(-1j * rabi_params.phase_shift)

    def normalized(self, rabi_params: RabiParams) -> NDArray:
        """Returns the normalized measured signals."""
        values = self.signals * np.exp(-1j * rabi_params.phase_shift)
        values_normalized = -(values.imag - rabi_params.offset) / rabi_params.amplitude
        return values_normalized


class Experiment:
    """
    Manages and conducts a variety of quantum experiments using QuBE devices.

    This class serves as a central point for setting up, executing, and analyzing
    quantum experiments. It supports various types of experiments like Rabi
    experiment and Ramsey experiment.

    Parameters
    ----------
    chip_id : str
        Identifier of the quantum chip.
    control_window : int, optional
        Duration of the control window in nanoseconds. Defaults to DEFAULT_CONTROL_WINDOW.
    data_dir : str, optional
        Path to the directory where the experiment data is stored. Defaults to "./data".
    """

    def __init__(
        self,
        *,
        chip_id: str,
        config_dir: str = DEFAULT_CONFIG_DIR,
        control_window: int = DEFAULT_CONTROL_WINDOW,
        data_dir="./data",
    ):
        self._chip_id: Final = chip_id
        self._data_dir: Final = data_dir
        self._measurement: Final = Measurement(
            chip_id=chip_id,
            config_dir=config_dir,
        )
        self.config: Final = Config(config_dir=config_dir)

    @property
    def qubits(self) -> list[Qubit]:
        """Get the list of qubits."""
        return self.config.get_qubits(self._chip_id)

    @property
    def params(self) -> Params:
        """Get the system parameters."""
        return self.config.get_params(self._chip_id)

    def measure(
        self,
        waveforms: dict[str, list | NDArray],
        *,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> MeasurementResult:
        result = self._measurement.measure(
            waveforms=waveforms,
            shots=shots,
            interval=interval,
        )
        return result

    def rabi_experiment(
        self,
        *,
        time_range: NDArray,
        amplitudes: dict[str, float],
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, SweepResult]:
        """
        Conducts a Rabi experiment.

        Parameters
        ----------
        time_range : NDArray
            Time range of the experiment.
        amplitudes : dict[str, float],
            Amplitudes of the control pulses.
        plot : bool, optional
            Whether to plot the results. Defaults to True.

        Returns
        -------
        dict[str, SweepResult]
            Result of the experiment.

        Examples
        --------
        >>> from qubex import Experiment
        >>> exp = Experiment(config_file="config.json")
        >>> exp.connect()
        >>> exp.rabi_experiment(
        ...     time_range=np.arange(0, 201, 10),
        ...     amplitudes={
        ...         "Q01": 0.5,
        ...         "Q02": 0.5,
        ...     },
        ... )
        {"Q01": SweepResult(...), "Q02": SweepResult(...)}
        """
        qubits = list(amplitudes.keys())
        signals: dict[str, list[complex]] = defaultdict(list)

        for index, duration in enumerate(time_range):
            waveforms = {
                qubit: Rect(
                    duration=duration,
                    amplitude=amplitudes[qubit],
                ).values
                for qubit in qubits
            }

            measured_values = self.measure(
                waveforms=waveforms,
                shots=shots,
                interval=interval,
            ).data

            for qubit, value in measured_values.items():
                signals[qubit].append(value)

            # if plot:

            print(f"{index+1}/{len(time_range)} : {duration} ns")

        result = {
            qubit: SweepResult(
                qubit=qubit,
                sweep_range=time_range,
                signals=np.array(values),
            )
            for qubit, values in signals.items()
        }
        return result

    def sweep_parameter(
        self,
        sweep_range: NDArray,
        parametric_waveforms: dict[str, Callable[..., Waveform]],
        pulse_count=1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, SweepResult]:
        """
        Conducts a sweep experiment.

        Parameters
        ----------
        sweep_range : NDArray
            Sweep range of the experiment.
        parametric_waveforms : dict[str, Callable[..., Waveform]]
            Parametric waveforms to apply to the qubits.
        pulse_count : int, optional
            Number of pulses to apply. Defaults to 1.
        plot : bool, optional
            Whether to plot the results. Defaults to True.

        Returns
        -------
        dict[str, SweepResult]
            Result of the experiment.
        """
        signals: dict[str, list[complex]] = defaultdict(list)

        for index, param in enumerate(sweep_range):
            waveforms = {
                qubit: waveform(param).repeated(pulse_count).values
                for qubit, waveform in parametric_waveforms.items()
            }

            measured_values = self.measure(
                waveforms=waveforms,
                shots=shots,
                interval=interval,
            ).data

            for qubit, value in measured_values.items():
                signals[qubit].append(value)

            # if plot:

            print(f"{index+1}/{len(sweep_range)}")

        result = {
            qubit: SweepResult(
                qubit=qubit,
                sweep_range=sweep_range,
                signals=np.array(values),
            )
            for qubit, values in signals.items()
        }
        return result

    def rabi_check(
        self,
        time_range=np.arange(0, 201, 10),
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict[str, SweepResult]:
        """
        Conducts a Rabi experiment with the default HPI amplitude.

        Parameters
        ----------
        time_range : NDArray, optional
            Time range of the experiment. Defaults to np.arange(0, 201, 10).

        Returns
        -------
        dict[str, SweepResult]
            Result of the experiment.
        """
        amplitudes = self.params.control_amplitude
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            shots=shots,
            interval=interval,
        )
        return result

    def repeat_pulse(
        self,
        waveforms: dict[str, Waveform],
        n: int,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict[str, SweepResult]:
        """
        Repeats the given pulse n times.

        Parameters
        ----------
        waveforms : dict[str, Waveform]
            Waveforms to apply to the qubits.
        n : int
            Number of times to repeat the pulse.

        Returns
        -------
        dict[str, SweepResult]
            Result of the experiment.
        """
        parametric_waveforms = {
            qubit: lambda param, w=waveform: w.repeated(int(param))
            for qubit, waveform in waveforms.items()
        }
        result = self.sweep_parameter(
            sweep_range=np.arange(n + 1),
            parametric_waveforms=parametric_waveforms,
            pulse_count=1,
            shots=shots,
            interval=interval,
        )
        return result

    def normalize(
        self,
        iq_value: NDArray,
        rabi_params: RabiParams,
    ) -> float:
        """
        Normalizes the measured IQ value.

        Parameters
        ----------
        iq_value : NDArray
            Measured IQ value.
        rabi_params : RabiParams
            Parameters of the Rabi oscillation.

        Returns
        -------
        float
            Normalized value.
        """
        iq_value = iq_value * np.exp(-1j * rabi_params.phase_shift)
        value = iq_value.imag
        value = -(value - rabi_params.offset) / rabi_params.amplitude
        return value

    def fit_rabi(
        self,
        data: SweepResult,
        wave_count=2.5,
    ) -> RabiParams:
        """
        Fits the measured data to a Rabi oscillation.

        Parameters
        ----------
        data : SweepResult
            Measured data.
        wave_count : float, optional
            Number of waves to fit. Defaults to 2.5.

        Returns
        -------
        RabiParams
            Parameters of the Rabi oscillation.
        """
        times = data.sweep_range
        signals = data.signals

        phase_shift, fluctuation, popt = fit_rabi(
            times=times,
            signals=signals,
            wave_count=wave_count,
        )

        rabi_params = RabiParams(
            qubit=data.qubit,
            phase_shift=phase_shift,
            fluctuation=fluctuation,
            amplitude=popt[0],
            omega=popt[1],
            phi=popt[2],
            offset=popt[3],
        )
        return rabi_params

    def fit_damped_rabi(
        self,
        data: SweepResult,
        wave_count=2.5,
    ) -> RabiParams:
        """
        Fits the measured data to a damped Rabi oscillation.

        Parameters
        ----------
        data : SweepResult
            Measured data.
        wave_count : float, optional
            Number of waves to fit. Defaults to 2.5.

        Returns
        -------
        RabiParams
            Parameters of the Rabi oscillation.
        """
        times = data.sweep_range
        signals = data.signals

        phase_shift, fluctuation, popt = fit_damped_rabi(
            times=times,
            signals=signals,
            wave_count=wave_count,
        )

        rabi_params = RabiParams(
            qubit=data.qubit,
            phase_shift=phase_shift,
            fluctuation=fluctuation,
            amplitude=popt[0],
            omega=popt[2],
            phi=popt[3],
            offset=popt[4],
        )
        return rabi_params

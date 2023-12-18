"""
Provides a comprehensive framework for conducting quantum experiments on QuBE
devices. This module includes functionalities for setting up experiments, 
measuring quantum states, and analyzing results.
"""

import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Final

import numpy as np
from IPython.display import clear_output
from numpy.typing import NDArray

from .analysis import fit_and_rotate, fit_chevron, fit_damped_rabi, fit_rabi
from .configs import Configs
from .consts import T_CONTROL, T_READOUT
from .experiment_record import ExperimentRecord
from .pulse import Rect, Waveform
from .qube_manager import QubeManager
from .typing import (
    FloatArray,
    IntArray,
    IQArray,
    IQValue,
    ParametricWaveform,
    QubitDict,
    QubitKey,
)
from .visualization import show_measurement_results, show_pulse_sequences


@dataclass
class RabiParams:
    """
    Data class representing the parameters of Rabi oscillation.

    This class is used to store the parameters of Rabi oscillation, which is
    obtained by fitting the measured data. The parameters are used to normalize
    the measured data.

    Attributes
    ----------
    qubit : QubitKey
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

    qubit: QubitKey
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
    qubit : QubitKey
        Identifier of the qubit.
    sweep_range : NDArray
        Sweep range of the experiment.
    signals : IQArray
        Measured signals.
    created_at : str
        Time when the experiment is conducted.
    """

    qubit: QubitKey
    sweep_range: NDArray
    signals: IQArray
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def rotated(self, rabi_params: RabiParams) -> IQArray:
        """Returns the measured signals after rotating them by the phase shift."""
        return self.signals * np.exp(-1j * rabi_params.phase_shift)

    def normalized(self, rabi_params: RabiParams) -> FloatArray:
        """Returns the normalized measured signals."""
        values = self.signals * np.exp(-1j * rabi_params.phase_shift)
        values_normalized = -(values.imag - rabi_params.offset) / rabi_params.amplitude
        return values_normalized


@dataclass
class ChevronResult:
    """
    Data class representing the result of a chevron experiment.

    This class is used to store the result of a chevron experiment. The result
    includes the sweep range, the measured signals, and the time when the
    experiment is conducted.

    Attributes
    ----------
    qubit : QubitKey
        Identifier of the qubit.
    center_frequency : float
        Center frequency of the qubit.
    freq_range : FloatArray
        Frequency range of the experiment.
    time_range : IntArray
        Time range of the experiment.
    signals : list[IQValue]
        Measured signals.
    created_at : str
        Time when the experiment is conducted.
    """

    qubit: QubitKey
    center_frequency: float
    freq_range: FloatArray
    time_range: IntArray
    signals: list[FloatArray]
    created_at: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Experiment:
    """
    Manages and conducts a variety of quantum experiments using QuBE devices.

    This class serves as a central point for setting up, executing, and analyzing
    quantum experiments. It supports various types of experiments like Rabi
    experiment and Ramsey experiment.

    Parameters
    ----------
    config_file : str
        Path to the configuration file.
    readout_window : int, optional
        Duration of the readout window in nanoseconds. Defaults to T_READOUT.
    control_window : int, optional
        Duration of the control window in nanoseconds. Defaults to T_CONTROL.
    data_dir : str, optional
        Path to the directory where the experiment data is stored. Defaults to
        "./data".

    Attributes
    ----------
    configs : Configs
        Configurations of the QuBE device.
    qube_manager : QubeManager
        Manager of the QuBE device.
    qubits : list[QubitKey]
        Identifiers of the qubits.
    params : Params
        Parameters of the qubits.
    data_dir : str
        Path to the directory where the experiment data is stored.
    """

    def __init__(
        self,
        config_file: str,
        readout_window: int = T_READOUT,
        control_window: int = T_CONTROL,
        data_dir="./data",
    ):
        self.configs: Final = Configs.load(config_file)
        self.qube_manager: Final = QubeManager(
            configs=self.configs,
            readout_window=readout_window,
            control_window=control_window,
        )
        self.qubits: Final = self.configs.qubits
        self.params: Final = self.configs.params
        self.data_dir: Final = data_dir

    def connect(self, ui: bool = True):
        """
        Connects to the QuBE device.

        Parameters
        ----------
        ui : bool, optional
            Whether to show the UI. Defaults to True.
        """
        self.qube_manager.connect(ui=ui)

    def loopback_mode(self, use_loopback: bool):
        """
        Sets the loopback mode.

        Parameters
        ----------
        use_loopback : bool
            Whether to use the loopback mode.
        """
        self.qube_manager.loopback_mode(use_loopback)

    def set_readout_range(self, readout_range: slice):
        """Sets the readout range."""
        self.qube_manager.readout_range = readout_range

    def get_control_frequency(self, qubit: QubitKey) -> float:
        """Returns the control frequency of the qubit."""
        return self.qube_manager.get_control_frequency(qubit)

    def set_control_frequency(self, qubit: QubitKey, frequency: float):
        """Sets the control frequency of the qubit."""
        self.qube_manager.set_control_frequency(qubit, frequency)

    def measure(
        self,
        waveforms: QubitDict[Waveform],
        repeats: int = 10_000,
        interval: int = 150_000,
    ) -> QubitDict[IQValue]:
        """
        Measures the quantum state of the qubits.

        Parameters
        ----------
        waveforms : QubitDict[Waveform]
            Waveforms to apply to the qubits.
        repeats : int, optional
            Number of times to repeat the experiment. Defaults to 10_000.
        interval : int, optional
            Interval between each experiment in nanoseconds. Defaults to 150_000.

        Returns
        -------
        QubitDict[IQValue]
            Measured values of the qubits.

        Examples
        --------
        >>> from qubex import Experiment
        >>> experiment = Experiment(config_file="config.json")
        >>> experiment.connect()
        >>> experiment.measure(
        ...     waveforms={
        ...         "Q01": Rect(duration=20, amplitude=0.5),
        ...         "Q02": Rect(duration=20, amplitude=0.5),
        ...     },
        ...     repeats=10_000,
        ...     interval=150_000,
        ... )
        {"Q01": (0.0005+0.0005j), "Q02": (0.0005+0.0005j)}
        """
        qubits = list(waveforms.keys())
        result = self.qube_manager.measure(
            control_qubits=qubits,
            readout_qubits=qubits,
            control_waveforms=waveforms,
            repeats=repeats,
            interval=interval,
        )
        return result

    def rabi_experiment(
        self,
        time_range: IntArray,
        amplitudes: QubitDict[float],
        plot: bool = True,
    ) -> QubitDict[SweepResult]:
        """
        Conducts a Rabi experiment.

        Parameters
        ----------
        time_range : IntArray
            Time range of the experiment.
        amplitudes : QubitDict[float]
            Amplitudes of the control pulses.
        plot : bool, optional
            Whether to plot the results. Defaults to True.

        Returns
        -------
        QubitDict[SweepResult]
            Result of the experiment.

        Examples
        --------
        >>> from qubex import Experiment
        >>> experiment = Experiment(config_file="config.json")
        >>> experiment.connect()
        >>> experiment.rabi_experiment(
        ...     time_range=np.arange(0, 201, 10),
        ...     amplitudes={
        ...         "Q01": 0.5,
        ...         "Q02": 0.5,
        ...     },
        ... )
        {"Q01": SweepResult(...), "Q02": SweepResult(...)}
        """
        qubits = list(amplitudes.keys())
        control_qubits = qubits
        readout_qubits = qubits
        signals: QubitDict[list[IQValue]] = defaultdict(list)

        for index, duration in enumerate(time_range):
            waveforms = {
                qubit: Rect(
                    duration=duration,
                    amplitude=amplitudes[qubit],
                )
                for qubit in qubits
            }

            measured_values = self.measure(waveforms)

            for qubit, value in measured_values.items():
                signals[qubit].append(value)

            if plot:
                self.show_experiment_results(
                    control_qubits=control_qubits,
                    readout_qubits=readout_qubits,
                    sweep_range=time_range,
                    index=index,
                    signals=signals,
                )

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
        parametric_waveforms: QubitDict[ParametricWaveform],
        pulse_count=1,
        plot: bool = True,
    ) -> QubitDict[SweepResult]:
        """
        Conducts a sweep experiment.

        Parameters
        ----------
        sweep_range : NDArray
            Sweep range of the experiment.
        parametric_waveforms : QubitDict[ParametricWaveform]
            Parametric waveforms to apply to the qubits.
        pulse_count : int, optional
            Number of pulses to apply. Defaults to 1.
        plot : bool, optional
            Whether to plot the results. Defaults to True.

        Returns
        -------
        QubitDict[SweepResult]
            Result of the experiment.
        """
        qubits = list(parametric_waveforms.keys())
        control_qubits = qubits
        readout_qubits = qubits
        signals: QubitDict[list[IQValue]] = defaultdict(list)

        for index, param in enumerate(sweep_range):
            waveforms = {
                qubit: waveform(param).repeated(pulse_count)
                for qubit, waveform in parametric_waveforms.items()
            }

            measured_values = self.measure(waveforms)

            for qubit, value in measured_values.items():
                signals[qubit].append(value)

            if plot:
                self.show_experiment_results(
                    control_qubits=control_qubits,
                    readout_qubits=readout_qubits,
                    sweep_range=sweep_range,
                    index=index,
                    signals=signals,
                )

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
    ) -> QubitDict[SweepResult]:
        """
        Conducts a Rabi experiment with the default HPI amplitude.

        Parameters
        ----------
        time_range : IntArray, optional
            Time range of the experiment. Defaults to np.arange(0, 201, 10).

        Returns
        -------
        QubitDict[SweepResult]
            Result of the experiment.
        """
        amplitudes = self.params.default_hpi_amplitude
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
        )
        return result

    def repeat_pulse(
        self,
        waveforms: QubitDict[Waveform],
        n: int,
    ) -> QubitDict[SweepResult]:
        """
        Repeats the given pulse n times.

        Parameters
        ----------
        waveforms : QubitDict[Waveform]
            Waveforms to apply to the qubits.
        n : int
            Number of times to repeat the pulse.

        Returns
        -------
        QubitDict[SweepResult]
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
        )
        return result

    def show_experiment_results(
        self,
        control_qubits: list[QubitKey],
        readout_qubits: list[QubitKey],
        sweep_range: NDArray,
        index: int,
        signals: QubitDict[list[IQValue]],
    ):
        """
        Plots the results of the experiment.

        Parameters
        ----------
        control_qubits : list[QubitKey]
            Identifiers of the control qubits.
        readout_qubits : list[QubitKey]
            Identifiers of the readout qubits.
        sweep_range : NDArray
            Sweep range of the experiment.
        index : int
            Index of the sweep range.
        signals : QubitDict[list[IQValue]]
            Measured signals.
        """
        signals_rotated = {
            qubit: fit_and_rotate(values) for qubit, values in signals.items()
        }

        ctrl_waveforms = self.qube_manager.get_control_waveforms(control_qubits)
        ctrl_times = self.qube_manager.get_control_times(control_qubits)
        rotx_waveforms = self.qube_manager.get_readout_tx_waveforms(readout_qubits)
        rotx_times = self.qube_manager.get_readout_tx_times(readout_qubits)
        rorx_waveforms = self.qube_manager.get_readout_rx_waveforms(readout_qubits)
        rorx_times = self.qube_manager.get_readout_rx_times(readout_qubits)
        readout_range = self.qube_manager.readout_range
        control_duration = self.qube_manager.control_window
        readout_duration = self.qube_manager.readout_window

        clear_output(True)
        show_measurement_results(
            qubits=readout_qubits,
            waveforms=rorx_waveforms,
            times=rorx_times,
            sweep_range=sweep_range[: index + 1],
            signals=signals,
            signals_rotated=signals_rotated,
            readout_range=readout_range,
        )
        show_pulse_sequences(
            control_qubits=control_qubits,
            control_waveforms=ctrl_waveforms,
            control_times=ctrl_times,
            control_duration=control_duration,
            readout_qubits=readout_qubits,
            readout_waveforms=rotx_waveforms,
            readout_times=rotx_times,
            readout_duration=readout_duration,
        )

    def normalize(
        self,
        iq_value: IQValue,
        rabi_params: RabiParams,
    ) -> float:
        """
        Normalizes the measured IQ value.

        Parameters
        ----------
        iq_value : IQValue
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

    def chevron_experiment(
        self,
        qubits: list[QubitKey],
        freq_range: FloatArray,
        time_range: IntArray,
        rabi_params: RabiParams,
    ) -> QubitDict[ChevronResult]:
        """
        Conducts a chevron experiment.

        Parameters
        ----------
        qubits : list[QubitKey]
            Identifiers of the qubits.
        freq_range : FloatArray
            Frequency range of the experiment.
        time_range : IntArray
            Time range of the experiment.
        rabi_params : RabiParams
            Parameters of the Rabi oscillation.

        Returns
        -------
        QubitDict[ChevronResult]
            Result of the experiment.
        """
        amplitudes = self.params.default_hpi_amplitude
        frequenties = self.params.transmon_dressed_frequency_ge

        signals = defaultdict(list)

        for idx, freq in enumerate(freq_range):
            print(f"### {idx+1}/{len(freq_range)}: {freq:.2f} MHz")
            for qubit in self.qubits:
                freq_mod = frequenties[qubit] + freq * 1e6
                self.set_control_frequency(qubit, freq_mod)

            result_rabi = self.rabi_experiment(
                time_range=time_range,
                amplitudes=amplitudes,
                plot=False,
            )

            for qubit in qubits:
                signals[qubit].append(result_rabi[qubit].normalized(rabi_params))

        result = {
            qubit: ChevronResult(
                qubit=qubit,
                center_frequency=frequenties[qubit],
                freq_range=freq_range,
                time_range=time_range,
                signals=values,
            )
            for qubit, values in signals.items()
        }

        ExperimentRecord.create(
            name="result_chevron",
            description="Chevron experiment",
            data=result,
        )

        return result

    def fit_chevron(
        self,
        data: ChevronResult,
    ):
        """
        Fits the measured data to a chevron.

        Parameters
        ----------
        data : ChevronResult
            Measured data.
        """
        fit_chevron(
            center_frequency=data.center_frequency,
            freq_range=data.freq_range,
            time_range=data.time_range,
            signals=data.signals,
        )

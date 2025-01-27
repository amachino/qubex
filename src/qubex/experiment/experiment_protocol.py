from __future__ import annotations

from typing import Collection, ContextManager, Literal, Optional

from numpy.typing import ArrayLike, NDArray
from typing_extensions import Protocol

from ..analysis import RabiParam
from ..backend import (
    Box,
    ControlParams,
    ControlSystem,
    DeviceController,
    ExperimentSystem,
    QuantumSystem,
    Qubit,
    Resonator,
    StateManager,
    Target,
)
from ..clifford import Clifford, CliffordGenerator
from ..measurement import (
    Measurement,
    MeasureResult,
    StateClassifier,
)
from ..measurement.measurement import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from ..pulse import (
    PulseSchedule,
    Waveform,
)
from ..typing import IQArray, ParametricPulseSchedule, ParametricWaveformDict, TargetMap
from .experiment_constants import (
    CALIBRATION_SHOTS,
    RABI_FREQUENCY,
    RABI_TIME_RANGE,
)
from .experiment_note import ExperimentNote
from .experiment_record import ExperimentRecord
from .experiment_result import (
    ExperimentResult,
    RabiData,
    SweepData,
)
from .experiment_util import ExperimentUtil


class ExperimentProtocol(Protocol):
    @property
    def util(self) -> type[ExperimentUtil]:
        """Get the experiment util."""
        ...

    @property
    def measurement(self) -> Measurement:
        """Get the measurement system."""
        ...

    @property
    def state_manager(self) -> StateManager:
        """Get the state manager."""
        ...

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        ...

    @property
    def quantum_system(self) -> QuantumSystem:
        """Get the quantum system."""
        ...

    @property
    def control_system(self) -> ControlSystem:
        """Get the control system."""
        ...

    @property
    def device_controller(self) -> DeviceController:
        """Get the device manager."""
        ...

    @property
    def params(self) -> ControlParams:
        """Get the control parameters."""
        ...

    @property
    def chip_id(self) -> str:
        """Get the chip ID."""
        ...

    @property
    def qubit_labels(self) -> list[str]:
        """Get the list of qubit labels."""
        ...

    @property
    def mux_labels(self) -> list[str]:
        """Get the list of mux labels."""
        ...

    @property
    def qubits(self) -> dict[str, Qubit]:
        """Get the available qubit dict."""
        ...

    @property
    def resonators(self) -> dict[str, Resonator]:
        """Get the available resonator dict."""
        ...

    @property
    def targets(self) -> dict[str, Target]:
        """Get the target dict."""
        ...

    @property
    def available_targets(self) -> dict[str, Target]:
        """Get the available target dict."""
        ...

    @property
    def ge_targets(self) -> dict[str, Target]:
        """Get the available ge target dict."""
        ...

    @property
    def ef_targets(self) -> dict[str, Target]:
        """Get the available ef target dict."""
        ...

    @property
    def cr_targets(self) -> dict[str, Target]:
        """Get the available CR target dict."""
        ...

    @property
    def boxes(self) -> dict[str, Box]:
        """Get the available box dict."""
        ...

    @property
    def box_ids(self) -> list[str]:
        """Get the available box IDs."""
        ...

    @property
    def config_path(self) -> str:
        """Get the path of the configuration file."""
        ...

    @property
    def params_path(self) -> str:
        """Get the path of the parameter file."""
        ...

    @property
    def system_note(self) -> ExperimentNote:
        """Get the system note."""
        ...

    @property
    def note(self) -> ExperimentNote:
        """Get the user note."""
        ...

    @property
    def hpi_pulse(self) -> dict[str, Waveform]:
        """
        Get the default π/2 pulse.

        Returns
        -------
        dict[str, Waveform]
            π/2 pulse.
        """
        ...

    @property
    def pi_pulse(self) -> dict[str, Waveform]:
        """
        Get the default π pulse.

        Returns
        -------
        dict[str, Waveform]
            π pulse.
        """
        ...

    @property
    def drag_hpi_pulse(self) -> dict[str, Waveform]:
        """
        Get the DRAG π/2 pulse.

        Returns
        -------
        dict[str, Waveform]
            DRAG π/2 pulse.
        """
        ...

    @property
    def drag_pi_pulse(self) -> dict[str, Waveform]:
        """
        Get the DRAG π pulse.

        Returns
        -------
        dict[str, Waveform]
            DRAG π pulse.
        """
        ...

    @property
    def ef_hpi_pulse(self) -> dict[str, Waveform]:
        """
        Get the ef π/2 pulse.

        Returns
        -------
        dict[str, Waveform]
            π/2 pulse.
        """
        ...

    @property
    def ef_pi_pulse(self) -> dict[str, Waveform]:
        """
        Get the ef π pulse.

        Returns
        -------
        dict[str, Waveform]
            π/2 pulse.
        """
        ...

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        """Get the Rabi parameters."""
        ...

    @property
    def ge_rabi_params(self) -> dict[str, RabiParam]:
        """Get the ge Rabi parameters."""
        ...

    @property
    def ef_rabi_params(self) -> dict[str, RabiParam]:
        """Get the ef Rabi parameters."""
        ...

    @property
    def classifier_type(self) -> Literal["kmeans", "gmm"]:
        """Get the classifier type."""
        ...

    @property
    def classifiers(self) -> dict[str, StateClassifier]:
        """Get the classifiers."""
        ...

    @property
    def state_centers(self) -> dict[str, dict[int, complex]]:
        """Get the state centers."""
        ...

    @property
    def clifford_generator(self) -> CliffordGenerator:
        """Get the Clifford generator."""
        ...

    @property
    def clifford(self) -> dict[str, Clifford]:
        """Get the Clifford dict."""
        ...

    def validate_rabi_params(
        self,
        targets: Collection[str] | None = None,
    ):
        """Check if the Rabi parameters are stored."""
        ...

    def store_rabi_params(
        self,
        rabi_params: dict[str, RabiParam],
    ):
        """
        Stores the Rabi parameters.

        Parameters
        ----------
        rabi_params : dict[str, RabiParam]
            Parameters of the Rabi oscillation.
        """
        ...

    def get_pulse_for_state(
        self,
        target: str,
        state: str,  # Literal["0", "1", "+", "-", "+i", "-i"],
    ) -> Waveform:
        """
        Get the pulse to prepare the given state from the ground state.

        Parameters
        ----------
        target : str
            Target qubit.
        state : Literal["0", "1", "+", "-", "+i", "-i"]
            State to prepare.

        Returns
        -------
        Waveform
            Pulse for the state.
        """
        ...

    def get_spectators(
        self,
        qubit: str,
        in_same_mux: bool = False,
    ) -> list[Qubit]:
        """
        Get the spectators of the given qubit.

        Parameters
        ----------
        qubit : str
            Qubit to get the spectators.
        in_same_mux : bool, optional
            Whether to get the spectators in the same mux. Defaults to False.

        Returns
        -------
        list[Qubit]
            List of the spectators.
        """
        ...

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        """
        Get the confusion matrix of the given targets.

        Parameters
        ----------
        targets : Collection[str]
            Target labels.

        Returns
        -------
        NDArray
            Confusion matrix (rows: true, columns: predicted).
        """
        ...

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        """
        Get the inverse confusion matrix of the given targets.

        Parameters
        ----------
        targets : Collection[str]
            Target labels.

        Returns
        -------
        NDArray
            Inverse confusion matrix.

        Notes
        -----
        The inverse confusion matrix should be multiplied from the right.

        Examples
        --------
        >>> cm_inv = ex.get_inverse_confusion_matrix(["Q00", "Q01"])
        >>> observed = np.array([300, 200, 200, 300])
        >>> predicted = observed @ cm_inv
        """
        ...

    def check_status(self):
        """Check the status of the measurement system."""
        ...

    def linkup(
        self,
        box_ids: list[str] | None = None,
        noise_threshold: int = 500,
    ) -> None:
        """
        Link up the measurement system.

        Parameters
        ----------
        box_ids : list[str], optional
            List of the box IDs to link up. Defaults to None.

        Examples
        --------
        >>> ex.linkup()
        """
        ...

    def resync_clocks(
        self,
        box_ids: list[str] | None = None,
    ) -> None:
        """
        Resynchronize the clocks of the measurement system.

        Parameters
        ----------
        box_ids : list[str], optional
            List of the box IDs to resynchronize. Defaults to None.

        Examples
        --------
        >>> ex.resync_clocks()
        """
        ...

    def configure(
        self,
        box_ids: list[str] | None = None,
        exclude: list[str] | None = None,
    ):
        """
        Configure the measurement system from the config files.

        Parameters
        ----------
        box_ids : list[str], optional
            List of the box IDs to configure. Defaults to None.

        Examples
        --------
        >>> ex.configure()
        """
        ...

    def reload(self):
        """Reload the configuration files."""
        ...

    def modified_frequencies(
        self,
        frequencies: dict[str, float] | None,
    ) -> ContextManager:
        """
        Temporarily modifies the frequencies of the qubits.

        Parameters
        ----------
        frequencies : dict[str, float]
            Modified frequencies in GHz.

        Examples
        --------
        >>> with ex.modified_frequencies({"Q00": 5.0}):
        ...     # Do something
        """
        ...

    def save_defaults(self):
        """Save the default params."""
        ...

    def clear_defaults(self):
        """Clear the default params."""
        ...

    def delete_defaults(self):
        """Delete the default params."""
        ...

    def load_record(
        self,
        name: str,
    ) -> ExperimentRecord:
        """
        Load an experiment record from a file.

        Parameters
        ----------
        name : str
            Name of the experiment record to load.

        Returns
        -------
        ExperimentRecord
            The loaded ExperimentRecord instance.

        Raises
        ------
        FileNotFoundError

        Examples
        --------
        >>> record = ex.load_record("some_record.json")
        """
        ...

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
        xaxis_title: str = "Sweep value",
        yaxis_title: str = "Measured value",
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
        xaxis_title : str, optional
            Title of the x-axis. Defaults to "Sweep value".
        yaxis_title : str, optional
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

    def calc_control_amplitudes(
        self,
        *,
        rabi_rate: float = RABI_FREQUENCY,
        current_amplitudes: dict[str, float] | None = None,
        current_rabi_params: dict[str, RabiParam] | None = None,
        print_result: bool = True,
    ) -> dict[str, float]:
        """
        Calculates the control amplitudes for the Rabi rate.

        Parameters
        ----------
        rabi_rate : float, optional
            Target Rabi rate in GHz. Defaults to RABI_FREQUENCY.
        current_amplitudes : dict[str, float], optional
            Current control amplitudes. Defaults to None.
        current_rabi_params : dict[str, RabiParam], optional
            Current Rabi parameters. Defaults to None.
        print_result : bool, optional
            Whether to print the result. Defaults to True.

        Returns
        -------
        dict[str, float]
            Control amplitudes for the Rabi rate.
        """
        ...

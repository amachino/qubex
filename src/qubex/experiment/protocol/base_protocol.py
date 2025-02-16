from __future__ import annotations

from typing import Collection, ContextManager, Literal, Protocol, Sequence

from numpy.typing import NDArray

from ...analysis import RabiParam
from ...backend import (
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
from ...clifford import Clifford, CliffordGenerator
from ...measurement import Measurement, StateClassifier
from ...pulse import PulseSchedule, VirtualZ, Waveform
from ...typing import TargetMap
from ..calibration_note import CalibrationNote
from ..experiment_constants import RABI_FREQUENCY
from ..experiment_note import ExperimentNote
from ..experiment_record import ExperimentRecord
from ..experiment_util import ExperimentUtil


class BaseProtocol(Protocol):
    @property
    def drag_hpi_duration(self) -> int:
        """Get the DRAG π/2 duration."""
        ...

    @property
    def drag_pi_duration(self) -> int:
        """Get the DRAG π duration."""
        ...

    @property
    def control_window(self) -> int | None:
        """Get the control window."""
        ...

    @property
    def capture_window(self) -> int:
        """Get the capture window."""
        ...

    @property
    def capture_margin(self) -> int:
        """Get the capture margin."""
        ...

    @property
    def readout_duration(self) -> int:
        """Get the readout duration."""
        ...

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
    def calib_note(self) -> CalibrationNote:
        """Get the calibration note."""
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

    def x90(
        self,
        target: str,
        /,
        *,
        type: Literal["flattop", "drag"] | None = None,
    ) -> Waveform:
        """
        Generate a π/2 pulse along the x-axis.

        Parameters
        ----------
        target : str
            Target qubit.
        type : Literal["flattop", "drag"], optional
            Type of the pulse. Defaults to None.

        Returns
        -------
        Waveform
            π/2 pulse along the x-axis.
        """
        ...

    def x180(
        self,
        target: str,
        /,
        *,
        type: Literal["flattop", "drag"] | None = None,
        use_hpi: bool = False,
    ) -> Waveform:
        """
        Generate a π pulse along the x-axis.

        Parameters
        ----------
        target : str
            Target qubit.
        type : Literal["flattop", "drag"], optional
            Type of the pulse. Defaults to None.
        use_hpi : bool, optional
            Whether to generate the π pulse as π/2 pulse * 2. Defaults to False.

        Returns
        -------
        Waveform
            π pulse along the x-axis.
        """
        ...

    def y90(
        self,
        target: str,
        /,
        *,
        type: Literal["flattop", "drag"] | None = None,
    ) -> Waveform:
        """
        Generate a π/2 pulse along the y-axis.

        Parameters
        ----------
        target : str
            Target qubit.
        type : Literal["flattop", "drag"], optional
            Type of the pulse. Defaults to None.

        Returns
        -------
        Waveform
            π/2 pulse along the y-axis.
        """
        ...

    def y180(
        self,
        target: str,
        /,
        *,
        type: Literal["flattop", "drag"] | None = None,
        use_hpi: bool = False,
    ) -> Waveform:
        """
        Generate a π pulse along the y-axis.

        Parameters
        ----------
        target : str
            Target qubit.
        type : Literal["flattop", "drag"], optional
            Type of the pulse. Defaults to None.
        use_hpi : bool, optional
            Whether to generate the π pulse as π/2 pulse * 2. Defaults to False.

        Returns
        -------
        Waveform
            π pulse along the y-axis.
        """
        ...

    def z90(
        self,
    ) -> VirtualZ:
        """
        Generate a π/2 virtual pulse along the z-axis.

        Returns
        -------
        VirtualZ
            π/2 virtual pulse along the z-axis.
        """
        ...

    def z180(
        self,
    ) -> VirtualZ:
        """
        Generate a π virtual pulse along the z-axis.

        Returns
        -------
        VirtualZ
            π virtual pulse along the z-axis.
        """
        ...

    def zx90(
        self,
        control_qubit: str | Sequence[str],
        target_qubit: str | None = None,
        /,
        *,
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
        control_qubit: str | Sequence[str],
        target_qubit: str | None = None,
        /,
        *,
        zx90: PulseSchedule | None = None,
        x90: Waveform | None = None,
    ) -> PulseSchedule: ...

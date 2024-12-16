from __future__ import annotations

import sys
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Collection, Final, Literal, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output, display
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from tqdm import tqdm

from ..analysis import (
    IQPlotter,
    RabiParam,
    fitting,
)
from ..analysis import visualization as vis
from ..backend import (
    SAMPLING_PERIOD,
    Box,
    ControlParams,
    ControlSystem,
    ExperimentSystem,
    MixingUtil,
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
    StateClassifierGMM,
    StateClassifierKMeans,
)
from ..measurement.measurement import (
    DEFAULT_CAPTURE_MARGIN,
    DEFAULT_CAPTURE_WINDOW,
    DEFAULT_CONFIG_DIR,
    DEFAULT_INTERVAL,
    DEFAULT_READOUT_DURATION,
    DEFAULT_SHOTS,
)
from ..pulse import (
    CPMG,
    Blank,
    Drag,
    FlatTop,
    Gaussian,
    PhaseShift,
    Pulse,
    PulseSchedule,
    PulseSequence,
    Rect,
    VirtualZ,
    Waveform,
)
from ..typing import IQArray, ParametricPulseSchedule, ParametricWaveformDict, TargetMap
from ..version import get_package_version
from . import experiment_tool
from .experiment_note import ExperimentNote
from .experiment_record import ExperimentRecord
from .experiment_result import (
    AmplCalibData,
    AmplRabiData,
    ExperimentResult,
    FreqRabiData,
    RabiData,
    RamseyData,
    RBData,
    SweepData,
    T1Data,
    T2Data,
    TimePhaseData,
)

console = Console()

USER_NOTE_PATH = ".user_note.json"
SYSTEM_NOTE_PATH = ".system_note.json"

RABI_PARAMS = "rabi_params"
STATE_CENTERS = "state_centers"

HPI_AMPLITUDE = "hpi_amplitude"
HPI_DURATION = 30
HPI_RISETIME = 10
PI_AMPLITUDE = "pi_amplitude"
PI_DURATION = 30
PI_RISETIME = 10
DRAG_HPI_AMPLITUDE = "drag_hpi_amplitude"
DRAG_HPI_BETA = "drag_hpi_beta"
DRAG_HPI_DURATION = 16
DRAG_PI_AMPLITUDE = "drag_pi_amplitude"
DRAG_PI_BETA = "drag_pi_beta"
DRAG_PI_DURATION = 16
DRAG_COEFF = 0.5


class Experiment:
    """
    Class representing an experiment.

    Parameters
    ----------
    chip_id : str
        Identifier of the quantum chip.
    qubits : Sequence[str]
        List of qubits to use in the experiment.
    config_dir : str, optional
        Directory of the configuration files. Defaults to DEFAULT_CONFIG_DIR.
    fetch_device_state : bool, optional
        Whether to fetch the device state. Defaults to True.
    control_window : int, optional
        Control window. Defaults to DEFAULT_CONTROL_WINDOW.
    capture_window : int, optional
        Capture window. Defaults to DEFAULT_CAPTURE_WINDOW.
    readout_duration : int, optional
        Readout duration. Defaults to DEFAULT_READOUT_DURATION.

    Examples
    --------
    >>> from qubex import Experiment
    >>> experiment = Experiment(
    ...     chip_id="64Q",
    ...     qubits=["Q00", "Q01"],
    ... )
    """

    def __init__(
        self,
        *,
        chip_id: str,
        muxes: Collection[str | int] | None = None,
        qubits: Collection[str | int] | None = None,
        exclude_qubits: Collection[str | int] | None = None,
        config_dir: str = DEFAULT_CONFIG_DIR,
        fetch_device_state: bool = True,
        connect_devices: bool = True,
        control_window: int | None = None,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        capture_margin: int = DEFAULT_CAPTURE_MARGIN,
        readout_duration: int = DEFAULT_READOUT_DURATION,
        use_neopulse: bool = False,
        classifier_type: Literal["kmeans", "gmm"] = "gmm",
    ):
        qubits = self._create_qubit_labels(
            chip_id=chip_id,
            muxes=muxes,
            qubits=qubits,
            exclude_qubits=exclude_qubits,
            config_dir=config_dir,
        )
        self._chip_id: Final = chip_id
        self._qubits: Final = qubits
        self._config_dir: Final = config_dir
        self._control_window: Final = control_window
        self._capture_window: Final = capture_window
        self._capture_margin: Final = capture_margin
        self._readout_duration: Final = readout_duration
        self._classifier_type: Final = classifier_type
        self._rabi_params: Final[dict[str, RabiParam]] = {}
        self._measurement = Measurement(
            chip_id=chip_id,
            qubits=qubits,
            config_dir=self._config_dir,
            fetch_device_state=fetch_device_state,
            use_neopulse=use_neopulse,
            connect_devices=connect_devices,
        )
        self._clifford_generator: CliffordGenerator | None = None
        self._user_note: Final = ExperimentNote(
            file_path=USER_NOTE_PATH,
        )
        self._system_note: Final = ExperimentNote(
            file_path=SYSTEM_NOTE_PATH,
        )
        self._validate()
        self.print_environment()

    def _create_qubit_labels(
        self,
        chip_id: str,
        muxes: Collection[str | int] | None,
        qubits: Collection[str | int] | None,
        exclude_qubits: Collection[str | int] | None,
        config_dir: str,
    ) -> list[str]:
        state_manager = StateManager.shared()
        state_manager.load(
            chip_id=chip_id,
            config_dir=config_dir,
        )
        quantum_system = state_manager.experiment_system.quantum_system
        qubit_labels = []
        if muxes is not None:
            for mux in muxes:
                labels = [
                    qubit.label for qubit in quantum_system.get_qubits_in_mux(mux)
                ]
                qubit_labels.extend(labels)
        if qubits is not None:
            for qubit in qubits:
                qubit_labels.append(quantum_system.get_qubit(qubit).label)
        if exclude_qubits is not None:
            for qubit in exclude_qubits:
                label = quantum_system.get_qubit(qubit).label
                if label in qubit_labels:
                    qubit_labels.remove(label)
        qubit_labels = sorted(list(set(qubit_labels)))
        return qubit_labels

    def _validate(self):
        """Check if the experiment is valid."""
        available_qubits = [
            target.qubit for target in self.experiment_system.ge_targets
        ]
        unavailable_qubits = [
            qubit for qubit in self._qubits if qubit not in available_qubits
        ]
        if len(unavailable_qubits) > 0:
            err_msg = f"Unavailable qubits: {unavailable_qubits}"
            print(err_msg)
            raise ValueError(err_msg)

    @property
    def tool(self):
        """Get the experiment tool."""
        return experiment_tool

    @property
    def util(self):
        """Get the experiment util."""
        return ExperimentUtil

    @property
    def state_manager(self) -> StateManager:
        """Get the state manager."""
        return StateManager.shared()

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Get the experiment system."""
        return self.state_manager.experiment_system

    @property
    def quantum_system(self) -> QuantumSystem:
        """Get the quantum system."""
        return self.experiment_system.quantum_system

    @property
    def control_system(self) -> ControlSystem:
        """Get the qube system."""
        return self.experiment_system.control_system

    @property
    def params(self) -> ControlParams:
        """Get the control parameters."""
        return self.experiment_system.control_params

    @property
    def chip_id(self) -> str:
        """Get the chip ID."""
        return self._chip_id

    @property
    def qubit_labels(self) -> list[str]:
        """Get the list of qubit labels."""
        return self._qubits

    @property
    def mux_labels(self) -> list[str]:
        """Get the list of mux labels."""
        mux_set = set()
        for qubit in self.qubit_labels:
            mux = self.experiment_system.get_mux_by_qubit(qubit)
            mux_set.add(mux.label)
        return sorted(list(mux_set))

    @property
    def qubits(self) -> dict[str, Qubit]:
        """Get the available qubit dict."""
        return {
            qubit.label: qubit
            for qubit in self.experiment_system.qubits
            if qubit.label in self.qubit_labels
        }

    @property
    def resonators(self) -> dict[str, Resonator]:
        """Get the available resonator dict."""
        return {
            resonator.qubit: resonator
            for resonator in self.experiment_system.resonators
            if resonator.qubit in self.qubit_labels
        }

    @property
    def targets(self) -> dict[str, Target]:
        """Get the target dict."""
        return {
            target.label: target
            for target in self.experiment_system.targets
            if target.qubit in self.qubit_labels
        }

    @property
    def available_targets(self) -> dict[str, Target]:
        """Get the available target dict."""
        return {
            target.label: target
            for target in self.experiment_system.targets
            if target.qubit in self.qubit_labels and target.is_available
        }

    @property
    def ge_targets(self) -> dict[str, Target]:
        """Get the available target dict."""
        return {
            label: target
            for label, target in self.available_targets.items()
            if target.is_ge
        }

    @property
    def ef_targets(self) -> dict[str, Target]:
        """Get the available target dict."""
        return {
            label: target
            for label, target in self.available_targets.items()
            if target.is_ef
        }

    @property
    def cr_targets(self) -> dict[str, Target]:
        """Get the available target dict."""
        return {
            label: target
            for label, target in self.available_targets.items()
            if target.is_cr
        }

    @property
    def boxes(self) -> dict[str, Box]:
        """Get the available box dict."""
        boxes = self.experiment_system.get_boxes_for_qubits(self.qubit_labels)
        return {box.id: box for box in boxes}

    @property
    def box_ids(self) -> list[str]:
        """Get the available box IDs."""
        return list(self.boxes.keys())

    @property
    def config_path(self) -> str:
        """Get the path of the configuration file."""
        return str(Path(self._config_dir).resolve())

    @property
    def note(self) -> ExperimentNote:
        """Get the user note."""
        return self._user_note

    @property
    def hpi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the default π/2 pulse.

        Returns
        -------
        TargetMap[Waveform]
            π/2 pulse.
        """
        # preset hpi amplitude
        amplitude = self.params.control_amplitude
        # calibrated hpi amplitude
        calib_amplitude: dict[str, float] = self._system_note.get(HPI_AMPLITUDE)
        if calib_amplitude is not None:
            for target in calib_amplitude:
                # use the calibrated hpi amplitude if it is stored
                amplitude[target] = calib_amplitude[target]
        return {
            target: FlatTop(
                duration=HPI_DURATION,
                amplitude=amplitude[target],
                tau=HPI_RISETIME,
            )
            for target in self._qubits
        }

    @property
    def pi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the default π pulse.

        Returns
        -------
        TargetMap[Waveform]
            π pulse.
        """
        # preset hpi pulse
        hpi = self.hpi_pulse
        # generate the pi pulse from the hpi pulse
        pi = {target: hpi[target].repeated(2) for target in self._qubits}
        # calibrated pi amplitude
        calib_amplitude: dict[str, float] = self._system_note.get(PI_AMPLITUDE)
        if calib_amplitude is not None:
            for target in calib_amplitude:
                # use the calibrated pi amplitude if it is stored
                pi[target] = FlatTop(
                    duration=PI_DURATION,
                    amplitude=calib_amplitude[target],
                    tau=PI_RISETIME,
                )
        return {target: pi[target] for target in self._qubits}

    @property
    def drag_hpi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the DRAG π/2 pulse.

        Returns
        -------
        TargetMap[Waveform]
            DRAG π/2 pulse.
        """
        calib_amplitude: dict[str, float] = self._system_note.get(DRAG_HPI_AMPLITUDE)
        calib_beta: dict[str, float] = self._system_note.get(DRAG_HPI_BETA)

        if calib_amplitude is None or calib_beta is None:
            raise ValueError("DRAG HPI amplitude or beta is not stored.")
        return {
            target: Drag(
                duration=DRAG_HPI_DURATION,
                amplitude=calib_amplitude[target],
                beta=calib_beta[target],
            )
            for target in self._qubits
        }

    @property
    def drag_pi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the DRAG π pulse.

        Returns
        -------
        TargetMap[Waveform]
            DRAG π pulse.
        """
        calib_amplitude: dict[str, float] = self._system_note.get(DRAG_PI_AMPLITUDE)
        calib_beta: dict[str, float] = self._system_note.get(DRAG_PI_BETA)
        if calib_amplitude is None or calib_beta is None:
            raise ValueError("DRAG PI amplitude or beta is not stored.")
        return {
            target: Drag(
                duration=DRAG_PI_DURATION,
                amplitude=calib_amplitude[target],
                beta=calib_beta[target],
            )
            for target in self._qubits
        }

    @property
    def ef_hpi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the ef π/2 pulse.

        Returns
        -------
        TargetMap[Waveform]
            π/2 pulse.
        """
        amplitude = self._system_note.get(HPI_AMPLITUDE)
        if amplitude is None:
            raise ValueError("EF π/2 amplitude is not stored.")
        ef_labels = [Target.ef_label(target) for target in self._qubits]
        return {
            target: FlatTop(
                duration=HPI_DURATION,
                amplitude=amplitude[target],
                tau=HPI_RISETIME,
            )
            for target in ef_labels
        }

    @property
    def ef_pi_pulse(self) -> TargetMap[Waveform]:
        """
        Get the ef π pulse.

        Returns
        -------
        TargetMap[Waveform]
            π/2 pulse.
        """
        amplitude = self._system_note.get(PI_AMPLITUDE)
        if amplitude is None:
            raise ValueError("EF π amplitude is not stored.")
        ef_labels = [Target.ef_label(target) for target in self._qubits]

        return {
            target: FlatTop(
                duration=PI_DURATION,
                amplitude=amplitude[target],
                tau=PI_RISETIME,
            )
            for target in ef_labels
        }

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        """Get the Rabi parameters."""
        params: dict[str, dict] | None
        params = self._system_note.get(RABI_PARAMS)
        if params is not None:
            rabi_params = {
                target: RabiParam(**param)
                for target, param in params.items()
                if target in self.qubit_labels
            }
            self._rabi_params.update(rabi_params)

        return self._rabi_params

    @property
    def ge_rabi_params(self) -> dict[str, RabiParam]:
        """Get the ge Rabi parameters."""
        return {
            target: param
            for target, param in self.rabi_params.items()
            if self.targets[target].is_ge
        }

    @property
    def ef_rabi_params(self) -> dict[str, RabiParam]:
        """Get the ef Rabi parameters."""
        return {
            Target.ge_label(target): param
            for target, param in self.rabi_params.items()
            if self.targets[target].is_ef
        }

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        """Get the classifiers."""
        return self._measurement.classifiers

    @property
    def state_centers(self) -> dict[str, dict[int, complex]]:
        """Get the state centers."""
        centers: dict[str, dict[str, list[float]]] | None
        centers = self._system_note.get(STATE_CENTERS)
        if centers is not None:
            return {
                target: {
                    int(state): complex(center[0], center[1])
                    for state, center in centers.items()
                }
                for target, centers in centers.items()
                if target in self.qubit_labels
            }

        return {
            target: classifier.centers
            for target, classifier in self.classifiers.items()
        }

    @property
    def clifford_generator(self) -> CliffordGenerator:
        """Get the Clifford generator."""
        if self._clifford_generator is None:
            self._clifford_generator = CliffordGenerator()
        return self._clifford_generator

    def _validate_rabi_params(
        self,
        targets: Collection[str] | None = None,
    ):
        """Check if the Rabi parameters are stored."""
        if len(self.rabi_params) == 0:
            raise ValueError("Rabi parameters are not stored.")
        if targets is not None:
            for target in targets:
                if target not in self.rabi_params:
                    raise ValueError(f"Rabi parameters for {target} are not stored.")
        if targets is not None:
            for target in targets:
                if target not in self.rabi_params:
                    raise ValueError(f"Rabi parameters for {target} are not stored.")

    def store_rabi_params(self, rabi_params: dict[str, RabiParam]):
        """
        Stores the Rabi parameters.

        Parameters
        ----------
        rabi_params : dict[str, RabiParam]
            Parameters of the Rabi oscillation.
        """
        if self._rabi_params.keys().isdisjoint(rabi_params.keys()):
            self._rabi_params.update(rabi_params)
        # else:
        #     if not Confirm.ask("Overwrite the existing Rabi parameters?"):
        #         return

        self._system_note.put(
            RABI_PARAMS,
            {label: asdict(rabi_param) for label, rabi_param in rabi_params.items()},
        )
        console.print("Rabi parameters are stored.")

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
        if state == "0":
            return Blank(0)
        elif state == "1":
            return self.pi_pulse[target]
        else:
            hpi = self.hpi_pulse[target]
            if state == "+":
                return hpi.shifted(np.pi / 2)
            elif state == "-":
                return hpi.shifted(-np.pi / 2)
            elif state == "+i":
                return hpi.shifted(np.pi)
            elif state == "-i":
                return hpi
            else:
                raise ValueError("Invalid state.")

    def get_spectators(self, qubit: str) -> list[Qubit]:
        """
        Get the spectators of the given qubit.

        Parameters
        ----------
        qubit : str
            Qubit to get the spectators.

        Returns
        -------
        list[Qubit]
            List of the spectators.
        """
        return self.quantum_system.get_spectator_qubits(qubit)

    def get_confusion_matrix(
        self,
        targets: Sequence[str],
    ) -> NDArray:
        """
        Get the confusion matrix of the given targets.

        Parameters
        ----------
        targets : Sequence[str]
            List of the targets.

        Returns
        -------
        NDArray
            Confusion matrix (rows: true, columns: predicted).
        """
        confusion_matrices = []
        for target in targets:
            cm = self.classifiers[target].confusion_matrix
            n_shots = cm[0].sum()
            confusion_matrices.append(cm / n_shots)
        return reduce(np.kron, confusion_matrices)

    def get_inverse_confusion_matrix(
        self,
        targets: Sequence[str],
    ) -> NDArray:
        """
        Get the inverse confusion matrix of the given targets.

        Parameters
        ----------
        targets : Sequence[str]
            List of the targets.

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
        confusion_matrix = self.get_confusion_matrix(targets)
        return np.linalg.inv(confusion_matrix)

    def print_environment(self, verbose: bool = False):
        """Print the environment information."""
        print("date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("python:", sys.version.split()[0])
        if verbose:
            print("numpy:", get_package_version("numpy"))
            print("quel_ic_config:", get_package_version("quel_ic_config"))
            print("quel_clock_master:", get_package_version("quel_clock_master"))
            print("qubecalib:", get_package_version("qubecalib"))
        print("qubex:", get_package_version("qubex"))
        print("env:", sys.prefix)
        print("config:", self.config_path)
        print("chip:", self.chip_id)
        print("qubits:", self.qubit_labels)
        print("muxes:", self.mux_labels)
        print("boxes:", self.box_ids)

    def print_boxes(self):
        """Print the box information."""
        table = Table(header_style="bold")
        table.add_column("ID", justify="left")
        table.add_column("NAME", justify="left")
        table.add_column("ADDRESS", justify="left")
        table.add_column("ADAPTER", justify="left")
        for box in self.boxes.values():
            table.add_row(box.id, box.name, box.address, box.adapter)
        console.print(table)

    def check_status(self):
        """Check the status of the measurement system."""
        # linnk status
        link_status = self._measurement.check_link_status(self.box_ids)
        if link_status["status"]:
            print("Link status: OK")
        else:
            print("Link status: NG")
        print(link_status["links"])

        # clock status
        clock_status = self._measurement.check_clock_status(self.box_ids)
        if clock_status["status"]:
            print("Clock status: OK")
        else:
            print("Clock status: NG")
        print(clock_status["clocks"])

        # config status
        config_status = self.state_manager.is_synced(box_ids=self.box_ids)
        if config_status:
            print("Config status: OK")
        else:
            print("Config status: NG")
        print(self.state_manager.device_settings)

    def linkup(
        self,
        box_ids: Optional[list[str]] = None,
        noise_threshold: int = 500,
    ) -> None:
        """
        Link up the measurement system.

        Parameters
        ----------
        box_ids : Optional[list[str]], optional
            List of the box IDs to link up. Defaults to None.

        Examples
        --------
        >>> ex.linkup()
        """
        if box_ids is None:
            box_ids = self.box_ids
        self._measurement.linkup(box_ids, noise_threshold=noise_threshold)

    def configure(
        self,
        box_ids: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ):
        """
        Configure the measurement system from the config files.

        Parameters
        ----------
        box_ids : Optional[list[str]], optional
            List of the box IDs to configure. Defaults to None.

        Examples
        --------
        >>> ex.configure()
        """
        self.state_manager.load(
            chip_id=self.chip_id,
            config_dir=self.config_path,
            targets_to_exclude=exclude,
        )
        self.state_manager.push(
            box_ids=box_ids or self.box_ids,
        )

    @contextmanager
    def modified_frequencies(self, frequencies: dict[str, float] | None):
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
        if frequencies is None:
            yield
        else:
            with self.state_manager.modified_frequencies(frequencies):
                yield

    def print_defaults(self):
        """Print the default params."""
        display(self._system_note)

    def save_defaults(self):
        """Save the default params."""
        self._system_note.save()

    def clear_defaults(self):
        """Clear the default params."""
        self._system_note.clear()

    def delete_defaults(self):
        """Delete the default params."""
        if Confirm.ask("Delete the default params?"):
            self._system_note.clear()
            self._system_note.save()

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
        record = ExperimentRecord.load(name)
        print(f"ExperimentRecord `{name}` is loaded.\n")
        print(f"description: {record.description}")
        print(f"created_at: {record.created_at}")
        return record

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
        ...     shots=3000,
        ...     interval=100 * 1024,
        ... )
        """
        return self._measurement.execute(
            schedule=schedule,
            mode=mode,
            shots=shots,
            interval=interval,
        )

    def measure(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
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
        ...     sequence={"Q00": np.zeros(0)},
        ...     mode="avg",
        ...     shots=3000,
        ...     interval=100 * 1024,
        ...     control_window=1024,
        ...     plot=True,
        ... )
        """
        control_window = control_window or self._control_window
        capture_window = capture_window or self._capture_window
        capture_margin = capture_margin or self._capture_margin
        readout_duration = readout_duration or self._readout_duration
        waveforms = {}

        if isinstance(sequence, PulseSchedule):
            sequence = sequence.get_sampled_sequences()

        for target, waveform in sequence.items():
            if isinstance(waveform, Waveform):
                waveforms[target] = waveform.values
            else:
                waveforms[target] = np.array(waveform, dtype=np.complex128)

        if frequencies is None:
            result = self._measurement.measure(
                waveforms=waveforms,
                mode=mode,
                shots=shots,
                interval=interval,
                control_window=control_window,
                capture_window=capture_window,
                capture_margin=capture_margin,
                readout_duration=readout_duration,
                readout_amplitudes=readout_amplitudes,
            )
        else:
            with self.modified_frequencies(frequencies):
                result = self._measurement.measure(
                    waveforms=waveforms,
                    mode=mode,
                    shots=shots,
                    interval=interval,
                    control_window=control_window,
                    capture_window=capture_window,
                    capture_margin=capture_margin,
                    readout_duration=readout_duration,
                    readout_amplitudes=readout_amplitudes,
                )
        if plot:
            result.plot()
        return result

    def _measure_batch(
        self,
        sequences: Sequence[TargetMap[IQArray]],
        *,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        readout_amplitudes: dict[str, float] | None = None,
    ):
        """
        Measures the signals using the given sequences.

        Parameters
        ----------
        sequences : Sequence[TargetMap[IQArray]]
            Sequences of the experiment.
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

        Yields
        ------
        MeasureResult
            Result of the experiment.
        """
        waveforms_list = [
            {
                target: np.array(waveform, dtype=np.complex128)
                for target, waveform in sequence.items()
            }
            for sequence in sequences
        ]
        return self._measurement.measure_batch(
            waveforms_list=waveforms_list,
            mode=mode,
            shots=shots,
            interval=interval,
            control_window=control_window or self._control_window,
            capture_window=capture_window or self._capture_window,
            capture_margin=capture_margin or self._capture_margin,
            readout_duration=readout_duration or self._readout_duration,
            readout_amplitudes=readout_amplitudes,
        )

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
        ...     shots=3000,
        ...     interval=100 * 1024,
        ...     control_window=1024,
        ...     plot=True,
        ... )
        """
        targets = []

        for target, state in states.items():
            targets.append(target)
            if state == "f":
                targets.append(Target.ef_label(target))

        with PulseSchedule(targets) as ps:
            for target, state in states.items():
                if state in ["0", "1", "+", "-", "+i", "-i"]:
                    ps.add(target, self.get_pulse_for_state(target, state))  # type: ignore
                elif state == "g":
                    ps.add(target, Blank(0))
                elif state == "e":
                    ps.add(target, self.pi_pulse[target])
                elif state == "f":
                    ps.add(target, self.pi_pulse[target])
                    ps.barrier()
                    ef_label = Target.ef_label(target)
                    ps.add(ef_label, self.ef_pi_pulse[ef_label])

        return self.measure(
            sequence=ps,
            mode=mode,
            shots=shots,
            interval=interval,
            control_window=control_window,
            capture_window=capture_window,
            capture_margin=capture_margin,
            readout_duration=readout_duration,
            plot=plot,
        )

    def check_noise(
        self,
        targets: list[str] | None = None,
        *,
        duration: int = 10240,
        plot: bool = True,
    ) -> MeasureResult:
        """
        Checks the noise level of the system.

        Parameters
        ----------
        targets : list[str], optional
            List of targets to check the noise. Defaults to None.
        duration : int, optional
            Duration of the noise measurement. Defaults to 2048.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.check_noise(["Q00", "Q01"])
        """
        if targets is None:
            targets = self.qubit_labels

        result = self._measurement.measure_noise(targets, duration)
        for target, data in result.data.items():
            if plot:
                vis.plot_waveform(
                    np.array(data.raw, dtype=np.complex64) * 2 ** (-32),
                    title=f"Readout noise of {target}",
                    xlabel="Capture time (μs)",
                    sampling_period=8e-3,
                )
        return result

    def check_waveform(
        self,
        targets: list[str] | None = None,
        *,
        plot: bool = True,
    ) -> MeasureResult:
        """
        Checks the readout waveforms of the given targets.

        Parameters
        ----------
        targets : list[str], optional
            List of targets to check the waveforms.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.check_waveform(["Q00", "Q01"])
        """
        if targets is None:
            targets = self.qubit_labels

        result = self.measure(sequence={target: np.zeros(0) for target in targets})
        if plot:
            result.plot()
        return result

    def check_rabi(
        self,
        targets: list[str] | None = None,
        *,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        store_params: bool = True,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]:
        """
        Checks the Rabi oscillation of the given targets.

        Parameters
        ----------
        targets : list[str], optional
            List of targets to check the Rabi oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        store_params : bool, optional
            Whether to store the Rabi parameters. Defaults to True.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[RabiData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.check_rabi(["Q00", "Q01"])
        """
        if targets is None:
            targets = self.qubit_labels
        if time_range is None:
            time_range = np.arange(0, 201, 8)
        ampl = self.params.control_amplitude
        amplitudes = {target: ampl[target] for target in targets}
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            shots=shots,
            interval=interval,
            store_params=store_params,
            plot=plot,
        )
        return result

    def obtain_rabi_params(
        self,
        targets: list[str],
        *,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]:
        """
        Conducts a Rabi experiment with the default amplitude.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Rabi oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[RabiData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.obtain_rabi_params(["Q00", "Q01"])
        """
        if time_range is None:
            time_range = np.arange(0, 201, 8)

        ampl = self.params.control_amplitude
        amplitudes = {target: ampl[target] for target in targets}
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            shots=shots,
            interval=interval,
            store_params=True,
            plot=plot,
        )
        return result

    def obtain_ef_rabi_params(
        self,
        targets: list[str],
        *,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]:
        """
        Conducts a Rabi experiment with the default amplitude.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Rabi oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[RabiData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.obtain_ef_rabi_params(["Q00", "Q01"])
        """
        if time_range is None:
            time_range = np.arange(0, 201, 8)

        ef_labels = [Target.ef_label(target) for target in targets]
        ef_targets = [self.targets[ef] for ef in ef_labels]

        ampl = self.params.control_amplitude
        amplitudes = {ef.label: ampl[ef.qubit] / np.sqrt(2) for ef in ef_targets}

        rabi_data = {}
        rabi_params = {}
        for label in ef_labels:
            data = self.ef_rabi_experiment(
                amplitudes={label: amplitudes[label]},
                time_range=time_range,
                shots=shots,
                interval=interval,
                store_params=True,
                plot=plot,
            ).data[label]
            rabi_data[label] = data
            rabi_params[label] = data.rabi_param

        result = ExperimentResult(
            data=rabi_data,
            rabi_params=rabi_params,
        )
        return result

    def rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike,
        detuning: float = 0.0,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]:
        """
        Conducts a Rabi experiment.

        Parameters
        ----------
        amplitudes : dict[str, float]
            Amplitudes of the control pulses.
        time_range : ArrayLike
            Time range of the experiment.
        detuning : float, optional
            Detuning of the control frequency. Defaults to 0.0.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        store_params : bool, optional
            Whether to store the Rabi parameters. Defaults to False.

        Returns
        -------
        ExperimentResult[RabiData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.rabi_experiment(
        ...     amplitudes={"Q00": 0.1},
        ...     time_range=range(0, 201, 4),
        ...     detuning=0.0,
        ...     shots=1024,
        ... )
        """
        # target labels
        targets = list(amplitudes.keys())

        # drive time range
        time_range = np.array(time_range, dtype=np.float64)

        # rabi sequence with rect pulses of duration T
        def rabi_sequence(T: int) -> PulseSchedule:
            with PulseSchedule(targets) as ps:
                for target in targets:
                    ps.add(target, Rect(duration=T, amplitude=amplitudes[target]))
            return ps

        # detune target frequencies if necessary
        detuned_frequencies = {
            target: self.targets[target].frequency + detuning for target in amplitudes
        }

        # run the Rabi experiment by sweeping the drive time
        sweep_result = self.sweep_parameter(
            sequence=rabi_sequence,
            sweep_range=time_range,
            frequencies=detuned_frequencies,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        # sweep data with the target labels
        sweep_data = sweep_result.data

        # fit the Rabi oscillation
        rabi_params = {
            target: fitting.fit_rabi(
                target=data.target,
                times=data.sweep_range,
                data=data.data,
                plot=plot,
            )
            for target, data in sweep_data.items()
        }

        # store the Rabi parameters if necessary
        if store_params:
            self.store_rabi_params(rabi_params)

        # create the Rabi data for each target
        rabi_data = {
            target: RabiData(
                target=target,
                data=data.data,
                time_range=time_range,
                rabi_param=rabi_params[target],
                state_centers=self.state_centers.get(target),
            )
            for target, data in sweep_data.items()
        }

        # create the experiment result
        result = ExperimentResult(
            data=rabi_data,
            rabi_params=rabi_params,
        )

        # return the result
        return result

    def ef_rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike,
        detuning: float = 0.0,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]:
        """
        Conducts a Rabi experiment.

        Parameters
        ----------
        amplitudes : dict[str, float]
            Amplitudes of the control pulses.
        time_range : ArrayLike
            Time range of the experiment.
        detuning : float, optional
            Detuning of the control frequency. Defaults to 0.0.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        store_params : bool, optional
            Whether to store the Rabi parameters. Defaults to False.

        Returns
        -------
        ExperimentResult[RabiData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.ef_rabi_experiment(
        ...     amplitudes={"Q00": 0.1},
        ...     time_range=range(0, 201, 4),
        ...     detuning=0.0,
        ...     shots=1024,
        ... )
        """
        amplitudes = {
            Target.ef_label(label): amplitude for label, amplitude in amplitudes.items()
        }
        ge_labels = [Target.ge_label(label) for label in amplitudes]
        ef_labels = [Target.ef_label(label) for label in amplitudes]
        ef_targets = [self.targets[ef] for ef in ef_labels]

        # drive time range
        time_range = np.array(time_range, dtype=np.float64)

        # ef rabi sequence with rect pulses of duration T
        def ef_rabi_sequence(T: int) -> PulseSchedule:
            with PulseSchedule(ge_labels + ef_labels) as ps:
                # prepare qubits to the excited state
                for ge in ge_labels:
                    ps.add(ge, self.pi_pulse[ge])
                ps.barrier()
                # apply the ef drive to induce the ef Rabi oscillation
                for ef in ef_labels:
                    ps.add(ef, Rect(duration=T, amplitude=amplitudes[ef]))
            return ps

        # detune ef frequencies if necessary
        detuned_frequencies = {ef.label: ef.frequency + detuning for ef in ef_targets}

        # run the Rabi experiment by sweeping the drive time
        sweep_result = self.sweep_parameter(
            sequence=ef_rabi_sequence,
            sweep_range=time_range,
            frequencies=detuned_frequencies,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        # sweep data with the ef labels
        sweep_data = {ef.label: sweep_result.data[ef.qubit] for ef in ef_targets}

        # fit the Rabi oscillation
        rabi_params = {
            target: fitting.fit_rabi(
                target=target,
                times=data.sweep_range,
                data=data.data,
                plot=plot,
            )
            for target, data in sweep_data.items()
        }

        # store the Rabi parameters if necessary
        if store_params:
            self.store_rabi_params(rabi_params)

        # create the Rabi data for each target
        rabi_data = {
            target: RabiData(
                target=target,
                data=data.data,
                time_range=time_range,
                rabi_param=rabi_params[target],
            )
            for target, data in sweep_data.items()
        }

        # create the experiment result
        result = ExperimentResult(
            data=rabi_data,
            rabi_params=rabi_params,
        )

        # return the result
        return result

    def sweep_parameter(
        self,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        *,
        sweep_range: ArrayLike,
        repetitions: int = 1,
        frequencies: Optional[dict[str, float]] = None,
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
        frequencies : Optional[dict[str, float]]
            Frequencies of the qubits.
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
        sweep_range = np.array(sweep_range)

        if rabi_level == "ge":
            rabi_params = self.ge_rabi_params
        elif rabi_level == "ef":
            rabi_params = self.ef_rabi_params
        else:
            raise ValueError("Invalid Rabi level.")

        if isinstance(sequence, dict):
            # TODO: this parameter type (dict[str, Callable[..., Waveform]]) will be deprecated
            targets = list(sequence.keys())
            sequences = [
                {
                    target: sequence[target](param).repeated(repetitions).values
                    for target in targets
                }
                for param in sweep_range
            ]

        if callable(sequence):
            if isinstance(sequence(0), PulseSchedule):
                sequences = [
                    sequence(param).repeated(repetitions).get_sampled_sequences()  # type: ignore
                    for param in sweep_range
                ]
            elif isinstance(sequence(0), dict):
                sequences = [
                    {
                        target: waveform.repeated(repetitions).values
                        for target, waveform in sequence(param).items()  # type: ignore
                    }
                    for param in sweep_range
                ]
        else:
            raise ValueError("Invalid sequence.")

        signals = defaultdict(list)
        plotter = IQPlotter(self.state_centers)

        generator = self._measure_batch(
            sequences=sequences,
            shots=shots,
            interval=interval,
            control_window=control_window or self._control_window,
            capture_window=capture_window or self._capture_window,
            capture_margin=capture_margin or self._capture_margin,
        )
        with self.modified_frequencies(frequencies):
            for result in generator:
                for target, data in result.data.items():
                    signals[target].append(data.kerneled)
                if plot:
                    plotter.update(signals)

        # with self.modified_frequencies(frequencies):
        #     for seq in sequences:
        #         measure_result = self.measure(
        #             sequence=seq,
        #             mode="avg",
        #             shots=shots,
        #             interval=interval,
        #             control_window=control_window,
        #             capture_window=capture_window,
        #             capture_margin=capture_margin,
        #         )
        #         for target, data in measure_result.data.items():
        #             signals[target].append(complex(data.kerneled))
        #         if plot:
        #             plotter.update(signals)

        if plot:
            plotter.show()

        sweep_data = {
            target: SweepData(
                target=target,
                data=np.array(values),
                sweep_range=sweep_range,
                rabi_param=rabi_params.get(target),
                state_centers=self.state_centers.get(target),
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                xaxis_type=xaxis_type,
                yaxis_type=yaxis_type,
            )
            for target, values in signals.items()
        }
        result = ExperimentResult(data=sweep_data, rabi_params=self.rabi_params)
        return result

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

        def repeated_sequence(N: int) -> PulseSchedule:
            if isinstance(sequence, dict):
                targets = list(sequence.keys())
                with PulseSchedule(targets) as ps:
                    for target, pulse in sequence.items():
                        ps.add(target, pulse.repeated(N))
            elif isinstance(sequence, PulseSchedule):
                ps = sequence.repeated(N)
            else:
                raise ValueError("Invalid sequence.")
            return ps

        result = self.sweep_parameter(
            sweep_range=np.arange(repetitions + 1),
            sequence=repeated_sequence,
            repetitions=1,
            shots=shots,
            interval=interval,
            plot=plot,
            xaxis_title="Number of repetitions",
        )

        if plot:
            result.plot(normalize=True)

        return result

    def obtain_freq_rabi_relation(
        self,
        targets: list[str],
        *,
        detuning_range: ArrayLike | None = None,
        time_range: ArrayLike | None = None,
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[FreqRabiData]:
        """
        Obtains the relation between the detuning and the Rabi frequency.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Rabi oscillation.
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
        if detuning_range is None:
            detuning_range = np.linspace(-0.01, 0.01, 15)
        else:
            detuning_range = np.array(detuning_range, dtype=np.float64)

        if time_range is None:
            time_range = np.arange(0, 101, 4)

        ampl = self.params.control_amplitude
        rabi_rates: dict[str, list[float]] = defaultdict(list)
        for detuning in detuning_range:
            if rabi_level == "ge":
                rabi_result = self.rabi_experiment(
                    time_range=time_range,
                    amplitudes={target: ampl[target] for target in targets},
                    detuning=detuning,
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
            elif rabi_level == "ef":
                rabi_result = self.ef_rabi_experiment(
                    time_range=time_range,
                    amplitudes={
                        target: ampl[target] / np.sqrt(2) for target in targets
                    },
                    detuning=detuning,
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
            else:
                raise ValueError("Invalid rabi_level.")
            clear_output()
            if plot:
                rabi_result.fit()
            clear_output(wait=True)
            rabi_params = rabi_result.rabi_params
            if rabi_params is None:
                raise ValueError("Rabi parameters are not stored.")
            for target, param in rabi_params.items():
                rabi_rate = param.frequency
                rabi_rates[target].append(rabi_rate)

        frequencies = {
            target: detuning_range + self.targets[target].frequency
            for target in rabi_rates
        }

        data = {
            target: FreqRabiData(
                target=target,
                data=np.array(values, dtype=np.float64),
                sweep_range=detuning_range,
                frequency_range=frequencies[target],
            )
            for target, values in rabi_rates.items()
        }

        result = ExperimentResult(data=data)
        return result

    def obtain_ampl_rabi_relation(
        self,
        targets: list[str],
        *,
        amplitude_range: ArrayLike | None = None,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[AmplRabiData]:
        """
        Obtains the relation between the control amplitude and the Rabi frequency.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Rabi oscillation.
        amplitude_range : ArrayLike, optional
            Range of the control amplitude to sweep.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
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
        if amplitude_range is None:
            amplitude_range = np.linspace(0.01, 0.1, 10)
        else:
            amplitude_range = np.array(amplitude_range, dtype=np.float64)

        if time_range is None:
            time_range = np.arange(0, 201, 4)

        rabi_rates: dict[str, list[float]] = defaultdict(list)
        for amplitude in amplitude_range:
            if amplitude <= 0:
                continue
            result = self.rabi_experiment(
                amplitudes={target: amplitude for target in targets},
                time_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
            )
            clear_output(wait=True)
            rabi_params = result.rabi_params
            if rabi_params is None:
                raise ValueError("Rabi parameters are not stored.")
            for target, param in rabi_params.items():
                rabi_rate = param.frequency
                rabi_rates[target].append(rabi_rate)
        data = {
            target: AmplRabiData(
                target=target,
                data=np.array(values, dtype=np.float64),
                sweep_range=amplitude_range,
            )
            for target, values in rabi_rates.items()
        }
        return ExperimentResult(data=data)

    def obtain_time_phase_relation(
        self,
        targets: list[str],
        *,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[TimePhaseData]:
        """
        Obtains the relation between the control window and the phase shift.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the phase shift.
        time_range : ArrayLike, optional
            The control window range to sweep in ns.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[PhaseShiftData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.obtain_time_phase_relation(
        ...     targets=["Q00", "Q01"],
        ...     time_range=range(0, 1024, 128),
        ... )
        """
        if time_range is None:
            time_range = np.arange(0, 1024, 128)
        else:
            time_range = np.array(time_range, dtype=np.int64)

        iq_data = defaultdict(list)
        plotter = IQPlotter()
        for window in time_range:
            result = self.measure(
                sequence={target: np.zeros(0) for target in targets},
                shots=shots,
                interval=interval,
                control_window=window,
                plot=False,
            )
            for qubit, value in result.data.items():
                iq = complex(value.kerneled)
                iq_data[qubit].append(iq)
            if plot:
                plotter.update(iq_data)
        data = {
            qubit: TimePhaseData(
                target=qubit,
                data=np.array(values),
                sweep_range=time_range,
            )
            for qubit, values in iq_data.items()
        }
        return ExperimentResult(data=data)

    def calc_control_amplitudes(
        self,
        *,
        rabi_rate: float = 12.5e-3,
        current_amplitudes: dict[str, float] | None = None,
        current_rabi_params: dict[str, RabiParam] | None = None,
        print_result: bool = True,
    ) -> dict[str, float]:
        """
        Calculates the control amplitudes for the Rabi rate.

        Parameters
        ----------
        rabi_rate : float, optional
            Target Rabi rate in GHz. Defaults to 12.5e-3.
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
        current_rabi_params = current_rabi_params or self.rabi_params

        if current_rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        if current_amplitudes is None:
            current_amplitudes = {}
            default_ampl = self.params.control_amplitude
            for target in current_rabi_params:
                if self.targets[target].is_ge:
                    current_amplitudes[target] = default_ampl[target]
                elif self.targets[target].is_ef:
                    qubit = Target.qubit_label(target)
                    current_amplitudes[target] = default_ampl[qubit] / np.sqrt(2)
                else:
                    raise ValueError("Invalid target.")

        amplitudes = {
            target: current_amplitudes[target]
            * rabi_rate
            / current_rabi_params[target].frequency
            for target in current_rabi_params
        }

        if print_result:
            print(f"control_amplitude for {rabi_rate * 1e3} MHz\n")
            for target, amplitude in amplitudes.items():
                print(f"{target}: {amplitude:.6f}")

            print(f"\n{1/rabi_rate/4} ns rect pulse → π/2 pulse")

        return amplitudes

    def calibrate_control_frequency(
        self,
        targets: list[str],
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 15),
        time_range: ArrayLike = range(0, 101, 8),
        amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]:
        # store the original control amplitudes
        original_control_amplitudes = deepcopy(self.params.control_amplitude)

        if amplitudes is not None:
            # modify the control amplitudes if necessary
            for target, amplitude in amplitudes.items():
                label = Target.qubit_label(target)
                self.params.control_amplitude[label] = amplitude

        # restore the original control amplitudes
        self.params.control_amplitude = original_control_amplitudes

        result = self.obtain_freq_rabi_relation(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            shots=shots,
            interval=interval,
            plot=plot,
        )
        fit_data = {target: data.fit()[0] for target, data in result.data.items()}

        print("\nResults\n-------")
        print("ge frequency (GHz):")
        for target, fit in fit_data.items():
            print(f"    {target}: {fit:.6f}")
        return fit_data

    def calibrate_ef_control_frequency(
        self,
        targets: list[str],
        *,
        detuning_range: ArrayLike | None = None,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]:
        if detuning_range is None:
            detuning_range = np.linspace(-0.01, 0.01, 15)

        if time_range is None:
            time_range = np.arange(0, 101, 4)

        result = self.obtain_freq_rabi_relation(
            targets=targets,
            detuning_range=detuning_range,
            rabi_level="ef",
            time_range=time_range,
            shots=shots,
            interval=interval,
            plot=plot,
        )
        fit_data = {target: data.fit()[0] for target, data in result.data.items()}

        print("\nResults\n-------")
        print("ef frequency (GHz):")
        for target, fit in fit_data.items():
            label = Target.ge_label(target)
            print(f"    {label}: {fit:.6f}")
        print("anharmonicity (GHz):")
        for target, fit in fit_data.items():
            label = Target.ge_label(target)
            ge_freq = self.targets[label].frequency
            print(f"    {label}: {fit - ge_freq:.6f}")
        return fit_data

    def calibrate_readout_frequency(
        self,
        targets: list[str],
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 15),
        time_range: ArrayLike = range(0, 101, 8),
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]:
        detuning_range = np.array(detuning_range, dtype=np.float64)

        # store the original readout amplitudes
        original_readout_amplitudes = deepcopy(self.params.readout_amplitude)

        result = defaultdict(list)
        for detuning in detuning_range:
            modified_frequencies = {
                resonator.label: resonator.frequency + detuning
                for resonator in self.resonators.values()
            }
            with self.modified_frequencies(modified_frequencies):
                if readout_amplitudes is not None:
                    # modify the readout amplitudes if necessary
                    for target, amplitude in readout_amplitudes.items():
                        label = Target.qubit_label(target)
                        self.params.readout_amplitude[label] = amplitude

                rabi_result = self.rabi_experiment(
                    time_range=time_range,
                    amplitudes={
                        target: self.params.control_amplitude[target]
                        for target in targets
                    },
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
                clear_output()
                if plot:
                    rabi_result.fit()
                clear_output(wait=True)
                for qubit, data in rabi_result.data.items():
                    rabi_amplitude = data.rabi_param.amplitude
                    result[qubit].append(rabi_amplitude)

        # restore the original readout amplitudes
        self.params.readout_amplitude = original_readout_amplitudes

        fit_data = {}
        for target, values in result.items():
            freq = self.resonators[target].frequency
            freq_fit = fitting.fit_lorentzian(
                target=target,
                freq_range=detuning_range + freq,
                data=np.array(values),
                plot=plot,
                title="Readout frequency calibration",
                xaxis_title="Readout frequency (GHz)",
            )
            fit_data[target] = freq_fit

        for target, freq in fit_data.items():
            print(f"{target}: {freq:.6f}")

        return fit_data

    def calibrate_default_pulse(
        self,
        targets: list[str],
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
        target : str
            Target qubit to calibrate.
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
        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        def calibrate(target: str) -> AmplCalibData:
            if pulse_type == "hpi":
                pulse = FlatTop(
                    duration=HPI_DURATION,
                    amplitude=1,
                    tau=HPI_RISETIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = FlatTop(
                    duration=PI_DURATION,
                    amplitude=1,
                    tau=PI_RISETIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")
            ampl = self.calc_control_amplitudes(
                rabi_rate=rabi_rate,
                print_result=False,
            )[target]
            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_range = np.linspace(ampl_min, ampl_max, 20)
            n_per_rotation = 2 if pulse_type == "pi" else 4
            sweep_data = self.sweep_parameter(
                sequence=lambda x: {target: pulse.scaled(x)},
                sweep_range=ampl_range,
                repetitions=n_per_rotation * n_rotations,
                shots=shots,
                interval=interval,
                plot=False,
            ).data[target]

            calib_value = fitting.fit_ampl_calib_data(
                target=target,
                amplitude_range=ampl_range,
                data=-sweep_data.normalized,
                plot=plot,
                title=f"{pulse_type} pulse calibration",
            )

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=calib_value,
            )

        data: dict[str, AmplCalibData] = {}
        for idx, target in enumerate(targets):
            print(f"[{idx+1}/{len(targets)}] Calibrating {target}...\n")
            data[target] = calibrate(target)
            print("")

        print(f"Calibration results for {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"{target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)

    def calibrate_ef_pulse(
        self,
        targets: list[str],
        pulse_type: Literal["pi", "hpi"],
        n_rotations: int = 1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the default pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
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
        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        ef_labels = [Target.ef_label(label) for label in targets]

        def calibrate(target: str) -> AmplCalibData:
            ge_label = Target.ge_label(target)
            ef_label = Target.ef_label(target)

            if pulse_type == "hpi":
                pulse = FlatTop(
                    duration=HPI_DURATION,
                    amplitude=1,
                    tau=HPI_RISETIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = FlatTop(
                    duration=PI_DURATION,
                    amplitude=1,
                    tau=PI_RISETIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")
            ampl = self.calc_control_amplitudes(
                rabi_rate=rabi_rate,
                print_result=False,
            )[ef_label]
            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_range = np.linspace(ampl_min, ampl_max, 20)
            n_per_rotation = 2 if pulse_type == "pi" else 4
            repetitions = n_per_rotation * n_rotations

            def sequence(x: float) -> PulseSchedule:
                with PulseSchedule([ge_label, ef_label]) as ps:
                    ps.add(ge_label, self.pi_pulse[ge_label])
                    ps.barrier()
                    ps.add(ef_label, pulse.scaled(x).repeated(repetitions))
                return ps

            sweep_data = self.sweep_parameter(
                sequence=sequence,
                sweep_range=ampl_range,
                repetitions=1,
                rabi_level="ef",
                shots=shots,
                interval=interval,
                plot=True,
            ).data[ge_label]

            calib_value = fitting.fit_ampl_calib_data(
                target=ef_label,
                amplitude_range=ampl_range,
                data=-sweep_data.normalized,
                title=f"ef {pulse_type} pulse calibration",
            )

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=calib_value,
            )

        data: dict[str, AmplCalibData] = {}
        for idx, target in enumerate(ef_labels):
            print(f"[{idx+1}/{len(targets)}] Calibrating {target}...\n")
            data[target] = calibrate(target)
            print("")

        return ExperimentResult(data=data)

        print(f"Calibration results for {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"{target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)

    def calibrate_drag_pulse(
        self,
        targets: list[str],
        pulse_type: Literal["pi", "hpi"],
        n_rotations: int = 4,
        drag_coeff: float = DRAG_COEFF,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the DRAG pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        n_rotations : int, optional
            Number of rotations. Defaults to 4.
        drag_coeff : float, optional
            DRAG coefficient. Defaults to DRAG_COEFF.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        def calibrate(target: str) -> AmplCalibData:
            if pulse_type == "hpi":
                pulse = Drag(
                    duration=DRAG_HPI_DURATION,
                    amplitude=1,
                    beta=-drag_coeff / self.qubits[target].alpha,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = Drag(
                    duration=DRAG_PI_DURATION,
                    amplitude=1,
                    beta=-drag_coeff / self.qubits[target].alpha,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")
            ampl = self.calc_control_amplitudes(
                rabi_rate=rabi_rate,
                print_result=False,
            )[target]

            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_range = np.linspace(ampl_min, ampl_max, 20)
            n_per_rotation = 2 if pulse_type == "pi" else 4
            sweep_data = self.sweep_parameter(
                sequence=lambda x: {target: pulse.scaled(x)},
                sweep_range=ampl_range,
                repetitions=n_per_rotation * n_rotations,
                shots=shots,
                interval=interval,
                plot=False,
            ).data[target]

            calib_value = fitting.fit_ampl_calib_data(
                target=target,
                amplitude_range=ampl_range,
                data=-sweep_data.normalized,
                title=f"DRAG {pulse_type} pulse calibration",
            )

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=calib_value,
            )

        data: dict[str, AmplCalibData] = {}
        for idx, target in enumerate(targets):
            print(f"[{idx+1}/{len(targets)}] Calibrating {target}...\n")
            data[target] = calibrate(target)
            print("")

        print(f"Calibration results for DRAG {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"{target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)

    def calibrate_hpi_pulse(
        self,
        targets: list[str],
        n_rotations: int = 1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π/2 pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
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
        result = self.calibrate_default_pulse(
            targets=targets,
            pulse_type="hpi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        self._system_note.put(HPI_AMPLITUDE, ampl)

        return result

    def calibrate_pi_pulse(
        self,
        targets: list[str],
        n_rotations: int = 1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the π pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
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
        result = self.calibrate_default_pulse(
            targets=targets,
            pulse_type="pi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        self._system_note.put(PI_AMPLITUDE, ampl)

        return result

    def calibrate_ef_hpi_pulse(
        self,
        targets: list[str],
        n_rotations: int = 1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the ef π/2 pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
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
        result = self.calibrate_ef_pulse(
            targets=targets,
            pulse_type="hpi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        self._system_note.put(HPI_AMPLITUDE, ampl)

        return result

    def calibrate_ef_pi_pulse(
        self,
        targets: list[str],
        n_rotations: int = 1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the ef π pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
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
        result = self.calibrate_ef_pulse(
            targets=targets,
            pulse_type="pi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        self._system_note.put(PI_AMPLITUDE, ampl)

        return result

    def calibrate_drag_hpi_pulse(
        self,
        targets: list[str],
        n_rotations: int = 4,
        shots: int = DEFAULT_SHOTS,
        drag_coeff: float = DRAG_COEFF,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the DRAG π/2 pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
        n_rotations : int, optional
            Number of rotations. Defaults to 4.
        drag_coeff : float, optional
            DRAG coefficient. Defaults to DRAG_COEFF.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        result = self.calibrate_drag_pulse(
            targets=targets,
            pulse_type="hpi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        beta = {target: -drag_coeff / self.qubits[target].alpha for target in targets}
        self._system_note.put(DRAG_HPI_AMPLITUDE, ampl)
        self._system_note.put(DRAG_HPI_BETA, beta)

        return result

    def calibrate_drag_pi_pulse(
        self,
        targets: list[str],
        n_rotations: int = 4,
        drag_coeff: float = DRAG_COEFF,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the DRAG π pulse.

        Parameters
        ----------
        target : str
            Target qubit to calibrate.
        n_rotations : int, optional
            Number of rotations. Defaults to 4.
        drag_coeff : float, optional
            DRAG coefficient. Defaults to DRAG_COEFF.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        result = self.calibrate_drag_pulse(
            targets=targets,
            pulse_type="pi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        beta = {target: -drag_coeff / self.qubits[target].alpha for target in targets}
        self._system_note.put(DRAG_PI_AMPLITUDE, ampl)
        self._system_note.put(DRAG_PI_BETA, beta)

        return result

    def t1_experiment(
        self,
        targets: Collection[str],
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
        targets : Collection[str]
            Collection of qubits to check the T1 decay.
        targets : Collection[str]
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
        ...     target="Q00",
        ...     time_range=2 ** np.arange(1, 19),
        ...     shots=1024,
        ... )
        """
        targets = list(targets)
        self._validate_rabi_params(targets)

        if time_range is None:
            time_range = self.util.discretize_time_range(
                np.logspace(
                    np.log10(100),
                    np.log10(100 * 1000),
                    51,
                )
            )
        time_range = np.asarray(time_range)

        data: dict[str, T1Data] = {}

        for subgroup in self.util.create_qubit_subgroups(targets):

            def t1_sequence(T: int) -> PulseSchedule:
                with PulseSchedule(subgroup) as ps:
                    for target in subgroup:
                        ps.add(target, self.pi_pulse[target])
                        ps.add(target, Blank(T))
                return ps

            sweep_result = self.sweep_parameter(
                sequence=t1_sequence,
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
                title="T1 decay",
                xaxis_title="Time (μs)",
                yaxis_title="Measured value",
                xaxis_type=xaxis_type,
            )

            for target, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_exp_decay(
                    target=target,
                    x=sweep_data.sweep_range,
                    y=0.5 * (1 - sweep_data.normalized),
                    plot=plot,
                    title="T1",
                    xaxis_title="Time (μs)",
                    yaxis_title="Normalized value",
                    xaxis_type=xaxis_type,
                    yaxis_type="linear",
                )
                t1 = fit_result["tau"]
                t1_data = T1Data.new(sweep_data, t1=t1)
                data[target] = t1_data

                if save_image:
                    fig = fit_result["fig"]
                    vis.save_image(
                        fig,
                        name=f"t1_{target}",
                    )

        return ExperimentResult(data=data)

    def t2_experiment(
        self,
        targets: Collection[str],
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
        targets : Collection[str]
            Collection of targets to check the T2 decay.
        targets : Collection[str]
            Collection of targets to check the T2 decay.
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
        ...     target="Q00",
        ...     time_range=2 ** np.arange(1, 19),
        ...     shots=1024,
        ... )
        """
        targets = list(targets)

        self._validate_rabi_params(targets)

        if time_range is None:
            time_range = self.util.discretize_time_range(
                np.logspace(
                    np.log10(300),
                    np.log10(100 * 1000),
                    51,
                ),
                2 * SAMPLING_PERIOD,
            )
        time_range = np.asarray(time_range)

        data: dict[str, T2Data] = {}

        for subgroup in self.util.create_qubit_subgroups(targets):

            def t2_sequence(T: int) -> PulseSchedule:
                with PulseSchedule(subgroup) as ps:
                    for target in subgroup:
                        hpi = self.hpi_pulse[target]
                        pi = pi_cpmg or hpi.repeated(2)
                        ps.add(target, hpi)
                        if T > 0:
                            ps.add(
                                target,
                                CPMG(
                                    tau=(T - pi.duration * n_cpmg) // (2 * n_cpmg),
                                    pi=pi,
                                    n=n_cpmg,
                                ),
                            )
                        ps.add(target, hpi.shifted(np.pi))
                return ps

            sweep_result = self.sweep_parameter(
                sequence=t2_sequence,
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
            )

            for target, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_exp_decay(
                    target=target,
                    x=sweep_data.sweep_range,
                    y=0.5 * (1 - sweep_data.normalized),
                    plot=plot,
                    title="T2 echo",
                    xaxis_title="Time (μs)",
                    yaxis_title="Normalized value",
                )
                t2 = fit_result["tau"]
                t2_data = T2Data.new(sweep_data, t2=t2)
                data[target] = t2_data

                if save_image:
                    fig = fit_result["fig"]
                    vis.save_image(
                        fig,
                        name=f"t2_echo_{target}",
                    )

        return ExperimentResult(data=data)

    def ramsey_experiment(
        self,
        targets: list[str],
        *,
        time_range: ArrayLike = np.arange(0, 50001, 100),
        detuning: float = 0.001,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RamseyData]:
        """
        Conducts a Ramsey experiment in series.

        Parameters
        ----------
        targets : list[str]
            List of targets to check the Ramsey oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns. Defaults to np.arange(0, 50001, 100).
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

        Returns
        -------
        ExperimentResult[RamseyData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.ramsey_experiment(
        ...     target="Q00",
        ...     time_range=range(0, 10_000, 100),
        ...     shots=1024,
        ... )
        """
        time_range = np.asarray(time_range)

        self._validate_rabi_params()

        target_groups = self.util.create_qubit_subgroups(targets)
        spectator_groups = reversed(target_groups)  # TODO: make it more general

        data: dict[str, RamseyData] = {}

        for targets, spectators in zip(target_groups, spectator_groups):
            if spectator_state != "0":
                target_list = targets + spectators
            else:
                target_list = targets

            def ramsey_sequence(T: int) -> PulseSchedule:
                with PulseSchedule(target_list) as ps:
                    # Excite spectator qubits if needed
                    if spectator_state != "0":
                        for spectator in spectators:
                            if spectator in self._qubits:
                                pulse = self.get_pulse_for_state(
                                    target=spectator,
                                    state=spectator_state,
                                )
                                ps.add(spectator, pulse)
                        ps.barrier()

                    # Ramsey sequence for the target qubit
                    for target in targets:
                        hpi = self.hpi_pulse[target]
                        ps.add(target, hpi)
                        ps.add(target, Blank(T))
                        ps.add(target, hpi.shifted(np.pi))
                return ps

            detuned_frequencies = {
                target: self.qubits[target].frequency + detuning for target in targets
            }

            sweep_result = self.sweep_parameter(
                sequence=ramsey_sequence,
                sweep_range=time_range,
                frequencies=detuned_frequencies,
                shots=shots,
                interval=interval,
                plot=plot,
            )

            for target, sweep_data in sweep_result.data.items():
                if target in targets:
                    fit_result = fitting.fit_ramsey(
                        target=target,
                        x=sweep_data.sweep_range,
                        y=sweep_data.normalized,
                        plot=plot,
                    )
                    ramsey_data = RamseyData.new(
                        sweep_data=sweep_data,
                        t2=fit_result["tau"],
                        ramsey_freq=fit_result["f"],
                    )
                    data[target] = ramsey_data

        return ExperimentResult(data=data)

    def obtain_effective_control_frequency(
        self,
        target: str,
        *,
        time_range: ArrayLike | None = None,
        detuning: float = 0.0005,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> float:
        """
        Obtains the effective control frequency of the qubit.

        Parameters
        ----------
        target : str
            Target qubit to check the Ramsey oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns.
        detuning : float, optional
            Detuning of the control frequency. Defaults to 0.0005 GHz.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        float
            Effective control frequency.

        Examples
        --------
        >>> result = ex.obtain_true_control_frequency(
        ...     target="Q00",
        ...     time_range=range(0, 10000, 100),
        ...     shots=1024,
        ... )
        """
        if time_range is None:
            time_range = np.arange(0, 10001, 200)

        ramsey_freq_0 = (
            self.ramsey_experiment(
                targets=[target],
                time_range=time_range,
                detuning=detuning,
                spectator_state="0",
                shots=shots,
                interval=interval,
                plot=plot,
            )
            .data[target]
            .ramsey_freq
        )

        ramsey_freq_1 = (
            self.ramsey_experiment(
                targets=[target],
                time_range=time_range,
                detuning=detuning,
                spectator_state="1",
                shots=shots,
                interval=interval,
                plot=plot,
            )
            .data[target]
            .ramsey_freq
        )

        bare_freq_0 = self.targets[target].frequency + detuning - ramsey_freq_0
        bare_freq_1 = self.targets[target].frequency + detuning - ramsey_freq_1
        effective_freq = (bare_freq_0 + bare_freq_1) / 2

        print(f"Original frequency: {self.targets[target].frequency:.6f}")
        print(f"Bare frequency with spectator state 0: {bare_freq_0:.6f}")
        print(f"Bare frequency with spectator state 1: {bare_freq_1:.6f}")
        print(f"Effective control frequency: {effective_freq:.6f}")

        return effective_freq

    def measure_state_distribution(
        self,
        targets: list[str],
        *,
        n_states: Literal[2, 3] = 2,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> list[MeasureResult]:
        states = ["g", "e", "f"][:n_states]
        result = {
            state: self.measure_state(
                {target: state for target in targets},  # type: ignore
                shots=shots,
                interval=interval,
            )
            for state in states
        }
        for target in targets:
            data = {
                f"|{state}⟩": result[state].data[target].kerneled for state in states
            }
            if plot:
                vis.plot_state_distribution(
                    data=data,
                    title=f"State distribution of {target}",
                )
        return list(result.values())

    def build_classifier(
        self,
        targets: list[str] | None = None,
        *,
        n_states: Literal[2, 3] = 2,
        shots: int = 1000,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ):
        if targets is None:
            targets = self.qubit_labels

        results = self.measure_state_distribution(
            targets=targets,
            n_states=n_states,
            shots=shots,
            interval=interval,
            plot=False,
        )

        data = {
            target: {
                state: result.data[target].kerneled
                for state, result in enumerate(results)
            }
            for target in targets
        }

        classifiers: TargetMap[StateClassifier]
        if self._classifier_type == "kmeans":
            classifiers = {
                target: StateClassifierKMeans.fit(data[target]) for target in targets
            }
        elif self._classifier_type == "gmm":
            classifiers = {
                target: StateClassifierGMM.fit(data[target]) for target in targets
            }
        else:
            raise ValueError("Invalid classifier type.")
        self._measurement.classifiers = classifiers

        for target in targets:
            clf = classifiers[target]
            classified = []
            for state in range(n_states):
                if plot:
                    print(f"{target} prepared as |{state}⟩:")
                result = clf.classify(
                    target,
                    data[target][state],
                    plot=plot,
                )
                classified.append(result)
            fidelity = [
                classified[state][state] / sum(classified[state].values())
                for state in range(n_states)
            ]
            if plot:
                print(f"{target}:")
                print(f"  Total shots: {shots}")
                for state in range(n_states):
                    print(
                        f"  |{state}⟩ → {classified[state]}, f_{state}: {fidelity[state] * 100:.2f}%"
                    )
                print(
                    f"  Average readout fidelity : {np.mean(fidelity) * 100:.2f}%\n\n"
                )

        self._system_note.put(
            STATE_CENTERS,
            {
                target: {
                    str(state): (center.real, center.imag)
                    for state, center in classifiers[target].centers.items()
                }
                for target in targets
            },
        )

    def rb_sequence(
        self,
        *,
        target: str,
        n: int,
        x90: dict[str, Waveform] | None = None,
        zx90: PulseSchedule | dict[str, Waveform] | None = None,
        interleaved_waveform: PulseSchedule
        | dict[str, PulseSequence]
        | dict[str, Waveform]
        | None = None,
        interleaved_clifford: Clifford | None = None,
        seed: int | None = None,
    ) -> PulseSchedule:
        target_object = self.experiment_system.get_target(target)
        if target_object.is_cr:
            sched = self.rb_sequence_2q(
                target=target,
                n=n,
                x90=x90,
                zx90=zx90,
                interleaved_waveform=interleaved_waveform,
                interleaved_clifford=interleaved_clifford,
                seed=seed,
            )
            return sched
        else:
            if isinstance(interleaved_waveform, PulseSchedule):
                interleaved_waveform = interleaved_waveform.get_sequences()
            seq = self.rb_sequence_1q(
                target=target,
                n=n,
                x90=x90,
                interleaved_waveform=interleaved_waveform,
                interleaved_clifford=interleaved_clifford,
                seed=seed,
            )
            with PulseSchedule([target]) as ps:
                ps.add(target, seq)
            return ps

    def rb_sequence_1q(
        self,
        *,
        target: str,
        n: int,
        x90: Waveform | dict[str, Waveform] | None = None,
        interleaved_waveform: Waveform
        | dict[str, PulseSequence]
        | dict[str, Waveform]
        | None = None,
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
        if isinstance(x90, dict):
            x90 = x90.get(target)
        x90 = x90 or self.hpi_pulse[target]
        z90 = VirtualZ(np.pi / 2)

        sequence: list[Waveform | VirtualZ] = []

        if interleaved_waveform is None:
            cliffords, inverse = self.clifford_generator.create_rb_sequences(
                n=n,
                type="1Q",
                seed=seed,
            )
        else:
            if interleaved_clifford is None:
                raise ValueError("`interleaved_clifford` must be provided.")
            cliffords, inverse = self.clifford_generator.create_irb_sequences(
                n=n,
                interleave=interleaved_clifford,
                type="1Q",
                seed=seed,
            )

        def add_gate(gate: str):
            if gate == "X90":
                sequence.append(x90)
            elif gate == "Z90":
                sequence.append(z90)
            else:
                raise ValueError("Invalid gate.")

        for clifford in cliffords:
            for gate in clifford:
                add_gate(gate)
            if isinstance(interleaved_waveform, dict):
                interleaved_waveform = interleaved_waveform.get(target)
            if interleaved_waveform is not None:
                sequence.append(interleaved_waveform)

        for gate in inverse:
            add_gate(gate)

        return PulseSequence(sequence)

    def rb_sequence_2q(
        self,
        *,
        target: str,
        n: int,
        x90: dict[str, Waveform] | None = None,
        zx90: PulseSchedule
        | dict[str, PulseSequence]
        | dict[str, Waveform]
        | None = None,
        interleaved_waveform: PulseSchedule
        | dict[str, PulseSequence]
        | dict[str, Waveform]
        | None = None,
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
        x90 : Waveform | dict[str, Waveform], optional
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
        target_object = self.experiment_system.get_target(target)
        if not target_object.is_cr:
            raise ValueError(f"`{target}` is not a 2Q target.")
        control_qubit, target_qubit = Target.cr_qubit_pair(target)
        xi90, ix90 = None, None
        if isinstance(x90, dict):
            xi90 = x90.get(control_qubit)
            ix90 = x90.get(target_qubit)
        xi90 = xi90 or self.hpi_pulse[control_qubit]
        ix90 = ix90 or self.hpi_pulse[target_qubit]
        z90 = VirtualZ(np.pi / 2)

        if interleaved_waveform is None:
            cliffords, inverse = self.clifford_generator.create_rb_sequences(
                n=n,
                type="2Q",
                seed=seed,
            )
        else:
            if interleaved_clifford is None:
                raise ValueError("Interleave map must be provided.")
            cliffords, inverse = self.clifford_generator.create_irb_sequences(
                n=n,
                interleave=interleaved_clifford,
                type="2Q",
                seed=seed,
            )

        with PulseSchedule([control_qubit, target_qubit]) as ps:

            def add_gate(gate: str):
                if gate == "XI90":
                    ps.add(control_qubit, xi90)
                elif gate == "IX90":
                    ps.add(target_qubit, ix90)
                elif gate == "ZI90":
                    ps.add(control_qubit, z90)
                elif gate == "IZ90":
                    ps.add(target_qubit, z90)
                elif gate == "ZX90":
                    if zx90 is not None:
                        ps.barrier()
                        if isinstance(zx90, dict):
                            ps.add(control_qubit, zx90[control_qubit])
                            ps.add(target_qubit, zx90[target_qubit])
                        elif isinstance(zx90, PulseSchedule):
                            ps.call(zx90)
                        ps.barrier()
                else:
                    raise ValueError("Invalid gate.")

            for clifford in cliffords:
                for gate in clifford:
                    add_gate(gate)
                if interleaved_waveform is not None:
                    ps.barrier()
                    if isinstance(interleaved_waveform, dict):
                        ps.add(control_qubit, interleaved_waveform[control_qubit])
                        ps.add(target_qubit, interleaved_waveform[target_qubit])
                    elif isinstance(interleaved_waveform, PulseSchedule):
                        ps.call(interleaved_waveform)
                    ps.barrier()

            for gate in inverse:
                add_gate(gate)
        return ps

    def rb_experiment(
        self,
        *,
        target: str,
        n_cliffords_range: ArrayLike | None = None,
        x90: Waveform | dict[str, Waveform] | None = None,
        interleaved_waveform: Waveform | None = None,
        interleaved_clifford: dict[str, tuple[complex, str]] | None = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        seed: int | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
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
        interleaved_clifford : dict[str, tuple[complex, str]], optional
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
        if n_cliffords_range is None:
            n_cliffords_range = np.arange(0, 1001, 50)

        target_object = self.experiment_system.get_target(target)
        if target_object.is_cr:
            raise ValueError(f"`{target}` is not a 1Q target.")

        def rb_sequence(N: int) -> PulseSchedule:
            with PulseSchedule([target]) as ps:
                # Excite spectator qubits if needed
                if spectator_state != "0":
                    spectators = self.get_spectators(target)
                    for spectator in spectators:
                        if spectator.label in self._qubits:
                            pulse = self.get_pulse_for_state(
                                target=spectator.label,
                                state=spectator_state,
                            )
                            ps.add(spectator.label, pulse)
                    ps.barrier()

                # Randomized benchmarking sequence
                rb_sequence = self.rb_sequence_1q(
                    target=target,
                    n=N,
                    x90=x90,
                    interleaved_waveform=interleaved_waveform,
                    interleaved_clifford=interleaved_clifford,
                    seed=seed,
                )
                ps.add(
                    target,
                    rb_sequence,
                )
            return ps

        sweep_result = self.sweep_parameter(
            rb_sequence,
            sweep_range=n_cliffords_range,
            shots=shots,
            interval=interval,
            plot=False,
        )

        sweep_data = sweep_result.data[target]

        fit_data = fitting.fit_rb(
            target=target,
            x=sweep_data.sweep_range,
            y=sweep_data.normalized,
            title="Randomized benchmarking",
            xaxis_title="Number of Cliffords",
            yaxis_title="Z expectation value",
            xaxis_type="linear",
            yaxis_type="linear",
            plot=plot,
        )

        data = {
            qubit: RBData.new(
                data,
                depolarizing_rate=fit_data[0],
                avg_gate_error=fit_data[1],
                avg_gate_fidelity=fit_data[2],
            )
            for qubit, data in sweep_result.data.items()
        }

        return ExperimentResult(data=data)

    def rb_experiment_2q(
        self,
        *,
        target: str,
        n_cliffords_range: ArrayLike | None = None,
        x90: dict[str, Waveform] | None = None,
        zx90: PulseSchedule
        | dict[str, PulseSequence]
        | dict[str, Waveform]
        | None = None,
        interleaved_waveform: PulseSchedule
        | dict[str, PulseSequence]
        | dict[str, Waveform]
        | None = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        seed: int | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ):
        if n_cliffords_range is None:
            n_cliffords_range = np.arange(0, 1001, 50)
        else:
            n_cliffords_range = np.array(n_cliffords_range, dtype=int)

        target_object = self.experiment_system.get_target(target)
        if not target_object.is_cr:
            raise ValueError(f"`{target}` is not a 2Q target.")
        control_qubit, target_qubit = Target.cr_qubit_pair(target)

        def rb_sequence(N: int) -> PulseSchedule:
            with PulseSchedule([control_qubit, target_qubit]) as ps:
                # Excite spectator qubits if needed
                if spectator_state != "0":
                    control_spectators = {
                        qubit.label for qubit in self.get_spectators(control_qubit)
                    }
                    target_spectators = {
                        qubit.label for qubit in self.get_spectators(target_qubit)
                    }
                    spectators = (control_spectators | target_spectators) - {
                        control_qubit,
                        target_qubit,
                    }
                    for spectator in spectators:
                        if spectator in self._qubits:
                            pulse = self.get_pulse_for_state(
                                target=spectator,
                                state=spectator_state,
                            )
                            ps.add(spectator, pulse)
                    ps.barrier()

                # Randomized benchmarking sequence
                rb_sequence = self.rb_sequence_2q(
                    target=target,
                    n=N,
                    x90=x90,
                    zx90=zx90,
                    interleaved_waveform=interleaved_waveform,
                    interleaved_clifford=interleaved_clifford,
                    seed=seed,
                )
                ps.call(rb_sequence)
            return ps

        fidelities = []

        for n_clifford in n_cliffords_range:
            result = self.measure(
                sequence=rb_sequence(n_clifford),
                shots=shots,
                interval=interval,
                plot=False,
            )
            p00 = result.probabilities["00"]
            fidelities.append(p00)

        fit_data = fitting.fit_rb(
            target=target,
            x=n_cliffords_range,
            y=np.array(fidelities),
            title="Randomized benchmarking",
            xaxis_title="Number of Cliffords",
            yaxis_title="Probability of |00⟩",
            xaxis_type="linear",
            yaxis_type="linear",
            plot=plot,
        )

        return {
            "depolarizing_rate": fit_data[0],
            "avg_gate_error": fit_data[1],
            "avg_gate_fidelity": fit_data[2],
        }

    def randomized_benchmarking(
        self,
        target: str,
        *,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int = 30,
        x90: Waveform | dict[str, Waveform] | None = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        """
        Conducts a randomized benchmarking experiment with multiple trials.

        Parameters
        ----------
        target : str
            Target qubit.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords. Defaults to range(0, 1001, 100).
        n_trials : int, optional
            Number of trials for different random seeds. Defaults to 30.
        x90 : Waveform | dict[str, Waveform], optional
            π/2 pulse used for the experiment. Defaults to None.
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

        Returns
        -------
        dict
            Results of the experiment.
        """
        if n_cliffords_range is None:
            n_cliffords_range = np.arange(0, 1001, 100)
        else:
            n_cliffords_range = np.array(n_cliffords_range, dtype=int)

        self._validate_rabi_params()

        results = []
        seeds = np.random.randint(0, 2**32, n_trials)
        for seed in seeds:
            result = self.rb_experiment(
                target=target,
                n_cliffords_range=n_cliffords_range,
                spectator_state=spectator_state,
                x90=x90,
                seed=seed,
                shots=shots,
                interval=interval,
                plot=False,
            )
            results.append(result.data[target].normalized)
            clear_output(wait=True)

        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)

        fit_result = fitting.fit_rb(
            target=target,
            x=n_cliffords_range,
            y=mean,
            error_y=std,
            plot=plot,
            title="Randomized benchmarking",
            xaxis_title="Number of Cliffords",
            yaxis_title="Z expectation value",
            xaxis_type="linear",
            yaxis_type="linear",
        )

        return {
            "depolarizing_rate": fit_result[0],
            "avg_gate_error": fit_result[1],
            "avg_gate_fidelity": fit_result[2],
            "n_cliffords": n_cliffords_range,
            "mean": mean,
            "std": std,
        }

    def interleaved_randomized_benchmarking(
        self,
        *,
        target: str,
        interleaved_waveform: Waveform,
        interleaved_clifford: dict[str, tuple[complex, str]],
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int = 30,
        x90: Waveform | None = None,
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        show_ref: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        """
        Conducts a randomized benchmarking experiment with multiple trials.

        Parameters
        ----------
        target : str
            Target qubit.
        interleaved_waveform : Waveform
            Waveform of the interleaved gate.
        interleaved_clifford : dict[str, tuple[complex, str]]
            Clifford map of the interleaved gate.
        n_cliffords_range : ArrayLike, optional
            Range of the number of Cliffords. Defaults to range(0, 1001, 100).
        n_trials : int, optional
            Number of trials for different random seeds. Defaults to 30.
        x90 : Waveform, optional
            π/2 pulse. Defaults to None.
        spectator_state : Literal["0", "1", "+", "-", "+i", "-i"], optional
            Spectator state. Defaults to "0".
        show_ref : bool, optional
            Whether to show the reference curve. Defaults to False.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

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
        if n_cliffords_range is None:
            n_cliffords_range = np.arange(0, 1001, 100)
        else:
            n_cliffords_range = np.array(n_cliffords_range, dtype=int)

        rb_results = []
        irb_results = []
        seeds = np.random.randint(0, 2**32, n_trials)
        for seed in seeds:
            if show_ref:
                rb_result = self.rb_experiment(
                    target=target,
                    n_cliffords_range=n_cliffords_range,
                    x90=x90,
                    spectator_state=spectator_state,
                    seed=seed,
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
                rb_results.append(rb_result.data[target].normalized)
            irb_result = self.rb_experiment(
                target=target,
                n_cliffords_range=n_cliffords_range,
                x90=x90,
                interleaved_waveform=interleaved_waveform,
                interleaved_clifford=interleaved_clifford,
                spectator_state=spectator_state,
                seed=seed,
                shots=shots,
                interval=interval,
                plot=plot,
            )
            irb_results.append(irb_result.data[target].normalized)
            clear_output(wait=True)

        if show_ref:
            rb_mean = np.mean(rb_results, axis=0)
            rb_std = np.std(rb_results, axis=0)
            rb_fit_result = fitting.fit_rb(
                target=target,
                x=n_cliffords_range,
                y=rb_mean,
                error_y=rb_std,
                plot=False,
                title="Randomized benchmarking",
            )
            p_rb = rb_fit_result[0]
        irb_mean = np.mean(irb_results, axis=0)
        irb_std = np.std(irb_results, axis=0)
        irb_fit_result = fitting.fit_rb(
            target=target,
            x=n_cliffords_range,
            y=irb_mean,
            error_y=irb_std,
            plot=plot,
            title="Interleaved randomized benchmarking",
        )
        p_irb = irb_fit_result[0]

        if show_ref:
            fitting.plot_irb(
                target=target,
                x=n_cliffords_range,
                y_rb=rb_mean,
                y_irb=irb_mean,
                error_y_rb=rb_std,
                error_y_irb=irb_std,
                p_rb=p_rb,
                p_irb=p_irb,
                title="Interleaved randomized benchmarking",
                xaxis_title="Number of Cliffords",
                yaxis_title="Z expectation value",
            )

        return {
            "depolarizing_rate": irb_fit_result[0],
            "avg_gate_error": irb_fit_result[1],
            "avg_gate_fidelity": irb_fit_result[2],
            "n_cliffords": n_cliffords_range,
            "mean": irb_mean,
            "std": irb_std,
        }

    def state_tomography(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict[str, tuple[float, float, float]]:
        """
        Conducts a state tomography experiment.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Sequence to measure for each target.
        x90 : TargetMap[Waveform], optional
            π/2 pulse. Defaults to None.
        initial_state : TargetMap[str], optional
            Initial state of each target. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Results of the experiment.
        """
        if isinstance(sequence, PulseSchedule):
            sequence = sequence.get_sequences()
        else:
            sequence = {
                target: Pulse(waveform)
                if not isinstance(waveform, Waveform)
                else waveform
                for target, waveform in sequence.items()
            }

        x90 = x90 or self.hpi_pulse

        buffer: dict[str, list[float]] = defaultdict(list)

        qubits = set(Target.qubit_label(target) for target in sequence)
        targets = list(qubits | sequence.keys())

        for basis in ["X", "Y", "Z"]:
            with PulseSchedule(targets) as ps:
                if initial_state is not None:
                    for qubit in qubits:
                        if qubit in initial_state:
                            init_pulse = self.get_pulse_for_state(
                                target=qubit,
                                state=initial_state[qubit],
                            )
                            ps.add(qubit, init_pulse)
                            ps.barrier()
                for target, waveform in sequence.items():
                    ps.add(target, waveform)
                    ps.barrier()
                for qubit in qubits:
                    x90p = x90[qubit]
                    y90m = x90p.shifted(-np.pi / 2)
                    if basis == "X":
                        ps.add(qubit, y90m)
                    elif basis == "Y":
                        ps.add(qubit, x90p)

            measure_result = self.measure(
                ps,
                shots=shots,
                interval=interval,
                plot=plot,
            )
            for qubit, data in measure_result.data.items():
                rabi_param = self.rabi_params[qubit]
                if rabi_param is None:
                    raise ValueError("Rabi parameters are not stored.")
                values = data.kerneled
                values_rotated = values * np.exp(-1j * rabi_param.angle)
                values_normalized = (
                    np.imag(values_rotated) - rabi_param.offset
                ) / rabi_param.amplitude
                buffer[qubit] += [values_normalized]

        result = {
            qubit: (
                values[0],  # X
                values[1],  # Y
                values[2],  # Z
            )
            for qubit, values in buffer.items()
        }
        return result

    def state_evolution_tomography(
        self,
        *,
        sequences: (
            Sequence[TargetMap[IQArray]]
            | Sequence[TargetMap[Waveform]]
            | Sequence[PulseSchedule]
        ),
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        """
        Conducts a state evolution tomography experiment.

        Parameters
        ----------
        sequences : Sequence[TargetMap[IQArray]] | Sequence[TargetMap[Waveform]] | Sequence[PulseSchedule]
            Sequences to measure for each target.
        x90 : TargetMap[Waveform], optional
            π/2 pulse. Defaults to None.
        initial_state : TargetMap[str], optional
            Initial state of each target. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Results of the experiment.
        """
        buffer: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for sequence in tqdm(sequences):
            state_vectors = self.state_tomography(
                sequence=sequence,
                x90=x90,
                initial_state=initial_state,
                shots=shots,
                interval=interval,
                plot=False,
            )
            for target, state_vector in state_vectors.items():
                buffer[target].append(state_vector)

        result = {target: np.array(states) for target, states in buffer.items()}

        if plot:
            for target, states in result.items():
                print(f"State evolution of {target}")
                vis.display_bloch_sphere(states)

        return result

    def pulse_tomography(
        self,
        waveforms: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        n_samples: int = 100,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> TargetMap[NDArray[np.float64]]:
        """
        Conducts a pulse tomography experiment.

        Parameters
        ----------
        waveforms : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Waveforms to measure for each target.
        x90 : TargetMap[Waveform], optional
            π/2 pulse. Defaults to None.
        initial_state : TargetMap[str], optional
            Initial state of each target. Defaults to None.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        """
        self._validate_rabi_params()

        if isinstance(waveforms, PulseSchedule):
            waveforms = waveforms.get_sequences()

        pulses: dict[str, Waveform] = {}
        pulse_length_set = set()
        for target, waveform in waveforms.items():
            if isinstance(waveform, Waveform):
                pulse = waveform
            elif isinstance(waveform, list) or isinstance(waveform, np.ndarray):
                pulse = Pulse(waveform)
            else:
                raise ValueError("Invalid waveform.")
            pulses[target] = pulse
            pulse_length_set.add(pulse.length)
        if len(pulse_length_set) != 1:
            raise ValueError("The lengths of the waveforms must be the same.")

        pulse_length = pulse_length_set.pop()

        if plot:
            for target in pulses:
                pulses[target].plot(title=f"Waveform of {target}")

        def partial_waveform(waveform: Waveform, index: int) -> Waveform:
            """Returns a partial waveform up to the given index."""

            # If the waveform is a PulseSequence, we need to handle the PhaseShift gate.
            if isinstance(waveform, PulseSequence):
                current_index = 0
                pulse_sequence = PulseSequence([])
                for pulse in waveform._sequence:
                    # If the pulse is a PhaseShift gate, we can simply add it to the sequence.
                    if isinstance(pulse, PhaseShift):
                        pulse_sequence = pulse_sequence.added(pulse)
                        continue
                    # If the pulse is a Pulse and the length is greater than the index, we need to create a partial pulse.
                    elif current_index + pulse.length > index:
                        pulse = Pulse(pulse.values[0 : index - current_index])
                        pulse_sequence = pulse_sequence.added(pulse)
                        break
                    # If the pulse is a Pulse and the length is less than the index, we can add the pulse to the sequence.
                    else:
                        pulse_sequence = pulse_sequence.added(pulse)
                        current_index += pulse.length
                return pulse_sequence
            # If the waveform is a Pulse, we can simply return the partial waveform.
            else:
                return Pulse(waveform.values[0:index])

        if n_samples < pulse_length:
            indices = np.linspace(0, pulse_length - 1, n_samples).astype(int)
        else:
            indices = np.arange(pulse_length)

        sequences = [
            {target: partial_waveform(pulse, i) for target, pulse in pulses.items()}
            for i in indices
        ]

        result = self.state_evolution_tomography(
            sequences=sequences,
            x90=x90,
            initial_state=initial_state,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        if plot:
            times = pulses.popitem()[1].times[indices]
            for target, states in result.items():
                vis.plot_bloch_vectors(
                    times=times,
                    bloch_vectors=states,
                    title=f"State evolution of {target}",
                )

        return result

    def measure_phase_shift_by_transmission_line(
        self,
        target: str,
        *,
        frequency_range: ArrayLike = np.arange(10.05, 10.1, 0.002),
        amplitude: float = 0.01,
        subrange_width: float = 0.3,
        shots: int = 128,
        interval: int = 0,
        plot: bool = False,
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
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        float
            Phase shift in rad/GHz.
        """
        read_label = Target.read_label(target)
        qubit_label = Target.qubit_label(target)
        mux = self.experiment_system.get_mux_by_qubit(qubit_label)

        # split frequency range to avoid the frequency sweep range limit
        frequency_range = np.array(frequency_range)
        subranges = ExperimentUtil.split_frequency_range(
            frequency_range=frequency_range,
            subrange_width=subrange_width,
        )

        # result buffer
        phases: list[float] = []
        # measure phase shift
        idx = 0

        # phase offset to avoid the phase jump after changing the LO/NCO frequency
        phase_offset = 0.0

        for subrange in subranges:
            # change LO/NCO frequency to the center of the subrange
            f_center = (subrange[0] + subrange[-1]) / 2
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f_center * 1e9,
                ssb="U",
                cnco_center=1_500_000_000,
            )
            with self.state_manager.modified_device_settings(
                label=read_label,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            ):
                for sub_idx, freq in enumerate(subrange):
                    if idx > 0 and sub_idx == 0:
                        # measure the phase at the previous frequency with the new LO/NCO settings
                        with self.modified_frequencies(
                            {read_label: frequency_range[idx - 1]}
                        ):
                            new_result = self.measure(
                                {qubit_label: np.zeros(0)},
                                mode="avg",
                                readout_amplitudes={qubit_label: amplitude},
                                shots=shots,
                                interval=interval,
                                plot=False,
                            )
                            new_signal = new_result.data[target].kerneled
                            new_phase = np.angle(new_signal)
                            phase_offset = new_phase - phases[-1]  # type: ignore

                    with self.modified_frequencies({read_label: freq}):
                        result = self.measure(
                            {qubit_label: np.zeros(0)},
                            mode="avg",
                            readout_amplitudes={qubit_label: amplitude},
                            shots=shots,
                            interval=interval,
                            plot=False,
                        )
                        signal = result.data[target].kerneled
                        phase = np.angle(signal)
                        phase = phase - phase_offset
                        phases.append(phase)  # type: ignore

                        idx += 1

        # fit the phase shift
        x, y = frequency_range, np.unwrap(phases)
        coefficients = np.polyfit(x, y, 1)
        y_fit = np.polyval(coefficients, x)

        if plot:
            fig = go.Figure()
            fig.add_scatter(name="data", mode="markers", x=x, y=y)
            fig.add_scatter(name="fit", mode="lines", x=x, y=y_fit)
            fig.update_layout(
                title=f"Phase shift by transmission line : {mux.label}",
                xaxis_title="Frequency (GHz)",
                yaxis_title="Unwrapped phase (rad)",
                showlegend=True,
            )
            fig.show()

        # return the phase shift
        phase_shift = coefficients[0]
        return phase_shift

    def scan_resonator_frequencies(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = np.arange(9.75, 10.75, 0.002),
        phase_shift: float | None = None,
        amplitude: float = 0.01,
        subrange_width: float = 0.3,
        shots: int = 128,
        interval: int = 0,
        plot: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
        """
        Scans the readout frequencies to find the resonator frequencies.

        Parameters
        ----------
        target : str
            Target qubit connected to the resonator of interest.
        frequency_range : ArrayLike, optional
            Frequency range of the scan in GHz.
        phase_shift : float, optional
            Phase shift in rad/GHz. If None, it will be measured.
        amplitude : float, optional
            Amplitude of the readout pulse. Defaults to 0.01.
        subrange_width : float, optional
            Width of the frequency subrange in GHz. Defaults to 0.3.
        shots : int, optional
            Number of shots. Defaults to 128.
        interval : int, optional
            Interval between shots. Defaults to 0.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]
            Frequency range, phase difference and raw complex signals.
        """
        # measure phase shift if not provided
        if phase_shift is None:
            phase_shift = self.measure_phase_shift_by_transmission_line(
                target,
                amplitude=amplitude,
                shots=shots,
                interval=interval,
                plot=False,
            )

        read_label = Target.read_label(target)
        qubit_label = Target.qubit_label(target)
        mux = self.experiment_system.get_mux_by_qubit(qubit_label)

        # split frequency range to avoid the frequency sweep range limit
        frequency_range = np.array(frequency_range)
        subranges = ExperimentUtil.split_frequency_range(
            frequency_range=frequency_range,
            subrange_width=subrange_width,
        )

        phases: list[float] = []
        signals: list[complex] = []

        idx = 0
        phase_offset = 0.0

        for subrange in subranges:
            f_center = (subrange[0] + subrange[-1]) / 2
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f_center * 1e9,
                ssb="U",
                cnco_center=1_500_000_000,
            )
            with self.state_manager.modified_device_settings(
                label=read_label,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            ):
                for sub_idx, freq in enumerate(subrange):
                    if idx > 0 and sub_idx == 0:
                        prev_freq = frequency_range[idx - 1]
                        with self.modified_frequencies({read_label: prev_freq}):
                            new_result = self.measure(
                                {qubit_label: np.zeros(0)},
                                mode="avg",
                                readout_amplitudes={qubit_label: amplitude},
                                shots=shots,
                                interval=interval,
                            )
                            new_signal = new_result.data[target].kerneled
                            new_phase = np.angle(new_signal) - prev_freq * phase_shift
                            phase_offset = new_phase - phases[-1]  # type: ignore

                    with self.modified_frequencies({read_label: freq}):
                        result = self.measure(
                            {qubit_label: np.zeros(0)},
                            mode="avg",
                            readout_amplitudes={qubit_label: amplitude},
                            shots=shots,
                            interval=interval,
                        )

                        raw = result.data[target].kerneled
                        ampl = np.abs(raw)
                        phase = np.angle(raw)
                        phase = phase - freq * phase_shift - phase_offset
                        signal = ampl * np.exp(1j * phase)
                        signals.append(signal)
                        phases.append(phase)  # type: ignore

                        idx += 1

        phases_unwrap = np.unwrap(phases)
        phases_diff = np.abs(np.diff(phases_unwrap))

        if plot:
            fig1 = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                # subplot_titles=["Phase", "Amplitude"],
            )
            fig1.add_scatter(
                name=target,
                mode="markers+lines",
                row=1,
                col=1,
                x=frequency_range,
                y=phases_unwrap,
            )
            fig1.add_scatter(
                name=target,
                mode="markers+lines",
                row=2,
                col=1,
                x=frequency_range,
                y=np.abs(signals),
            )
            fig1.update_xaxes(title_text="Readout frequency (GHz)", row=2, col=1)
            fig1.update_yaxes(title_text="Unwrapped phase (rad)", row=1, col=1)
            fig1.update_yaxes(title_text="Amplitude (arb. units)", row=2, col=1)
            fig1.update_layout(
                title=f"Resonator frequency scan : {mux.label}",
                height=450,
                showlegend=False,
            )
            fig1.show()

            fig2 = go.Figure()
            fig2.add_scatter(
                name=target,
                mode="markers+lines",
                x=frequency_range,
                y=phases_diff,
            )
            fig2.update_layout(
                title=f"Resonator frequency scan : {mux.label}",
                xaxis_title="Readout frequency (GHz)",
                yaxis_title="Phase diff (rad)",
            )
            fig2.show()

        return frequency_range, phases_diff, np.array(signals)

    def resonator_spectroscopy(
        self,
        target: str,
        *,
        phase_shift: float,
        frequency_range: ArrayLike = np.arange(9.75, 10.75, 0.002),
        power_range: ArrayLike = np.arange(-60, 5, 5),
        shots: int = 128,
        interval: int = 0,
        plot: bool = True,
        save_image: bool = True,
    ) -> NDArray[np.float64]:
        """
        Conducts a resonator spectroscopy experiment.

        Parameters
        ----------
        target : str
            Target qubit connected to the resonator of interest.
        phase_shift : float
            Phase shift in rad/GHz.
        frequency_range : ArrayLike, optional
            Frequency range of the scan in GHz. Defaults to np.arange(9.75, 10.75, 0.002).
        power_range : ArrayLike, optional
            Power range in dB. Defaults to np.arange(-60, 5, 5).
        shots : int, optional
            Number of shots. Defaults to 128.
        interval : int, optional
            Interval between shots. Defaults to 0.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        save_image : bool, optional
            Whether to save the image. Defaults to True.

        Returns
        -------
        NDArray[np.float64]
            Phase shift in rad.
        """
        power_range = np.array(power_range)
        qubit_label = Target.qubit_label(target)
        mux = self.experiment_system.get_mux_by_qubit(qubit_label)

        result = []
        for power in tqdm(power_range):
            power_linear = 10 ** (power / 10)
            amplitude = np.sqrt(power_linear)
            _, phase_diff, _ = self.scan_resonator_frequencies(
                target,
                frequency_range=frequency_range,
                phase_shift=phase_shift,
                amplitude=amplitude,
                shots=shots,
                interval=interval,
                plot=False,
            )
            phase_diff = np.append(phase_diff, phase_diff[-1])
            result.append(phase_diff)
        if plot:
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=frequency_range,
                    y=power_range,
                    z=result,
                    colorscale="Viridis",
                    colorbar=dict(
                        title=dict(
                            text="Phase shift (rad)",
                            side="right",
                        ),
                    ),
                )
            )
            fig.update_layout(
                title=f"Resonator spectroscopy : {mux.label}",
                xaxis_title="Frequency (GHz)",
                yaxis_title="Power (dB)",
                width=600,
                height=300,
            )
            fig.show()

        if save_image:
            vis.save_image(
                fig,
                name=f"resonator_spectroscopy_{mux.label}",
            )

        return np.array(result)

    def measure_reflection_coefficient(
        self,
        target: str,
        *,
        frequency_range: ArrayLike,
        phase_shift: float,
        amplitude: float = 0.01,
        shots: int = 128,
        interval: int = 0,
        plot: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128], float, float, float]:
        """
        Scans the readout frequencies to find the resonator frequencies.

        Parameters
        ----------
        target : str
            Target qubit connected to the resonator of interest.
        frequency_range : ArrayLike
            Frequency range of the scan in GHz.
        phase_shift : float
            Phase shift in rad/GHz.
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
        tuple[NDArray[np.float64], NDArray[np.complex128], float, float, float]
            Frequency range and reflection coefficients (complex), resonator frequency, external coupling rate, internal coupling rate.
        """
        freq_range = np.array(frequency_range)
        center_frequency = np.mean(freq_range).astype(float)
        frequency_width = (freq_range[-1] - freq_range[0]).astype(float)

        if frequency_width > 0.4:
            raise ValueError("Frequency scan range must be less than 400 MHz.")

        read_label = Target.read_label(target)
        qubit_label = Target.qubit_label(target)

        lo, cnco, _ = MixingUtil.calc_lo_cnco(
            center_frequency * 1e9,
            ssb="U",
            cnco_center=1_500_000_000,
        )

        signals = []

        with self.state_manager.modified_device_settings(
            label=read_label,
            lo_freq=lo,
            cnco_freq=cnco,
            fnco_freq=0,
        ):
            for freq in freq_range:
                with self.modified_frequencies({read_label: freq}):
                    result = self.measure(
                        {qubit_label: np.zeros(0)},
                        mode="avg",
                        readout_amplitudes={qubit_label: amplitude},
                        shots=shots,
                        interval=interval,
                    )
                    signal = result.data[target].kerneled
                    signal = signal * np.exp(-1j * freq * phase_shift)
                    signals.append(signal)

        # if plot:
        #     fig = go.Figure()
        #     fig.add_trace(
        #         go.Scatter(
        #             x=freq_range,
        #             y=np.unwrap(np.angle(signals)),
        #             mode="markers+lines",
        #         )
        #     )
        #     fig.update_layout(
        #         title=f"Phase of reflection wave : {qubit_label}",
        #         xaxis_title="Frequency (GHz)",
        #         yaxis_title="Phase (rad)",
        #     )
        #     fig.show()

        phi = (np.angle(signals[0]) + np.angle(signals[-1])) / 2
        coeffs = np.array(signals) * np.exp(-1j * phi)

        f_r, kappa_ex, kappa_in = fitting.fit_reflection_coefficient(
            target=target,
            freq_range=freq_range,
            data=coeffs,
            plot=plot,
        )

        return freq_range, coeffs, f_r, kappa_ex, kappa_in

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
        # control and readout pulses
        qubit = Target.qubit_label(target)
        resonator = Target.read_label(target)
        control_pulse = Gaussian(
            duration=1024,
            amplitude=control_amplitude,
            sigma=128,
        )
        readout_pulse = FlatTop(
            duration=1024,
            amplitude=readout_amplitude,
            tau=128,
        )

        # split frequency range to avoid the frequency sweep range limit
        frequency_range = np.array(frequency_range)
        subranges = ExperimentUtil.split_frequency_range(
            frequency_range=frequency_range,
            subrange_width=subrange_width,
        )

        # readout frequency
        readout_frequency = readout_frequency or self.targets[resonator].frequency

        # result buffer
        phases = []
        amplitudes = []

        # measure the phase and amplitude
        idx = 0
        for subrange in subranges:
            f_center = (subrange[0] + subrange[-1]) / 2
            lo, cnco, _ = MixingUtil.calc_lo_cnco(
                f_center * 1e9,
                ssb="L",
                cnco_center=2_250_000_000,
            )
            with self.state_manager.modified_device_settings(
                label=qubit,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            ):
                for control_frequency in subrange:
                    with self.modified_frequencies(
                        {
                            qubit: control_frequency,
                            resonator: readout_frequency,
                        }
                    ):
                        with PulseSchedule([qubit, resonator]) as ps:
                            ps.add(qubit, control_pulse)
                            ps.add(resonator, readout_pulse)
                        result = self.execute(
                            schedule=ps,
                            mode="avg",
                            shots=shots,
                            interval=interval,
                        )
                        signal = result.data[qubit].kerneled
                        phases.append(np.angle(signal))
                        amplitudes.append(np.abs(signal))

                        idx += 1

        phases_unwrap = np.unwrap(phases)
        if np.min(phases_unwrap) < 0:
            phases_unwrap += 2 * np.pi
        elif np.max(phases_unwrap) > 2 * np.pi:
            phases_unwrap -= 2 * np.pi

        if plot:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
            )
            fig.add_scatter(
                name=target,
                mode="markers+lines",
                row=1,
                col=1,
                x=frequency_range,
                y=phases_unwrap,
            )
            fig.add_scatter(
                name=target,
                mode="markers+lines",
                row=2,
                col=1,
                x=frequency_range,
                y=amplitudes,
            )
            fig.update_xaxes(title_text="Control frequency (GHz)", row=2, col=1)
            fig.update_yaxes(title_text="Unwrapped phase (rad)", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude (arb. units)", row=2, col=1)
            fig.update_layout(
                title=f"Control frequency scan : {qubit}",
                height=450,
                showlegend=False,
            )
            fig.show()

        return frequency_range, phases_unwrap, np.asarray(amplitudes)

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
        power_range = np.array(power_range)
        result = []
        for power in tqdm(power_range):
            power_linear = 10 ** (power / 10)
            amplitude = np.sqrt(power_linear)
            _, phase, _ = self.scan_qubit_frequencies(
                target,
                frequency_range=frequency_range,
                control_amplitude=amplitude,
                readout_amplitude=readout_amplitude,
                readout_frequency=readout_frequency,
                shots=shots,
                interval=interval,
                plot=False,
            )
            result.append(phase)
        if plot:
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=frequency_range,
                    y=power_range,
                    z=result,
                    colorscale="Viridis",
                    colorbar=dict(
                        title=dict(
                            text="Phase (rad)",
                            side="right",
                        )
                    ),
                )
            )
            fig.update_layout(
                title=f"Qubit spectroscopy : {target}",
                xaxis_title="Frequency (GHz)",
                yaxis_title="Power (dB)",
                width=600,
                height=300,
            )
            fig.show()

        if save_image:
            vis.save_image(
                fig,
                name=f"qubit_spectroscopy_{target}",
                width=600,
                height=300,
            )

        return np.array(result)

    def measure_population(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        fit_gmm: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """
        Measures the state populations of the target qubits.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Sequence to measure for each target.
        fit_gmm : bool, optional
            Whether to fit the data with a Gaussian mixture model. Defaults to False
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]
            State probabilities and standard deviations.

        Examples
        --------
        >>> sequence = {
        ...     "Q00": ex.hpi_pulse["Q00"],
        ...     "Q01": ex.hpi_pulse["Q01"],
        ... }
        >>> result = ex.measure_population(sequence)
        """
        if self.classifiers is None:
            raise ValueError("Classifiers are not built. Run `build_classifier` first.")

        result = self.measure(
            sequence,
            mode="single",
            shots=shots,
            interval=interval,
        )
        if fit_gmm:
            probabilities = {
                target: self.classifiers[target].estimate_weights(
                    result.data[target].kerneled
                )
                for target in result.data
            }
            standard_deviations = {
                target: np.zeros_like(probabilities[target]) for target in probabilities
            }
        else:
            probabilities = {
                target: data.probabilities for target, data in result.data.items()
            }
            standard_deviations = {
                target: data.standard_deviations for target, data in result.data.items()
            }
        return probabilities, standard_deviations

    def measure_population_dynamics(
        self,
        *,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        params_list: Sequence | NDArray,
        fit_gmm: bool = False,
        xlabel: str = "Index",
        scatter_mode: str = "lines+markers",
        show_error: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """
        Measures the population dynamics of the target qubits.

        Parameters
        ----------
        sequence : ParametricPulseSchedule | ParametricWaveformDict
            Parametric sequence to measure.
        params_list : Sequence | NDArray
            List of parameters.
        fit_gmm : bool, optional
            Whether to fit the data with a Gaussian mixture model. Defaults to False.
        xlabel : str, optional
            Label of the x-axis. Defaults to "Index".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]
            State probabilities and standard deviations.

        Examples
        --------
        >>> sequence = lambda x: {
        ...     "Q00": ex.hpi_pulse["Q00"].scaled(x),
        ...     "Q01": ex.hpi_pulse["Q01"].scaled(x),
        >>> params_list = np.linspace(0.5, 1.5, 100)
        >>> result = ex.measure_popultion_dynamics(sequence, params_list)
        """
        if isinstance(params_list[0], int):
            x = params_list
        else:
            try:
                float(params_list[0])
                x = params_list
            except ValueError:
                x = np.arange(len(params_list))

        buffer_pops = defaultdict(list)
        buffer_errs = defaultdict(list)

        for params in tqdm(params_list):
            prob_dict, err_dict = self.measure_population(
                sequence=sequence(params),
                fit_gmm=fit_gmm,
                shots=shots,
                interval=interval,
            )
            for target, probs in prob_dict.items():
                buffer_pops[target].append(probs)
            for target, errors in err_dict.items():
                buffer_errs[target].append(errors)

        result_pops = {
            target: np.array(buffer_pops[target]).T for target in buffer_pops
        }
        result_errs = {
            target: np.array(buffer_errs[target]).T for target in buffer_errs
        }

        for target in result_pops:
            fig = go.Figure()
            for state, probs in enumerate(result_pops[target]):
                fig.add_scatter(
                    name=f"|{state}⟩",
                    mode=scatter_mode,
                    x=x,
                    y=probs,
                    error_y=(
                        dict(
                            type="data",
                            array=result_errs[target][state],
                            visible=True,
                            thickness=1.5,
                            width=3,
                        )
                        if show_error
                        else None
                    ),
                    marker=dict(size=5),
                )
            fig.update_layout(
                title=f"Population dynamics : {target}",
                xaxis_title=xlabel,
                yaxis_title="Probability",
                yaxis_range=[-0.1, 1.1],
            )
            fig.show()

        return result_pops, result_errs


class ExperimentUtil:
    @staticmethod
    def discretize_time_range(
        time_range: ArrayLike,
        sampling_period: float = SAMPLING_PERIOD,
    ) -> NDArray[np.float64]:
        """
        Discretizes the time range.

        Parameters
        ----------
        time_range : ArrayLike
            Time range to discretize in ns.
        sampling_period : float, optional
            Sampling period in ns. Defaults to SAMPLING_PERIOD.

        Returns
        -------
        NDArray[np.float64]
            Discretized time range.
        """
        discretized_range = np.array(time_range)
        discretized_range = (
            np.round(discretized_range / sampling_period) * sampling_period
        )
        return discretized_range

    @staticmethod
    def split_frequency_range(
        frequency_range: ArrayLike,
        subrange_width: float = 0.3,
    ) -> list[NDArray[np.float64]]:
        """
        Splits the frequency range into sub-ranges.

        Parameters
        ----------
        frequency_range : ArrayLike
            Frequency range to split in GHz.
        ubrange_width : float, optional
            Width of the sub-ranges. Defaults to 0.3 GHz.

        Returns
        -------
        list[NDArray[np.float64]]
            Sub-ranges.
        """
        frequency_range = np.array(frequency_range)
        range_count = (frequency_range[-1] - frequency_range[0]) // subrange_width + 1
        sub_ranges = np.array_split(frequency_range, range_count)
        return sub_ranges

    @staticmethod
    def create_qubit_subgroups(
        qubits: Collection[str],
    ) -> list[list[str]]:
        """
        Creates subgroups of qubits.

        Parameters
        ----------
        qubits : Collection[str]
            Collection of qubits.

        Returns
        -------
        list[list[str]]
            Subgroups of qubits.
        """
        # TODO: Implement a more general method
        qubit_labels = list(qubits)
        system = StateManager.shared().experiment_system
        qubit_objects = [system.get_qubit(qubit) for qubit in qubit_labels]
        group03 = [qubit.label for qubit in qubit_objects if qubit.index % 4 in [0, 3]]
        group12 = [qubit.label for qubit in qubit_objects if qubit.index % 4 in [1, 2]]
        return [group03, group12]

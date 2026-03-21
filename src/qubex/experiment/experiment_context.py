"""Experiment context setup and device access helpers."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from typing_extensions import deprecated

from qubex.backend.backend_controller import (
    BACKEND_KIND_QUEL1,
    BACKEND_KIND_QUEL3,
    SystemBackendController,
)
from qubex.measurement import (
    Measurement,
    StateClassifier,
)
from qubex.measurement.measurement_defaults import resolve_measurement_defaults
from qubex.system import (
    Box,
    Chip,
    ConfigLoader,
    ControlParameters,
    ControlSystem,
    ExperimentSystem,
    GenPort,
    MixingUtil,
    Mux,
    QuantumSystem,
    Qubit,
    Resonator,
    SystemManager,
    Target,
)
from qubex.system.config_paths import resolve_default_calibration_note_path
from qubex.system.target_type import TargetType
from qubex.typing import ConfigurationMode, TargetMap
from qubex.version import get_version

from . import experiment_tool
from .experiment_constants import (
    CALIBRATION_VALID_DAYS,
    CLASSIFIER_DIR,
    DRAG_HPI_DURATION,
    DRAG_PI_DURATION,
    PROPERTY_DIR,
    SYSTEM_NOTE_PATH,
    USER_NOTE_PATH,
)
from .experiment_exceptions import CalibrationMissingError
from .experiment_util import ExperimentUtil
from .models.calibration_note import CalibrationNote
from .models.experiment_note import ExperimentNote
from .models.experiment_record import ExperimentRecord
from .models.rabi_param import RabiParam

logger = logging.getLogger(__name__)

console = Console()


class ExperimentContext:
    """
    Class representing the context of an experiment.

    Parameters
    ----------
    chip_id : str | None, optional
        Deprecated chip identifier compatibility input.
    system_id : str | None, optional
        Canonical system identifier used to resolve configuration resources.
    muxes : Collection[str | int], optional
        Mux labels to use in the experiment.
    qubits : Collection[str | int], optional
        Qubit labels to use in the experiment.
    exclude_qubits : Collection[str | int], optional
        Qubit labels to exclude in the experiment.
    config_dir : str, optional
        Path to the configuration directory containing:
          - box.yaml
          - chip.yaml
          - wiring.yaml
          - skew.yaml
    params_dir : str, optional
        Path to the parameters directory containing:
          - params.yaml
          - props.yaml
    calib_note_path : Path | str, optional
        Path to the calibration note file.
    calibration_valid_days : int, optional
        Number of days for which the calibration is valid.
    drag_hpi_duration : int, optional
        Duration of the DRAG HPI pulse.
    drag_pi_duration : int, optional
        Duration of the DRAG π pulse.
    readout_duration : int, optional
        Duration of the readout pulse.
    readout_pre_margin : int, optional
        Pre-margin of the readout pulse.
    readout_post_margin : int, optional
        Post-margin of the readout pulse.
    classifier_dir : Path | str, optional
        Directory of the state classifiers.
    classifier_type : Literal["kmeans", "gmm"], optional
        Type of the state classifier. Defaults to "gmm".
    configuration_mode : ConfigurationMode, optional
        Priority-ordered control layout. `"ge-ef-cr"` assigns channels to GE,
        then EF, then CR. `"ge-cr-cr"` assigns GE, then two CR channels.
        Ports with fewer channels keep the leftmost roles. Defaults to
        `"ge-cr-cr"`.

    Examples
    --------
    >>> from qubex import Experiment
    >>> exp = Experiment(
    ...     system_id="64Q-HF-Q1",
    ...     qubits=["Q00", "Q01"],
    ... )
    """

    def __init__(
        self,
        *,
        chip_id: str | None = None,
        system_id: str | None = None,
        muxes: Collection[str | int] | None = None,
        qubits: Collection[str | int] | None = None,
        exclude_qubits: Collection[str | int] | None = None,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        calib_note_path: Path | str | None = None,
        calibration_valid_days: int | None = None,
        drag_hpi_duration: float | None = None,
        drag_pi_duration: float | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        property_dir: Path | str | None = None,
        classifier_dir: Path | str | None = None,
        classifier_type: Literal["kmeans", "gmm"] | None = None,
        configuration_mode: ConfigurationMode | None = None,
        backend_controller: SystemBackendController | None = None,
        mock_mode: bool | None = None,
    ):
        if calibration_valid_days is None:
            calibration_valid_days = CALIBRATION_VALID_DAYS
        if drag_hpi_duration is None:
            drag_hpi_duration = DRAG_HPI_DURATION
        if drag_pi_duration is None:
            drag_pi_duration = DRAG_PI_DURATION
        if property_dir is None:
            property_dir = PROPERTY_DIR
        if classifier_dir is None:
            classifier_dir = CLASSIFIER_DIR
        if classifier_type is None:
            classifier_type = "gmm"
        if configuration_mode is None:
            configuration_mode = "ge-cr-cr"
        if mock_mode is None:
            mock_mode = False

        if chip_id is None and system_id is None:
            raise ValueError("Either `system_id` or `chip_id` must be provided.")

        self._load_config(
            chip_id=chip_id,
            system_id=system_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
            backend_controller=backend_controller,
            mock_mode=mock_mode,
        )
        measurement_defaults = resolve_measurement_defaults(
            self.experiment_system.measurement_defaults
        )
        if readout_duration is None:
            readout_duration = measurement_defaults.readout.duration_ns
        if readout_pre_margin is None:
            readout_pre_margin = measurement_defaults.readout.pre_margin_ns
        if readout_post_margin is None:
            readout_post_margin = measurement_defaults.readout.post_margin_ns
        qubits = self._create_qubit_labels(
            muxes=muxes,
            qubits=qubits,
            exclude_qubits=exclude_qubits,
        )
        resolved_chip_id = self.system_manager.config_loader.chip_id
        self._chip_id: Final = resolved_chip_id
        self._qubits: Final = qubits
        self._readout_duration: Final = readout_duration
        self._readout_pre_margin: Final = readout_pre_margin
        self._readout_post_margin: Final = readout_post_margin
        self._drag_hpi_duration: Final = drag_hpi_duration
        self._drag_pi_duration: Final = drag_pi_duration
        self._property_dir: Final = property_dir
        self._classifier_dir: Final = classifier_dir
        self._classifier_type: Final = classifier_type
        self._configuration_mode: Final = configuration_mode
        self._calibration_valid_days: Final = calibration_valid_days
        self._measurement = Measurement(
            chip_id=resolved_chip_id,
            system_id=system_id,
            qubits=qubits,
            load_configs=False,
            connect_devices=False,
        )
        self._user_note: Final = ExperimentNote(file_path=USER_NOTE_PATH)
        self._system_note: Final = ExperimentNote(file_path=SYSTEM_NOTE_PATH)
        self._calib_note: Final = CalibrationNote(
            chip_id=resolved_chip_id,
            file_path=calib_note_path,
        )
        self._load_skew_file()
        self._load_classifiers()
        self.print_environment(verbose=False)

    def _load_classifiers(self):
        for qubit in self.qubit_labels:
            classifier_path = self.classifier_dir / self.chip_id / f"{qubit}.pkl"
            if classifier_path.exists():
                try:
                    classifier = StateClassifier.load(classifier_path)
                except Exception as exc:
                    if not self._is_classifier_compatibility_error(exc):
                        raise
                    logger.warning(
                        (
                            f"Failed to load state classifier for {qubit} from "
                            f"`{classifier_path}` due to a compatibility issue: "
                            f"{exc}. The classifier was skipped."
                        ),
                    )
                    continue
                self._measurement.update_classifiers({qubit: classifier})

    @staticmethod
    def _is_classifier_compatibility_error(exc: BaseException) -> bool:
        """Return whether a classifier load failure looks compatibility-related."""
        if isinstance(
            exc, (ModuleNotFoundError, ImportError, AttributeError, TypeError)
        ):
            return True
        next_exc = exc.__cause__ or exc.__context__
        if next_exc is None:
            return False
        return ExperimentContext._is_classifier_compatibility_error(next_exc)

    def _load_config(
        self,
        chip_id: str | None,
        system_id: str | None,
        config_dir: Path | str | None,
        params_dir: Path | str | None,
        configuration_mode: ConfigurationMode,
        backend_controller: SystemBackendController | None = None,
        mock_mode: bool | None = None,
    ):
        """Load the configuration files."""
        if mock_mode is None:
            mock_mode = False
        self.system_manager.load(
            chip_id=chip_id,
            system_id=system_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
            backend_controller=backend_controller,
            mock_mode=mock_mode,
        )

    def _load_skew_file(self) -> None:
        """Load skew calibration data from the current config directory."""
        skew_file_path = self.config_loader.config_path / "skew.yaml"
        if not skew_file_path.exists():
            if self._should_warn_missing_skew_file():
                logger.warning(f"Skew file not found: {skew_file_path}")
            return
        backend_controller = self.backend_controller
        load_skew_yaml = getattr(backend_controller, "load_skew_yaml", None)
        if not callable(load_skew_yaml):
            return
        try:
            load_skew_yaml(skew_file_path)
        except Exception:
            logger.exception("Failed to load the skew file.")

    def _should_warn_missing_skew_file(self) -> bool:
        """Return whether a missing skew file should emit a warning."""
        return (
            self.config_loader.backend_kind == BACKEND_KIND_QUEL1
            and len(self.control_system.boxes) > 1
        )

    def _create_qubit_labels(
        self,
        muxes: Collection[str | int] | None,
        qubits: Collection[str | int] | None,
        exclude_qubits: Collection[str | int] | None,
    ) -> list[str]:
        """Create the list of qubit labels."""
        if muxes is None and qubits is None:
            return []
        quantum_system = self.quantum_system
        qubit_labels = []
        if muxes is not None:
            for mux in muxes:
                labels = [
                    qubit.label for qubit in quantum_system.get_qubits_in_mux(mux)
                ]
                qubit_labels.extend(labels)
        if qubits is not None:
            qubit_labels.extend(
                quantum_system.get_qubit(qubit).label for qubit in qubits
            )
        if exclude_qubits is not None:
            for qubit in exclude_qubits:
                label = quantum_system.get_qubit(qubit).label
                if label in qubit_labels:
                    qubit_labels.remove(label)
        qubit_labels = sorted(set(qubit_labels))

        available_qubits = [
            target.qubit for target in self.experiment_system.ge_targets
        ]
        unavailable_qubits = [
            qubit for qubit in qubit_labels if qubit not in available_qubits
        ]
        if len(unavailable_qubits) > 0:
            logger.warning(f"Unavailable qubits: {unavailable_qubits}")

        qubit_labels = [qubit for qubit in qubit_labels if qubit in available_qubits]

        return qubit_labels

    def print_environment(self, verbose: bool | None = None) -> None:
        """Print the environment information."""
        if verbose is None:
            verbose = True
        logger.info("========================================")
        logger.info(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"python: {sys.version.split()[0]}")
        logger.info(f"qubex: {get_version('qubex')}")
        if verbose:
            logger.info(f"  qxcore: {get_version('qxcore')}")
            logger.info(f"  qxdriver-quel1: {get_version('qxdriver-quel1')}")
            logger.info(f"  qxpulse: {get_version('qxpulse')}")
            logger.info(f"  qxschema: {get_version('qxschema')}")
            logger.info(f"  qxsimulator: {get_version('qxsimulator')}")
            logger.info(f"  quel_ic_config: {get_version('quel_ic_config')}")
            logger.info(f"  numpy: {get_version('numpy')}")
            logger.info(f"  scipy: {get_version('scipy')}")
            logger.info(f"  scikit-learn: {get_version('scikit-learn')}")
        logger.info(f"env: {sys.prefix}")
        logger.info(f"config: {self.config_path}")
        logger.info(f"params: {self.params_path}")
        logger.info(f"chip: {self.chip_id} ({self.chip.name})")
        logger.info(f"qubits: {self.qubit_labels}")
        logger.info(f"muxes: {self.mux_labels}")
        logger.info(f"boxes: {self.box_ids}")
        logger.info("========================================")

    def print_boxes(self) -> None:
        """Print the box information."""
        if not logger.isEnabledFor(logging.INFO):
            return

        table = Table(header_style="bold")
        table.add_column("ID", justify="left")
        table.add_column("NAME", justify="left")
        table.add_column("ADDRESS", justify="left")
        table.add_column("ADAPTER", justify="left")
        for box in self.boxes.values():
            table.add_row(box.id, box.name, box.address, box.adapter)
        console.print(table)

    @property
    def tool(self):  # noqa: ANN201
        """Return the experiment tool module."""
        return experiment_tool

    @property
    def util(self) -> Any:
        """Return the experiment utility class."""
        return ExperimentUtil

    @property
    def measurement(self) -> Measurement:
        """Return the measurement instance."""
        return self._measurement

    @property
    def system_manager(self) -> SystemManager:
        """Return the shared system manager."""
        return SystemManager.shared()

    @property
    def config_loader(self) -> ConfigLoader:
        """Return the configuration loader."""
        return self.system_manager.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Return the experiment system."""
        return self.system_manager.experiment_system

    @property
    def quantum_system(self) -> QuantumSystem:
        """Return the quantum system."""
        return self.experiment_system.quantum_system

    @property
    def control_system(self) -> ControlSystem:
        """Return the control system."""
        return self.experiment_system.control_system

    @property
    @deprecated("Use `backend_controller` instead.")
    def device_controller(self) -> SystemBackendController:
        """Return the device controller."""
        return self.system_manager.device_controller

    @property
    def backend_controller(self) -> SystemBackendController:
        """Return the backend controller."""
        return self.system_manager.backend_controller

    @property
    def params(self) -> ControlParameters:
        """Return the control parameters."""
        return self.experiment_system.control_params

    @property
    def chip(self) -> Chip:
        """Return the chip model."""
        return self.experiment_system.chip

    @property
    def chip_id(self) -> str:
        """Return the chip identifier."""
        return self._chip_id

    @property
    def qubit_labels(self) -> list[str]:
        """Return the list of active qubit labels."""
        return self._qubits

    @property
    def resonator_labels(self) -> list[str]:
        """Return the list of active resonator labels."""
        resonators = self.resonators
        return [
            resonators[qubit].label
            for qubit in self.qubit_labels
            if qubit in resonators
        ]

    @property
    def mux_labels(self) -> list[str]:
        """Return the list of mux labels for the active qubits."""
        mux_set = set()
        for qubit in self.qubit_labels:
            mux = self.experiment_system.get_mux_by_qubit(qubit)
            mux_set.add(mux.label)
        return sorted(mux_set)

    @property
    def qubits(self) -> dict[str, Qubit]:
        """Return qubit objects keyed by label."""
        return {
            qubit.label: qubit
            for qubit in self.experiment_system.qubits
            if qubit.label in self.qubit_labels
        }

    @property
    def resonators(self) -> dict[str, Resonator]:
        """Return resonator objects keyed by qubit label."""
        return {
            resonator.qubit: resonator
            for resonator in self.experiment_system.resonators
            if resonator.qubit in self.qubit_labels
        }

    @property
    def targets(self) -> dict[str, Target]:
        """Return all targets related to active qubits."""
        return {
            target.label: target
            for target in self.experiment_system.targets
            if target.is_related_to_qubits(self.qubit_labels)
            and self._is_visible_target_for_active_qubits(target)
        }

    def _is_visible_target_for_active_qubits(self, target: Target) -> bool:
        """Return whether one target should be exposed for the active qubits."""
        if not target.is_cr:
            return True
        try:
            _control_qubit, target_qubit = self.experiment_system.resolve_cr_pair(
                target.label
            )
        except ValueError:
            return True
        return target_qubit == "CR" or target_qubit in self.qubit_labels

    @property
    def available_targets(self) -> dict[str, Target]:
        """Return available targets keyed by label."""
        return {
            label: target
            for label, target in self.targets.items()
            if target.is_available
        }

    @property
    def ge_targets(self) -> dict[str, Target]:
        """Return available GE targets."""
        return {
            label: target
            for label, target in self.available_targets.items()
            if target.is_ge
        }

    @property
    def ef_targets(self) -> dict[str, Target]:
        """Return available EF targets."""
        return {
            label: target
            for label, target in self.available_targets.items()
            if target.is_ef
        }

    @property
    def cr_targets(self) -> dict[str, Target]:
        """Return available CR targets."""
        return {
            label: target
            for label, target in self.available_targets.items()
            if target.is_cr
        }

    @property
    def cr_labels(self) -> list[str]:
        """Return CR labels for the current chip."""
        return self.get_cr_labels()

    @property
    def cr_pairs(self) -> list[tuple[str, str]]:
        """Return CR qubit pairs."""
        return self.get_cr_pairs()

    @property
    def edge_pairs(self) -> list[tuple[str, str]]:
        """Return edge pairs in the chip graph."""
        return self.get_edge_pairs()

    @property
    def edge_labels(self) -> list[str]:
        """Return edge labels in the chip graph."""
        return self.get_edge_labels()

    @property
    def boxes(self) -> dict[str, Box]:
        """Return control/readout boxes keyed by ID."""
        boxes = self.experiment_system.get_boxes_for_qubits(self.qubit_labels)
        return {box.id: box for box in boxes}

    @property
    def box_ids(self) -> list[str]:
        """Return the list of box IDs."""
        return list(self.boxes.keys())

    @property
    def config_path(self) -> str:
        """Return the resolved configuration path."""
        return str(Path(self.config_loader.config_path).resolve())

    @property
    def params_path(self) -> str:
        """Return the resolved parameters path."""
        return str(Path(self.config_loader.params_path).resolve())

    @property
    def calib_note(self) -> CalibrationNote:
        """Return the calibration note instance."""
        return self._calib_note

    @property
    def note(self) -> ExperimentNote:
        """Return the user experiment note instance."""
        return self._user_note

    @property
    def calibration_valid_days(self) -> int:
        """Return calibration validity period in days."""
        return self._calibration_valid_days

    @property
    def readout_duration(self) -> float:
        """Return the readout duration."""
        return self._readout_duration

    @property
    def readout_pre_margin(self) -> float:
        """Return the readout pre margin."""
        return self._readout_pre_margin

    @property
    def readout_post_margin(self) -> float:
        """Return the readout post margin."""
        return self._readout_post_margin

    @property
    def drag_hpi_duration(self) -> float:
        """Return the DRAG half-pi duration."""
        return self._drag_hpi_duration

    @property
    def drag_pi_duration(self) -> float:
        """Return the DRAG pi duration."""
        return self._drag_pi_duration

    @property
    def property_dir(self) -> Path:
        """Return the property directory path."""
        return Path(self._property_dir)

    @property
    def classifier_dir(self) -> Path:
        """Return the classifier directory path."""
        return Path(self._classifier_dir)

    @property
    def classifier_type(self) -> Literal["kmeans", "gmm"]:
        """Return the classifier type."""
        return self._classifier_type

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        """Return the active state classifiers."""
        return self._measurement.classifiers

    @property
    def state_centers(self) -> dict[str, dict[int, complex]]:
        """Return state centers from calibration notes."""
        result = {}
        for target in self.qubit_labels:
            param = self.calib_note.get_state_param(
                target,
                valid_days=self._calibration_valid_days,
            )
            if param is not None:
                result[target] = {
                    int(state): complex(center[0], center[1])
                    for state, center in param["centers"].items()
                }
        return result

    @property
    def configuration_mode(self) -> ConfigurationMode:
        """Return the configuration mode."""
        return self._configuration_mode

    @property
    def reference_phases(self) -> dict[str, float]:
        """Return reference phases by target."""
        return self.calib_note.reference_phases

    def load_property(self, property_name: str) -> dict:
        """Load a JSON property dictionary by name."""
        property_path = self.property_dir / self.chip_id / f"{property_name}.json"
        if property_path.exists():
            with open(property_path) as f:
                property_data = json.load(f)
                return property_data
        else:
            raise FileNotFoundError(f"Property file not found: {property_path}")

    def save_property(
        self,
        property_name: str,
        data: dict,
        *,
        save_path: Path | str | None = None,
    ) -> None:
        """Save a JSON property dictionary by name."""
        if save_path is not None:
            property_path = Path(save_path)
        else:
            property_path = self.property_dir / self.chip_id / f"{property_name}.json"
        if not property_path.parent.exists():
            property_path.parent.mkdir(parents=True)
        try:
            with open(property_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Property '{property_name}' saved to {property_path}")
        except Exception as e:
            raise OSError(f"Failed to save property '{property_name}': {e}") from e

    def load_calib_note(self, path: Path | str | None = None) -> None:
        """Load the calibration data from a given path or the default calibration note file."""
        if path is None:
            path = resolve_default_calibration_note_path(
                system_id=self.config_loader.system_id,
                chip_id=self.chip_id,
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"Calibration file '{path}' does not exist.")
        try:
            self._calib_note.load(path)
            logger.info(f"Calibration data loaded from {path}")
        except Exception as e:
            raise CalibrationMissingError(
                f"Failed to load calibration data from {path}: {e}"
            ) from e

    def get_qubit_label(self, index: int) -> str:
        """Get the qubit label from the given qubit index."""
        return self.quantum_system.get_qubit(index).label

    def get_resonator_label(self, index: int) -> str:
        """Get the resonator label from the given resonator index."""
        return self.quantum_system.get_resonator(index).label

    def get_cr_label(
        self,
        control_index: int,
        target_index: int,
    ) -> str:
        """Get the cross-resonance label from the given control and target qubit indices."""
        control_qubit = self.quantum_system.get_qubit(control_index)
        target_qubit = self.quantum_system.get_qubit(target_index)
        label = Target.cr_label(control_qubit.label, target_qubit.label)
        return label

    def get_cr_pairs(
        self,
        low_to_high: bool | None = None,
        high_to_low: bool | None = None,
    ) -> list[tuple[str, str]]:
        """Get the cross-resonance pairs."""
        if low_to_high is None:
            low_to_high = True
        if high_to_low is None:
            high_to_low = False
        target_registry = self.experiment_system.target_registry
        cr_pairs = []
        for label in self.cr_targets:
            try:
                pair = target_registry.resolve_cr_pair(label)
                if pair[1] == "CR":
                    continue
                control_qubit = self.quantum_system.get_qubit(pair[0])
                target_qubit = self.quantum_system.get_qubit(pair[1])
                if target_qubit.label not in self.available_targets:
                    continue
                # if control_qubit.frequency < target_qubit.frequency:
                if control_qubit.index % 4 in [0, 3]:
                    if low_to_high:
                        cr_pairs.append(pair)
                else:
                    if high_to_low:
                        cr_pairs.append(pair)
            except Exception as exc:
                logger.debug(
                    "Failed to parse CR target label %s: %s",
                    label,
                    exc,
                    exc_info=True,
                )
                continue
        return cr_pairs

    def get_cr_labels(
        self,
        low_to_high: bool | None = None,
        high_to_low: bool | None = None,
    ) -> list[str]:
        """Get the cross-resonance labels."""
        if low_to_high is None:
            low_to_high = True
        if high_to_low is None:
            high_to_low = False
        target_registry = self.experiment_system.target_registry
        return [
            target_registry.resolve_cr_label(*pair)
            for pair in self.get_cr_pairs(low_to_high, high_to_low)
        ]

    def get_edge_pairs(
        self,
    ) -> list[tuple[str, str]]:
        """Get the qubit edge pairs."""
        edge_pairs = []
        for qubit in self.qubit_labels:
            spectators = self.get_spectators(qubit, in_same_mux=True)
            for spectator in spectators:
                pair = (qubit, spectator.label)
                edge_pairs.append(pair)
        return edge_pairs

    def get_edge_labels(
        self,
    ) -> list[str]:
        """Get the qubit edge labels."""
        return [f"{pair[0]}-{pair[1]}" for pair in self.get_edge_pairs()]

    def cr_pair(self, cr_label: str) -> tuple[str, str]:
        """Return the control/target qubit pair for a CR label."""
        return self.experiment_system.resolve_cr_pair(cr_label)

    def resolve_qubit_label(self, label: str) -> str:
        """Resolve qubit label through the experiment system."""
        return self.experiment_system.resolve_qubit_label(label)

    def resolve_read_label(self, label: str) -> str:
        """Resolve readout label through the experiment system."""
        return self.experiment_system.resolve_read_label(label)

    def resolve_ge_label(self, label: str) -> str:
        """Resolve GE label through the experiment system."""
        return self.experiment_system.resolve_ge_label(label)

    def resolve_ef_label(self, label: str) -> str:
        """Resolve EF label through the experiment system."""
        return self.experiment_system.resolve_ef_label(label)

    def ordered_qubit_labels(self, labels: Sequence[str]) -> list[str]:
        """Resolve labels to qubits while preserving first appearance order."""
        return self.experiment_system.ordered_qubit_labels(labels)

    def get_rabi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> RabiParam | None:
        """Get the Rabi parameters for the given target."""
        if valid_days is None:
            valid_days = self._calibration_valid_days

        param = self.calib_note.get_rabi_param(
            target,
            valid_days=valid_days,
        )
        if param is not None:
            try:
                return RabiParam(
                    target=param.get("target"),
                    frequency=param.get("frequency"),
                    amplitude=param.get("amplitude"),
                    phase=param.get("phase"),
                    offset=param.get("offset"),
                    noise=param.get("noise"),
                    angle=param.get("angle"),
                    distance=param.get("distance"),
                    r2=param.get("r2"),
                    reference_phase=param.get("reference_phase"),
                )
            except TypeError:
                raise ValueError(
                    f"Invalid Rabi parameters for {target}: {param}"
                ) from None
        else:
            return None

    def store_rabi_params(
        self,
        rabi_params: dict[str, RabiParam],
        r2_threshold: float | None = None,
    ) -> None:
        """Store Rabi parameters that meet the quality threshold."""
        if r2_threshold is None:
            r2_threshold = 0.5
        not_stored = []
        for label, rabi_param in rabi_params.items():
            if not np.isfinite(rabi_param.r2):
                logger.info(
                    "Skipping Rabi parameter storage for %s: non-finite r2 (%s).",
                    label,
                    rabi_param.r2,
                )
                not_stored.append(label)
            elif rabi_param.r2 < r2_threshold:
                logger.info(
                    "Skipping Rabi parameter storage for %s: r2 %.6f is below threshold %.6f.",
                    label,
                    rabi_param.r2,
                    r2_threshold,
                )
                not_stored.append(label)
            else:
                self.calib_note.update_rabi_param(
                    label,
                    {
                        "target": rabi_param.target,
                        "frequency": rabi_param.frequency,
                        "amplitude": rabi_param.amplitude,
                        "phase": rabi_param.phase,
                        "offset": rabi_param.offset,
                        "noise": rabi_param.noise,
                        "angle": rabi_param.angle,
                        "distance": rabi_param.distance,
                        "r2": rabi_param.r2,
                        "reference_phase": rabi_param.reference_phase,
                    },
                )

        if len(not_stored) > 0:
            logger.warning(f"Rabi parameters are not stored for qubits: {not_stored}")

    def get_spectators(
        self,
        qubit: str,
        in_same_mux: bool | None = None,
    ) -> list[Qubit]:
        """Return spectator qubits for the given target."""
        if in_same_mux is None:
            in_same_mux = False
        return self.quantum_system.get_spectator_qubits(qubit, in_same_mux=in_same_mux)

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        """Return the confusion matrix for the specified targets."""
        return self.measurement.get_confusion_matrix(targets)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        """Return the inverse confusion matrix for the specified targets."""
        return self.measurement.get_inverse_confusion_matrix(targets)

    def reset_awg_and_capunits(
        self,
        box_ids: str | Collection[str] | None = None,
        qubits: Collection[str] | None = None,
    ) -> None:
        """Reset AWG and capture units for specified boxes or qubits."""
        initialize_awg_and_capunits = getattr(
            self.backend_controller, "initialize_awg_and_capunits", None
        )
        if not callable(initialize_awg_and_capunits):
            # QuEL-3 currently has no explicit reset capability. Treat reset
            # requests as a compatibility no-op and let execution continue.
            if self.config_loader.backend_kind == BACKEND_KIND_QUEL3:
                return
            raise NotImplementedError(
                "Active backend does not support AWG/CAP unit reset."
            )

        selected_box_ids: list[str] = []
        if qubits is not None:
            boxes = self.experiment_system.get_boxes_for_qubits(qubits)
            selected_box_ids += [box.id for box in boxes]
        if len(selected_box_ids) == 0:
            if isinstance(box_ids, str):
                selected_box_ids = [box_ids]
            elif box_ids is not None:
                selected_box_ids = list(box_ids)
            else:
                selected_box_ids = self.box_ids
        initialize_awg_and_capunits(selected_box_ids)

    def _resolve_custom_target_qubit_label(
        self,
        *,
        label: str,
        qubit_label: str | None,
    ) -> str:
        """Resolve qubit label for custom-target registration."""
        target_registry = self.experiment_system.target_registry

        if qubit_label is not None:
            try:
                return target_registry.resolve_qubit_label(qubit_label)
            except ValueError:
                pass
            try:
                return self.experiment_system.get_qubit(qubit_label).label
            except Exception:
                raise ValueError(
                    f"Unknown qubit label `{qubit_label}` for custom target registration."
                ) from None

        try:
            return target_registry.resolve_qubit_label(label)
        except ValueError:
            raise ValueError(
                "Qubit label could not be resolved from "
                f"`{label}`. Pass `qubit_label` explicitly."
            ) from None

    def _resolve_custom_target_object(
        self,
        *,
        qubit_label: str,
        target_type: TargetType,
    ) -> Qubit | Resonator | Mux:
        """Resolve physical object bound to custom target type."""
        qubit = self.experiment_system.get_qubit(qubit_label)
        if target_type in (
            TargetType.CTRL_GE,
            TargetType.CTRL_EF,
            TargetType.CTRL_CR,
            TargetType.UNKNOWN,
        ):
            return qubit
        if target_type == TargetType.READ:
            return self.experiment_system.get_resonator(qubit.resonator)
        if target_type == TargetType.PUMP:
            return self.experiment_system.get_mux_by_qubit(qubit_label)
        raise ValueError(f"Unsupported target type `{target_type}`.")

    def register_custom_target(
        self,
        *,
        label: str,
        frequency: float,
        box_id: str,
        port_number: int,
        channel_number: int,
        qubit_label: str | None = None,
        target_type: TargetType | None = None,
        update_lsi: bool | None = None,
    ) -> None:
        """Register a custom target with the control system."""
        if target_type is None:
            target_type = TargetType.CTRL_GE
        if update_lsi is None:
            update_lsi = False
        if not np.isfinite(frequency):
            raise ValueError("Target frequency must be finite.")

        resolved_qubit_label = self._resolve_custom_target_qubit_label(
            label=label,
            qubit_label=qubit_label,
        )
        port = self.control_system.get_port(box_id, port_number)
        if not isinstance(port, GenPort):
            raise TypeError(
                f"Custom target registration port `{port.id}` must be a GenPort."
            )
        if channel_number < 0 or channel_number >= len(port.channels):
            raise IndexError(
                f"Channel number `{channel_number}` is out of range for port `{port.id}`."
            )
        channel = port.channels[channel_number]
        target_object = self._resolve_custom_target_object(
            qubit_label=resolved_qubit_label,
            target_type=target_type,
        )
        target = Target.new_target(
            label=label,
            frequency=frequency,
            object=target_object,
            channel=channel,
            type=target_type,
        )
        define_target = getattr(self.backend_controller, "define_target", None)
        if not callable(define_target):
            raise NotImplementedError(
                "Active backend does not support custom target registration."
            )
        define_target(
            target_name=target.label,
            channel_name=target.channel.id,
            target_frequency_ghz=target.frequency,
        )
        self.experiment_system.add_target(target)
        if update_lsi:
            cnco = port.cnco_freq
            if cnco is None:
                raise ValueError("CNCO frequency is not set for the target port.")
            fnco, _ = MixingUtil.calc_fnco(
                f=frequency * 1e9,
                ssb=port.sideband,
                lo=port.lo_freq,
                cnco=cnco,
            )
            channel.fnco_freq = fnco
            self.system_manager.push(box_ids=[box_id])

    @contextmanager
    def modified_frequencies(
        self,
        frequencies: dict[str, float] | None,
    ) -> Iterator[None]:
        """Temporarily override target frequencies within the context."""
        if frequencies is None:
            yield
        else:
            with self.system_manager.modified_frequencies(frequencies):
                yield

    def save_calib_note(
        self,
        file_path: Path | str | None = None,
    ) -> None:
        """Save the calibration note to disk."""
        self.calib_note.save(file_path=file_path)

    @deprecated("Use `calib_note.save()` instead.")
    def save_defaults(self) -> None:
        """Save default calibration notes (deprecated)."""
        self._system_note.save()

    @deprecated("Use `calib_note.clear()` instead.")
    def clear_defaults(self) -> None:
        """Clear default calibration notes (deprecated)."""
        self._system_note.clear()

    @deprecated("")
    def delete_defaults(self) -> None:
        """Delete default calibration notes after confirmation."""
        if Confirm.ask("Delete the default params?"):
            self._system_note.clear()
            self._system_note.save()

    def load_record(
        self,
        name: str,
    ) -> ExperimentRecord:
        """Load an experiment record by name and log its metadata."""
        record = ExperimentRecord.load(name)
        logger.info(f"ExperimentRecord `{name}` is loaded.\n")
        logger.info(f"description: {record.description}")
        logger.info(f"created_at: {record.created_at}")
        return record

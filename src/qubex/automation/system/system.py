from typing import Final
from ...backend import (
    StateManager,
)
from ...measurement.measurement import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_PARAMS_DIR,

)
from typing import Collection
from ...experiment.experiment import (
    SYSTEM_NOTE_PATH,
    STATE_CENTERS,
)
from ...typing import TargetMap
from ...experiment.experiment_note import ExperimentNote
from ...measurement import (
    Measurement,
    # MeasureResult,
    StateClassifier,
    # StateClassifierGMM,
    # StateClassifierKMeans,
)
class System:
    def __init__(
        self,
        chip_id: str,
        muxes: Collection[str | int] | None = None,
        qubits: Collection[str | int] | None = None,
        exclude_qubits: Collection[str | int] | None = None,
        config_dir: str = DEFAULT_CONFIG_DIR,
        params_dir: str = DEFAULT_PARAMS_DIR,
        fetch_device_state: bool = True,
        use_neopulse: bool = False,
        connect_devices: bool = True,
        ) -> None:
        self._system_note: Final[ExperimentNote] = ExperimentNote(
            file_path=SYSTEM_NOTE_PATH,
        )
        qubits = self._create_qubit_labels(
            chip_id="QUBEX",
            muxes=None,
            qubits=qubits,
            exclude_qubits=None,
            config_dir="config",
            params_dir="params",
        )
        self._chip_id: Final = chip_id
        self._qubits: Final = qubits
        self._config_dir: Final = config_dir
        self._params_dir: Final = params_dir
        self._measurement = Measurement(
            chip_id=chip_id,
            qubits=qubits,
            config_dir=self._config_dir,
            params_dir=self._params_dir,
            fetch_device_state=fetch_device_state,
            use_neopulse=use_neopulse,
            connect_devices=connect_devices,
        )
    def _create_qubit_labels(
        self,
        chip_id: str,
        muxes: Collection[str | int] | None,
        qubits: Collection[str | int] | None,
        exclude_qubits: Collection[str | int] | None,
        config_dir: str,
        params_dir: str,
    ) -> list[str]:
        state_manager = StateManager.shared()
        state_manager.load(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
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
    @property
    def qubit_labels(self) -> list[str]:
        """Get the list of qubit labels."""
        return self._qubits

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        """Get the classifiers."""
        return self._measurement.classifiers



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

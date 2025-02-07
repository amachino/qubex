import numpy as np
from qiskit.qasm3 import loads
from ..experiment import Experiment
from ..pulse import PulseSchedule, VirtualZ
from ..measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from .base import BaseBackend

class PhysicalBackend(BaseBackend):
    def __init__(self, virtual_physical_map: dict):
        """
        Backend for QASM 3 circuit.
        """
        self._virtual_physical_map = {
            "qubits": {k: f"Q{v:02}" for k, v in virtual_physical_map["qubits"].items()},
            "couplings": {k: (f"Q{v[0]:02}", f"Q{v[1]:02}") for k, v in virtual_physical_map["couplings"].items()},
        }
        self.experiment = Experiment(chip_id="64Q", qubits=self.qubits)
        if self.experiment.state_centers is None:
            self.experiment.build_classifier(plot=False)

    @property
    def program(self) -> str:
        """
        Returns the QASM 3 program.
        """
        return self._program

    @property
    def qubits(self) -> list:
        """
        Returns a list of qubit labels, e.g., ["Q05", "Q07"]
        """
        return list(self._virtual_physical_map["qubits"].values()) # type: ignore

    @property
    def couplings(self) -> list:
        """
        Returns a list of couplings in the format "QXX-QYY", e.g., ["Q05-Q07", "Q07-Q05"]
        """
        return [f"{v[0]}-{v[1]}" for v in self._virtual_physical_map["couplings"].values()] # type: ignore

    @property
    def virtual_physical_qubits(self) -> dict:
        """
        Returns the virtual-to-physical mapping, e.g., {0: "Q05", 1: "Q07"}
        """
        # Return a shallow copy to avoid accidental modifications
        return self._virtual_physical_map["qubits"].copy() # type: ignore

    @property
    def physical_virtual_qubits(self) -> dict:
        """
        Returns the physical-to-virtual mapping, e.g., {"Q05": 0, "Q07": 1}
        """
        return {v: k for k, v in self.virtual_physical_qubits.items()}

    def load_program(self, program: str):
        self._program = program

    def cnot(self, control: str, target: str):
        """Apply CNOT gate"""
        if control not in self.qubits or target not in self.qubits:
            raise ValueError(f"Invalid qubits for CNOT: {control}, {target}")

        cr_label = f"{control}-{target}"
        zx90_pulse = self.experiment.zx90(control, target)

        x90_pulse = (
            self.experiment.drag_hpi_pulse.get(target, self.experiment.hpi_pulse[target])
        )

        with PulseSchedule([control, cr_label, target]) as ps:
            ps.call(zx90_pulse)
            ps.add(control, VirtualZ(-np.pi / 2))
            ps.add(target, x90_pulse.scaled(-1))

        self.circuit.call(ps)

    def x90(self, target: str):
        """Apply X90 gate"""
        if target not in self.qubits:
            raise ValueError(f"Invalid qubit: {target}")

        x90_pulse = (
            self.experiment.drag_hpi_pulse.get(target, self.experiment.hpi_pulse[target])
        )

        with PulseSchedule([target]) as ps:
            ps.add(target, x90_pulse)

        self.circuit.call(ps)

    def x180(self, target: str):
        """Apply X180 gate"""
        if target not in self.qubits:
            raise ValueError(f"Invalid qubit: {target}")

        x180_pulse = (
            self.experiment.drag_pi_pulse.get(target, self.experiment.pi_pulse[target])
        )

        with PulseSchedule([target]) as ps:
            ps.add(target, x180_pulse)

        self.circuit.call(ps)

    def rz(self, target: str, angle: float):
        """Apply Rz gate"""
        if target not in self.qubits:
            raise ValueError(f"Invalid qubit: {target}")

        with PulseSchedule([target]) as ps:
            ps.add(target, VirtualZ(angle))

        self.circuit.call(ps)

    def compile(self):
        """Load QASM 3 program into the pulse schedule"""
        qiskit_circuit = loads(self.program)
        self.circuit = PulseSchedule(self.qubits + self.couplings)

        for instruction in qiskit_circuit.data:
            name = instruction.name
            virtual_index = instruction.qubits[0]._index
            physical_index = self.virtual_physical_qubits[virtual_index]

            if name == "sx":
                self.x90(physical_index)
            elif name == "x":
                self.x180(physical_index)
            elif name == "rz":
                angle = instruction.params[0]
                self.rz(physical_index, angle)
            elif name == "cx":
                virtual_target_index = instruction.qubits[1]._index
                physical_target_index = self.virtual_physical_qubits[virtual_target_index]
                self.cnot(physical_index, physical_target_index)
            elif name == "measure":
                # Ignore measurement (handled in `execute`)
                pass
            else:
                raise ValueError(f"Unsupported instruction: {name}")

    def get_circuit(self) -> PulseSchedule:
        """Get the constructed pulse schedule"""
        return self.circuit

    def execute(self, shots: int = DEFAULT_SHOTS) -> dict:
        """Run the quantum circuit with specified shots"""
        self.experiment.build_classifier(plot=False)
        schedule = self.get_circuit()

        initial_states = {qubit: "0" for qubit in self.qubits}

        return self.experiment.measure(
            schedule, initial_states=initial_states, mode="single", shots=shots, interval=DEFAULT_INTERVAL
        ).counts




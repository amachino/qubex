from qiskit.qasm3 import loads
from ..measurement.measurement import DEFAULT_SHOTS
from qulacs import QuantumCircuit, QuantumState
from .base import BaseBackend
from collections import Counter



class SimulatorBackend(BaseBackend):
    def __init__(self, virtual_physical_map: dict):
        """
        Backend for QASM 3 circuit.
        """
        self._virtual_physical_map = virtual_physical_map
        self.circuit = QuantumCircuit(len(self.qubits))

    @property
    def qubits(self) -> list:
        """
        # Returns
                List of couplings,
        eg. ["Q05", "Q07"]
        """
        qubits = []
        for _, v in self._virtual_physical_map["qubits"].items():
            qubits.append(f"Q{v:02}")
        return qubits
    
    @property
    def couplings(self) -> list:
        """
        # Returns
                List of couplings,
        eg. ["Q05-Q07", "Q07-Q05"]
        """
        couplings = []
        for _, v in self._virtual_physical_map["couplings"].items():
            couplings.append(f"Q{v[0]:02}-Q{v[1]:02}")
        return couplings


    @property
    def virtual_physical_qubits(self) -> dict:
        """
        # Returns
                Virtual to Physical mapping,
        eg. {0: "Q05", 1: "Q07"}
        """
        for k, v in self._virtual_physical_map["qubits"].items():
            self._virtual_physical_map["qubits"][k] = f"Q{v:02}"
        return self._virtual_physical_map["qubits"]

    @property
    def physical_virtual_qubits(self) -> dict:
        """
        # Returns
                Physical to Virtual mapping,
        eg. {"Q05": 0, "Q07": 1}
        """
        return {v: k for k, v in self.virtual_physical_qubits.items()}

    def cnot(self, control: str, target: str):
        """Apply CNOT gate"""
        if control not in self.qubits or target not in self.qubits:
            raise ValueError(f"Invalid qubits for CNOT: {control}, {target}")
        self.circuit.add_CNOT_gate(
            self.physical_virtual_qubits[control],
            self.physical_virtual_qubits[target]
        )

    def x90(self, target: str):
        """Apply X90 gate"""
        if target not in self.qubits:
            raise ValueError(f"Invalid qubit: {target}")
        self.circuit.add_sqrtX_gate(self.physical_virtual_qubits[target])

    def x180(self, target: str):
        """Apply X180 gate"""
        if target not in self.qubits:
            raise ValueError(f"Invalid qubit: {target}")
        self.circuit.add_X_gate(self.physical_virtual_qubits[target])

    def rz(self, target: str, angle: float):
        """Apply RZ gate"""
        if target not in self.qubits:
            raise ValueError(f"Invalid qubit: {target}")
        self.circuit.add_RZ_gate(self.physical_virtual_qubits[target], angle)

    def load_program(self, program: str):
        """Load QASM 3 program into the pulse schedule test"""
        qiskit_circuit = loads(program)

        for instruction in qiskit_circuit.data:
            name = instruction.name
            virtual_index = instruction.qubits[0]._index
            physical_label = self.virtual_physical_qubits[virtual_index]

            if name == "sx":
                self.x90(physical_label)
            elif name == "x":
                self.x180(physical_label)
            elif name == "rz":
                angle = instruction.params[0]
                self.rz(physical_label, angle)
            elif name == "cx":
                virtual_target_index = instruction.qubits[1]._index
                physical_target_label = self.virtual_physical_qubits[virtual_target_index]
                self.cnot(physical_label, physical_target_label)
            elif name == "measure":
                pass
            else:
                raise ValueError(f"Unsupported instruction: {name}")



    @property
    def get_circuit(self) -> QuantumCircuit:
        """Get the constructed quantum circuit"""
        return self.circuit

    def execute(self, shots: int = DEFAULT_SHOTS) -> dict:
        """
        Run the quantum circuit with specified shots
        """
        state = QuantumState(self.circuit.get_qubit_count())
        self.circuit.update_quantum_state(state)
        result  = Counter(state.sampling(shots))
        counts = dict()
        for key, value in result.items():
            counts[format(key, "0" + str(self.circuit.get_qubit_count()) + "b")] = value 
        return counts
    
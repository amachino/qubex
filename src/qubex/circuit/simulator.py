from qiskit.qasm3 import loads
from ..measurement.measurement import DEFAULT_SHOTS
from qulacs import QuantumCircuit, QuantumState
from .base import BaseBackend
from collections import Counter

class SimulatorBackend(BaseBackend):
    def __init__(self, chip_id: str, nodes: list[str], edges: list[str]):
        """
        Backend for QASM 3 circuit.
        """
        self.nodes = nodes  # e.g. ["Q05", "Q07"]
        self.edges = edges  # e.g. ["Q05-Q07"]
        self.circuit = QuantumCircuit(len(nodes))

        # Virtual to Physical mapping
        self.virtual_to_physical = {i: nodes[i] for i in range(len(nodes))}
        self.physical_to_virtual = {node: i for i, node in enumerate(nodes)}

    def cnot(self, control: str, target: str):
        """Apply CNOT gate"""
        if control not in self.nodes or target not in self.nodes:
            raise ValueError(f"Invalid qubits for CNOT: {control}, {target}")
        self.circuit.add_CNOT_gate(
            self.physical_to_virtual[control],
            self.physical_to_virtual[target]
        )

    def x90(self, target: str):
        """Apply X90 gate"""
        if target not in self.nodes:
            raise ValueError(f"Invalid qubit: {target}")
        self.circuit.add_sqrtX_gate(self.physical_to_virtual[target])

    def x180(self, target: str):
        """Apply X180 gate"""
        if target not in self.nodes:
            raise ValueError(f"Invalid qubit: {target}")
        self.circuit.add_X_gate(self.physical_to_virtual[target])

    def rz(self, target: str, angle: float):
        """Apply RZ gate"""
        if target not in self.nodes:
            raise ValueError(f"Invalid qubit: {target}")
        self.circuit.add_RZ_gate(self.physical_to_virtual[target], angle)

    def load_program(self, program: str):
        """Load QASM 3 program into the pulse schedule"""
        qiskit_circuit = loads(program)

        for instruction in qiskit_circuit.data:
            name = instruction.name
            # 最初のキュービットから物理ラベルを取得
            virtual_index = instruction.qubits[0]._index
            physical_label = self.virtual_to_physical[virtual_index]

            if name == "sx":
                self.x90(physical_label)
            elif name == "x":
                self.x180(physical_label)
            elif name == "rz":
                angle = instruction.params[0]
                self.rz(physical_label, angle)
            elif name == "cx":
                virtual_target_index = instruction.qubits[1]._index
                physical_target_label = self.virtual_to_physical[virtual_target_index]
                self.cnot(physical_label, physical_target_label)
            elif name == "measure":
                # 測定は execute 時に行うため、ここでは無視します。
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
    
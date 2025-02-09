from qiskit.qasm3 import loads
from ..measurement.measurement import DEFAULT_SHOTS
from qulacs import QuantumCircuit, QuantumState
from .base import BaseBackend
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class QulacsBackend(BaseBackend):
    def __init__(self, virtual_physical_map: dict, job_id: str="test_job"):
        """
        Backend for QASM 3 circuits.
        """
        super().__init__(virtual_physical_map, job_id)
        self.job_id = job_id
        self._virtual_physical_map = virtual_physical_map
    
    @property
    def program(self) -> str:
        return super().program
    
    @property
    def qubits(self) -> list:
        return super().qubits
    
    @property
    def couplings(self) -> list:
        return super().couplings
    
    @property
    def virtual_physical_qubits(self) -> dict:
        return super().virtual_physical_qubits
    
    @property
    def physical_virtual_qubits(self) -> dict:
        return super().physical_virtual_qubits
    
    @property
    def result(self) -> dict:
        return super().result
    
    @property
    def get_circuit(self) -> QuantumCircuit:
        """Return the constructed quantum circuit."""
        return self.circuit
    
    def plot_histogram(self):
        return super().plot_histogram()
    
    def physical_qubit(self, virtual_qubit):
        return super().physical_qubit(virtual_qubit)
    
    def virtual_qubit(self, physical_qubit):
        return super().virtual_qubit(physical_qubit)

    def load_program(self, program: str):
        super().load_program(program)

    def cnot(self, control: str, target: str):
        """Apply CNOT gate."""
        super().cnot(control, target)
        self.circuit.add_CNOT_gate(
            self.physical_virtual_qubits[control],
            self.physical_virtual_qubits[target]
        )

    def x90(self, target: str):
        """Apply X90 gate."""
        super().x90(target)
        self.circuit.add_sqrtX_gate(self.physical_virtual_qubits[target])

    def x180(self, target: str):
        """Apply X180 gate."""
        super().x180(target)
        self.circuit.add_X_gate(self.physical_virtual_qubits[target])

    def rz(self, target: str, angle: float):
        """Apply RZ gate."""
        super().rz(target, angle)
        self.circuit.add_RZ_gate(self.physical_virtual_qubits[target], angle)

    def compile(self):
        """Load a QASM 3 program and apply the corresponding gates to the circuit."""
        logger.info(f"QASM 3 program: {self.program}, job_id={self.job_id}")
        qiskit_circuit = loads(self.program)
        self.circuit = QuantumCircuit(qiskit_circuit.num_qubits)

        for instruction in qiskit_circuit.data:
            name = instruction.name
            virtual_index =  qiskit_circuit.find_bit(instruction.qubits[0]).index
            physical_label = self.physical_qubit(virtual_index)

            if name == "sx":
                self.x90(physical_label)
            elif name == "x":
                self.x180(physical_label)
            elif name == "rz":
                angle = instruction.params[0]
                self.rz(physical_label, angle)
            elif name == "cx":
                virtual_target_index =  qiskit_circuit.find_bit(instruction.qubits[1]).index
                physical_target_label = self.virtual_physical_qubits[virtual_target_index]
                self.cnot(physical_label, physical_target_label)
            elif name == "measure":
                pass
            else:
                logger.error(f"Unsupported instruction: {name}, job_id={self.job_id}")
                raise ValueError(f"Unsupported instruction: {name}")
        logger.info(f"Compilation complete, job_id={self.job_id}")

    def execute(self, shots: int = DEFAULT_SHOTS) -> dict:
        """
        Execute the quantum circuit with a specified number of shots.
        """
        logger.info(f"Executing quantum circuit with {shots} shots")
        state = QuantumState(self.circuit.get_qubit_count())
        self.circuit.update_quantum_state(state)
        result = Counter(state.sampling(shots))
        counts = {}
        for key, value in result.items():
            counts[format(key, "0" + str(self.circuit.get_qubit_count()) + "b")] = value 
        logger.info(f"Execution complete, counts: {counts}, job_id={self.job_id}")
        self._result = counts
        return counts

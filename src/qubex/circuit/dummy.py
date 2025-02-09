from ..measurement.measurement import DEFAULT_SHOTS
from .base import BaseBackend
import logging

logger = logging.getLogger(__name__)

class DummyBackend(BaseBackend):
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
        pass

    def x90(self, target: str):
        """Apply X90 gate."""
        super().x90(target)
        pass

    def x180(self, target: str):
        """Apply X180 gate."""
        super().x180(target)
        pass

    def rz(self, target: str, angle: float):
        """Apply RZ gate."""
        super().rz(target, angle)
        pass

    def compile(self):
        """Load a QASM 3 program and apply the corresponding gates to the circuit."""
        pass


    def execute(self, shots: int = DEFAULT_SHOTS) -> dict:
        """
        Run the quantum circuit with specified shots
        """
        counts = {
            "00":455,
            "01":0,
            "10":0,
            "11":545
        }
        self._result = counts
        return counts
    
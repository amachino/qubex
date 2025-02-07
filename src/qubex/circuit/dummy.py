from ..measurement.measurement import DEFAULT_SHOTS
from .base import BaseBackend
import logging

logger = logging.getLogger(__name__)

class DummyBackend(BaseBackend):
    def __init__(self, virtual_physical_map: dict):
        """
        Dummy backend for QASM 3 circuit.
        """
        self._virtual_physical_map = {
            "qubits": {k: f"Q{v:02}" for k, v in virtual_physical_map["qubits"].items()},
            "couplings": {k: (f"Q{v[0]:02}", f"Q{v[1]:02}") for k, v in virtual_physical_map["couplings"].items()},
        }

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
        pass
        

    def x90(self, target: str):
        """Apply X90 gate"""
        pass

    def x180(self, target: str):
        """Apply X180 gate"""
        pass

    def rz(self, target: str, angle: float):
        """Apply RZ gate"""
        pass
    
    def compile(self):
        """
        Compile the quantum circuit
        """
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
        return counts
    
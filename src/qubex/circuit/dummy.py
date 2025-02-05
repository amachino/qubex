from ..measurement.measurement import DEFAULT_SHOTS
from .base import BaseBackend


class DummyBackend(BaseBackend):
    def __init__(self, chip_id: str, nodes: list[str], edges: list[str]):
        """
        Dummy backend for QASM 3 circuit.
        """
        self.nodes = nodes  # e.g. ["Q05", "Q07"]
        self.edges = edges  # e.g. ["Q05-Q07"]

        # Virtual to Physical mapping
        self.virtual_to_physical = {i: nodes[i] for i in range(len(nodes))}
        self.physical_to_virtual = {node: i for i, node in enumerate(nodes)}

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

    def load_program(self, program: str):
        """Load QASM 3 program into the pulse schedule"""
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
    
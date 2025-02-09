from abc import ABCMeta, abstractmethod

import logging

logger = logging.getLogger(__name__)
class BaseBackend(metaclass=ABCMeta):

    def __init__(self, virtual_physical_map: dict, job_id: str="test_job"):
        """
        Backend for QASM 3 circuits.
        """
        self.job_id = job_id
        self._virtual_physical_map = {
            "qubits": {k: f"Q{v:02}" for k, v in virtual_physical_map["qubits"].items()},
            "couplings": {k: (f"Q{v[0]:02}", f"Q{v[1]:02}") for k, v in virtual_physical_map["couplings"].items()},
        }
    
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
    
    @property
    def result(self) -> dict:
        """
        Returns the result of the execution.
        """
        return self._result
    
    def plot_histogram(self):
        """
        Plot the histogram of the execution result.
        """
        from qiskit.visualization import plot_histogram
        return plot_histogram(self.result)
        
    def physical_qubit(self, virtual_qubit: str) -> str:
        """
        Returns the physical qubit corresponding to the virtual qubit.
        """
        return self.virtual_physical_qubits[virtual_qubit]
    
    def virtual_qubit(self, physical_qubit: str) -> int:
        """
        Returns the virtual qubit corresponding to the physical qubit.
        """
        return self.physical_virtual_qubits[physical_qubit]

    def load_program(self, program: str):
        logger.info(f"Loading QASM 3 program: {program}, job_id={self.job_id}")
        self._program = program

    @abstractmethod
    def cnot(self, control: str, target: str):
        """
		Apply CNOT gate
		"""
        if control not in self.qubits or target not in self.qubits:
            logger.error(f"Invalid qubits for CNOT: {control}, {target}, job_id={self.job_id}")
            raise ValueError(f"Invalid qubits for CNOT: {control}, {target}")
        logger.debug(f"Applying CNOT gate: {self.virtual_qubit(control)} -> {self.virtual_qubit(target)}, Physical qubits: {control} -> {target}, job_id={self.job_id}")

    @abstractmethod
    def x90(self, target: str):
        """ Apply X90 gate """
        if target not in self.qubits:
            logger.error(f"Invalid qubit: {target}, job_id={self.job_id}")
            raise ValueError(f"Invalid qubit: {target}")
        logger.debug(f"Applying X90 gate: {self.virtual_qubit(target)}, Physical qubit: {target}, job_id={self.job_id}")

    @abstractmethod 
    def x180(self, target: str):
        """ Apply X180 gate """
        if target not in self.qubits:
            logger.error(f"Invalid qubit: {target}, job_id={self.job_id}")
            raise ValueError(f"Invalid qubit: {target}")
        logger.debug(f"Applying X180 gate: {self.virtual_qubit(target)}, Physical qubit: {target}, job_id={self.job_id}")

    @abstractmethod
    def rz(self,  target: str, angle: float):
        """ Apply RZ gate """
        if target not in self.qubits:
            logger.error(f"Invalid qubit: {target}, job_id={self.job_id}")
            raise ValueError(f"Invalid qubit: {target}")
        logger.debug(f"Applying RZ gate: {self.virtual_qubit(target)}, Physical qubit: {target}, angle={angle}, job_id={self.job_id}")
    
    @abstractmethod
    def compile(self):
        """Load a QASM 3 program and apply the corresponding gates to the circuit."""
        raise NotImplementedError("This method is not implemented")
    
    @abstractmethod
    def execute(self, shots: int) -> dict:
        """Run the quantum circuit with specified shots"""
        self._result :dict= {}
        raise NotImplementedError("This method is not implemented")

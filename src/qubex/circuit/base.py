from abc import ABCMeta, abstractmethod


class BaseBackend(metaclass=ABCMeta):

    def __init__(
        self,
    ):
        pass
    
    @abstractmethod
    def cnot(self, control: str, target: str):
        """
		Apply CNOT gate
		"""
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def x90(self, target: str):
        """ Apply X90 gate """
        raise NotImplementedError("This method is not implemented")

    @abstractmethod 
    def x180(self, target: str):
        """ Apply X180 gate """
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def rz(self,  target: str, angle: float):
        """ Apply RZ gate """
        raise NotImplementedError("This method is not implemented")

    @abstractmethod
    def load_program(self, program: str):
        """ Load QASM 3 program into the pulse schedule """
        raise NotImplementedError("This method is not implemented")
    

    @abstractmethod
    def execute(self, shots: int) -> dict:
        """Run the quantum circuit with specified shots"""
        raise NotImplementedError("This method is not implemented")

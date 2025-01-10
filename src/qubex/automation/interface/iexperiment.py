import abc
from ..system.system import System
from ...experiment.experiment_result import ExperimentResult
class TDM:
    def __init__(self, ):
        pass
    pass



class IExperiment(metaclass=abc.ABCMeta):
    experiment_name: str
    input_parameters: list[str]
    output_parameters: list[str]

    def __init__(self, **kwargs: object):
        pass

    @abc.abstractmethod
    def take_data(self,system:System) -> object:
        return object

    @abc.abstractmethod
    def analyze(self, system:System, result: ExperimentResult):
        pass

# pylint: disable=all


def test_init():
    from qubex.qube_manager import QubeManager
    from qubex.configs import Configs

    configs = Configs.load("qubex/configs/example.json")
    qube_manager = QubeManager(configs)

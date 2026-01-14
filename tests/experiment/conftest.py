from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qubex.backend import (
    ControlSystem,
    DeviceController,
    ExperimentSystem,
    QuantumSystem,
    Qubit,
    SystemManager,
    Target,
)
from qubex.experiment.experiment import Experiment
from qubex.experiment.experiment_context import ExperimentContext


@pytest.fixture
def mock_system_manager():
    """
    Patches SystemManager.shared() to return a MagicMock.
    Configures basic attributes to avoid AttributeErrors during ExperimentContext init.
    """
    with patch("qubex.backend.SystemManager.shared") as mock_shared:
        manager = MagicMock(spec=SystemManager)
        mock_shared.return_value = manager

        # Setup sub-components
        manager.config_loader = MagicMock()
        manager.experiment_system = MagicMock(spec=ExperimentSystem)
        manager.device_controller = MagicMock(spec=DeviceController)
        manager.control_system = MagicMock(spec=ControlSystem)

        # Setup QuantumSystem
        qs = MagicMock(spec=QuantumSystem)
        manager.experiment_system.quantum_system = qs

        # Setup Qubit "Q0"
        q0 = MagicMock(spec=Qubit)
        q0.label = "Q0"
        qs.get_qubit.return_value = q0

        # Setup targets
        target0 = MagicMock(spec=Target)
        target0.qubit = "Q0"
        target0.label = "Q0"

        target_rq0 = MagicMock(spec=Target)
        target_rq0.qubit = "Q0"
        target_rq0.label = "RQ0"
        target_rq0.frequency = 6.0
        target_rq0.sideband = "LSB"

        manager.experiment_system.ge_targets = [target0]
        manager.experiment_system.qubits = [q0]
        manager.experiment_system.targets = [target0, target_rq0]
        manager.experiment_system.muxes = {}

        # Mock load methods
        manager.load.return_value = None
        manager.load_skew_file.return_value = None

        yield manager


@pytest.fixture
def experiment_context(mock_system_manager, tmp_path):
    """
    Creates an ExperimentContext instance using the mock_system_manager.
    """
    # Create fake directories so pathlib checks don't fail if they happen before SystemManager calls
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    params_dir = tmp_path / "params"
    params_dir.mkdir()

    # Need to ensure _create_qubit_labels works
    # We'll rely on the default mocking of quantum_system.get_qubit above
    # But usually creating context with no qubits specified is safer for generic tests

    ctx = ExperimentContext(
        chip_id="test_chip",
        config_dir=config_dir,
        params_dir=params_dir,
        mock_mode=True,
    )
    return ctx


@pytest.fixture
def experiment(mock_system_manager, tmp_path):
    """
    Creates an Experiment instance using the mock_system_manager.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    params_dir = tmp_path / "params"
    params_dir.mkdir(exist_ok=True)

    ex = Experiment(
        chip_id="test_chip",
        config_dir=config_dir,
        params_dir=params_dir,
        qubits=["Q0"],  # This will trigger quantum_system.get_qubit("Q0")
    )
    return ex

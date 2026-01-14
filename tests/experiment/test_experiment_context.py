from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qubex.backend import Target
from qubex.experiment.experiment_context import ExperimentContext


def test_init_context(mock_system_manager, tmp_path):
    """ExperimentContext should initialize and load config via SystemManager."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)

    ctx = ExperimentContext(chip_id="test_chip", config_dir=config_dir, mock_mode=True)

    assert ctx.chip_id == "test_chip"
    mock_system_manager.load.assert_called_once()
    mock_system_manager.load_skew_file.assert_called_once()


def test_qubit_labels_filtering(mock_system_manager, tmp_path):
    """ExperimentContext should filter qubit labels based on available targets."""
    # Setup available targets in the system
    target_q0 = MagicMock(spec=Target)
    target_q0.qubit = "Q0"
    target_q1 = MagicMock(spec=Target)
    target_q1.qubit = "Q1"

    # Only Q0 and Q1 are available in the system
    mock_system_manager.experiment_system.ge_targets = [target_q0, target_q1]

    # Q0's object mock
    q0_obj = MagicMock()
    q0_obj.label = "Q0"

    # Q2's object mock (valid in quantum system, but not in experiment targets?)
    q2_obj = MagicMock()
    q2_obj.label = "Q2"

    # Configure get_qubit to return labeled objects
    def get_qubit_side_effect(q):
        if q == "Q0":
            return q0_obj
        if q == "Q2":
            return q2_obj
        return MagicMock(label=str(q))

    mock_system_manager.experiment_system.quantum_system.get_qubit.side_effect = (
        get_qubit_side_effect
    )

    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)

    # Request Q0 and Q2
    ctx = ExperimentContext(
        chip_id="test_chip", qubits=["Q0", "Q2"], config_dir=config_dir, mock_mode=True
    )

    # Q2 should be filtered out because it's not in ge_targets
    assert "Q0" in ctx.qubit_labels
    assert "Q2" not in ctx.qubit_labels


def test_services_access(experiment_context):
    """ExperimentContext should provide access to services."""
    # Just checking they don't crash and return expected types/modules
    from qubex.experiment import experiment_tool

    assert experiment_context.tool == experiment_tool

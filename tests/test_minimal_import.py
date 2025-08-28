"""
Test for minimal import scenario - importing Experiment without backend extras.
"""
import pytest


def test_import_experiment_without_backend():
    """Test that Experiment can be imported without backend extras installed."""
    # This test should pass even when qubecalib, quel_ic_config, etc. are not installed
    from qubex import Experiment
    assert Experiment is not None


def test_construct_experiment_without_backend():
    """Test that Experiment can be constructed without backend functionality."""
    from qubex import Experiment
    
    # Should work without eager backend loading
    exp = Experiment(chip_id='test_chip', qubits=['Q0', 'Q1'])
    assert exp is not None
    assert exp.chip_id == 'test_chip'


def test_backend_unavailable_error_on_backend_access():
    """Test that accessing backend-dependent properties raises appropriate errors."""
    from qubex import Experiment
    from qubex.experiment.experiment_exceptions import BackendUnavailableError
    
    exp = Experiment(chip_id='test_chip', qubits=['Q0', 'Q1'])
    
    # Accessing backend-dependent properties should raise errors 
    # (either BackendUnavailableError or file not found errors)
    with pytest.raises((BackendUnavailableError, FileNotFoundError)):
        _ = exp.qubits  # This requires backend config loading


def test_backend_unavailable_error_message():
    """Test that BackendUnavailableError provides helpful installation guidance."""
    from qubex.experiment.experiment_exceptions import BackendUnavailableError
    
    error = BackendUnavailableError("Test message")
    error_str = str(error)
    
    assert "Test message" in error_str
    assert "pip install" in error_str
    assert "qubex[backend]" in error_str


def test_lazy_backend_loading():
    """Test that backend components use lazy loading."""
    from qubex.backend.device_controller import _ensure_qubecalib
    from qubex.experiment.experiment_exceptions import BackendUnavailableError
    
    # Should raise BackendUnavailableError when qubecalib is not available
    with pytest.raises(BackendUnavailableError):
        _ensure_qubecalib()
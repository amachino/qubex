from __future__ import annotations

from unittest.mock import MagicMock


def test_init_experiment(experiment):
    """Experiment should initialize correctly using fixture."""
    assert experiment.chip_id == "test_chip"
    # Q0 was in "qubits" arg, confirming context setup
    assert "Q0" in experiment.qubits


def test_delegation_to_context(experiment):
    """Experiment should delegate facade properties to context."""
    assert experiment.chip_id == "test_chip"
    # Verify accessing a context property
    assert experiment.ctx is not None


def test_measurement_service_delegation(experiment):
    """Experiment should delegate measure calls to MeasurementService."""
    # Patch the internal service with a mock
    mock_service = MagicMock()
    experiment._measurement_service = mock_service

    dummy_schedule = MagicMock()
    experiment.measure(sequence=dummy_schedule, shots=500)

    mock_service.measure.assert_called_once()
    # Check arguments
    call_args = mock_service.measure.call_args
    assert call_args.kwargs["sequence"] == dummy_schedule
    assert call_args.kwargs["shots"] == 500


def test_calibration_service_delegation(experiment):
    """Experiment should delegate calibration calls to CalibrationService."""
    mock_service = MagicMock()
    experiment._calibration_service = mock_service

    # calibrate_pi_pulse is delegated to calibration service
    experiment.calibrate_pi_pulse(targets="Q0", shots=2000)

    mock_service.calibrate_pi_pulse.assert_called_once()
    call_args = mock_service.calibrate_pi_pulse.call_args
    assert call_args.kwargs["targets"] == "Q0"
    assert call_args.kwargs["shots"] == 2000


def test_characterization_service_delegation(experiment):
    """Experiment should delegate characterization calls to CharacterizationService."""
    mock_service = MagicMock()
    experiment._characterization_service = mock_service

    # t1_experiment takes 'targets'
    experiment.t1_experiment(targets="Q0", time_range=(0, 100, 10))

    mock_service.t1_experiment.assert_called_once()

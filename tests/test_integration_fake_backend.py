"""Integration tests with fake backend for testing experiment workflows."""

import pytest
from unittest.mock import Mock
import numpy as np

from qubex.experiment.rabi_param import RabiParam
from qubex.backend.device_controller import RawResult


class FakeDeviceController:
    """A fake device controller for testing experiment workflows."""
    
    def __init__(self):
        """Initialize fake device controller with predictable behavior."""
        self.connected = False
        self.configured = False
        self.call_log = []
        self.fake_measurements = {}
        
    def connect(self):
        """Simulate device connection."""
        self.call_log.append("connect")
        self.connected = True
        return {"success": True}
    
    def configure(self, targets=None):
        """Simulate device configuration."""
        self.call_log.append(f"configure:{targets}")
        self.configured = True
        return {"success": True}
    
    def execute_sequencer(self, sequencer, **kwargs):
        """Simulate sequencer execution with fake measurement data."""
        self.call_log.append("execute_sequencer")
        
        # Generate fake measurement data based on sequencer
        fake_data = self._generate_fake_measurement_data(sequencer)
        
        return RawResult(
            status={"success": True, "shots": kwargs.get("repeats", 1024)},
            data=fake_data,
            config={"sequencer_id": id(sequencer)}
        )
    
    def _generate_fake_measurement_data(self, sequencer):
        """Generate realistic fake measurement data."""
        # Simulate some I/Q measurement data
        shots = 1024
        targets = ["Q00", "Q01"]  # Fake targets
        
        data = {}
        for target in targets:
            # Generate fake Rabi oscillation data
            i_data = np.random.normal(0.5, 0.1, shots)
            q_data = np.random.normal(0.0, 0.1, shots)
            data[target] = i_data + 1j * q_data
            
        return data


class TestExperimentIntegration:
    """Integration tests using fake device controller."""
    
    @pytest.fixture
    def fake_controller(self):
        """Provide a fake device controller."""
        return FakeDeviceController()
    
    @pytest.fixture
    def mock_experiment(self, fake_controller):
        """Create a mock experiment with fake controller."""
        experiment = Mock()
        experiment.device_controller = fake_controller
        experiment.qubit_labels = ["Q00", "Q01"]
        experiment.rabi_params = {}
        experiment.calib_note = Mock()
        experiment.save_calib_note = Mock()
        experiment.obtain_reference_points = Mock(return_value={"phase": {"Q00": 1.0, "Q01": 1.5}})
        return experiment
    
    def test_experiment_workflow_with_fake_backend(self, mock_experiment, fake_controller):
        """Test a complete experiment workflow with fake backend."""
        # Test connection
        assert not fake_controller.connected
        result = fake_controller.connect()
        assert result["success"]
        assert fake_controller.connected
        assert "connect" in fake_controller.call_log
        
        # Test configuration
        fake_controller.configure(targets=["Q00", "Q01"])
        assert fake_controller.configured
        assert "configure:['Q00', 'Q01']" in fake_controller.call_log
        
        # Test measurement execution
        mock_sequencer = Mock()
        result = fake_controller.execute_sequencer(mock_sequencer, repeats=2048)
        
        assert isinstance(result, RawResult)
        assert result.status["success"]
        assert result.status["shots"] == 2048
        assert "Q00" in result.data
        assert "Q01" in result.data
        
        # Verify data structure
        for target, iq_data in result.data.items():
            assert len(iq_data) == 1024  # Default shots
            assert np.iscomplexobj(iq_data)
    
    def test_rabi_calibration_workflow(self, mock_experiment, fake_controller):
        """Test Rabi parameter calibration workflow with fake backend."""
        from qubex.experiment.experiment import Experiment
        
        # Setup fake rabi parameters
        rabi_param_q00 = RabiParam(
            target="Q00", amplitude=0.5, frequency=10.0, phase=1.5, 
            offset=0.1, noise=0.02, angle=2.1, distance=0.8, 
            r2=0.95, reference_phase=0.0
        )
        rabi_param_q01 = RabiParam(
            target="Q01", amplitude=0.6, frequency=11.0, phase=1.6, 
            offset=0.2, noise=0.03, angle=2.2, distance=0.9, 
            r2=0.93, reference_phase=0.1
        )
        
        mock_experiment.rabi_params = {
            "Q00": rabi_param_q00,
            "Q01": rabi_param_q01
        }
        
        # Execute correction workflow
        Experiment.correct_rabi_params(mock_experiment, save=False)
        
        # Verify corrections were applied
        assert rabi_param_q00.reference_phase == 1.0  # From mock obtain_reference_points
        assert rabi_param_q01.reference_phase == 1.5
        
        # Verify calib_note was updated
        assert mock_experiment.calib_note.update_rabi_param.call_count == 2
        
        # Verify no save was called
        mock_experiment.save_calib_note.assert_not_called()
    
    def test_fake_backend_error_simulation(self, fake_controller):
        """Test error simulation with fake backend."""
        # Simulate connection failure
        fake_controller.connected = False
        
        def failing_connect():
            fake_controller.call_log.append("connect_failed")
            raise ConnectionError("Simulated connection failure")
        
        fake_controller.connect = failing_connect
        
        with pytest.raises(ConnectionError, match="Simulated connection failure"):
            fake_controller.connect()
        
        assert "connect_failed" in fake_controller.call_log
        assert not fake_controller.connected
    
    def test_measurement_data_validation(self, fake_controller):
        """Test that fake measurement data meets expected criteria."""
        mock_sequencer = Mock()
        result = fake_controller.execute_sequencer(mock_sequencer)
        
        # Validate data quality
        for target, iq_data in result.data.items():
            # Check data is reasonable (within expected ranges)
            i_parts = np.real(iq_data)
            q_parts = np.imag(iq_data)
            
            # I component should be around 0.5 ± some noise
            assert 0.0 < np.mean(i_parts) < 1.0
            assert np.std(i_parts) > 0  # Should have some noise
            
            # Q component should be around 0 ± some noise  
            assert -0.5 < np.mean(q_parts) < 0.5
            assert np.std(q_parts) > 0  # Should have some noise
    
    def test_fake_backend_state_tracking(self, fake_controller):
        """Test that fake backend correctly tracks state changes."""
        # Initial state
        assert not fake_controller.connected
        assert not fake_controller.configured
        assert len(fake_controller.call_log) == 0
        
        # Connect
        fake_controller.connect()
        assert fake_controller.connected
        assert len(fake_controller.call_log) == 1
        
        # Configure
        fake_controller.configure(["Q00"])
        assert fake_controller.configured
        assert len(fake_controller.call_log) == 2
        
        # Execute multiple measurements
        for i in range(3):
            fake_controller.execute_sequencer(Mock())
        
        assert len(fake_controller.call_log) == 5  # connect + configure + 3 executions
        
        # Verify call order
        expected_calls = ["connect", "configure:['Q00']"] + ["execute_sequencer"] * 3
        assert fake_controller.call_log == expected_calls
    
    def test_integration_with_real_experiment_methods(self, mock_experiment):
        """Test integration of fake backend with real experiment method logic.""" 
        from qubex.experiment.experiment import Experiment
        
        # Test validate_rabi_params with empty params
        mock_experiment.rabi_params = {}
        
        with pytest.raises(ValueError, match="Rabi parameters are not stored"):
            Experiment.validate_rabi_params(mock_experiment)
        
        # Test with valid params
        mock_experiment.rabi_params = {"Q00": Mock(), "Q01": Mock()}
        
        # Should not raise exception
        Experiment.validate_rabi_params(mock_experiment, targets=["Q00"])
    
    def test_reproducible_fake_measurements(self, fake_controller):
        """Test that fake measurements are reproducible with seeding."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        mock_sequencer = Mock()
        result1 = fake_controller.execute_sequencer(mock_sequencer)
        
        # Reset seed and get same result
        np.random.seed(42)
        result2 = fake_controller.execute_sequencer(mock_sequencer)
        
        # Results should be identical
        for target in result1.data:
            np.testing.assert_array_equal(result1.data[target], result2.data[target])


class TestFakeDeviceControllerStandalone:
    """Standalone tests for FakeDeviceController functionality."""
    
    def test_fake_controller_initialization(self):
        """Test FakeDeviceController initializes correctly."""
        controller = FakeDeviceController()
        
        assert not controller.connected
        assert not controller.configured
        assert controller.call_log == []
        assert controller.fake_measurements == {}
    
    def test_fake_controller_api_compatibility(self):
        """Test that FakeDeviceController has expected API."""
        controller = FakeDeviceController()
        
        # Should have required methods
        assert hasattr(controller, "connect")
        assert hasattr(controller, "configure") 
        assert hasattr(controller, "execute_sequencer")
        
        # Methods should be callable
        assert callable(controller.connect)
        assert callable(controller.configure)
        assert callable(controller.execute_sequencer)
    
    def test_fake_controller_data_generation(self):
        """Test fake data generation produces valid measurement data."""
        controller = FakeDeviceController()
        mock_sequencer = Mock()
        
        # Generate multiple datasets
        datasets = []
        for _ in range(5):
            result = controller.execute_sequencer(mock_sequencer)
            datasets.append(result.data)
        
        # All datasets should have same structure
        for dataset in datasets[1:]:
            assert set(dataset.keys()) == set(datasets[0].keys())
            
            for target in dataset:
                assert len(dataset[target]) == len(datasets[0][target])
                assert dataset[target].dtype == datasets[0][target].dtype
        
        # But values should be different (due to randomness)
        different_values = False
        for target in datasets[0]:
            if not np.array_equal(datasets[0][target], datasets[1][target]):
                different_values = True
                break
        
        assert different_values, "Fake data should have some randomness"
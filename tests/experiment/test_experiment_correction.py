"""Tests for Experiment calibration correction methods."""

from unittest.mock import Mock
import pytest

from qubex.experiment.rabi_param import RabiParam


class TestExperimentCorrection:
    """Tests for Experiment calibration correction methods."""

    @pytest.fixture
    def mock_experiment(self):
        """Create a mock experiment with necessary attributes."""
        experiment = Mock()
        experiment.qubit_labels = ["Q00", "Q01", "Q02"]
        experiment.rabi_params = {}
        experiment.calib_note = Mock()
        experiment.obtain_reference_points = Mock()
        experiment.save_calib_note = Mock()
        return experiment

    @pytest.fixture
    def sample_rabi_param(self):
        """Create a sample RabiParam for testing."""
        return RabiParam(
            target="Q00",
            amplitude=0.5,
            frequency=10.0,
            phase=1.5,
            offset=0.1,
            noise=0.02,
            angle=2.1,
            distance=0.8,
            r2=0.95,
            reference_phase=0.0,
        )

    def test_correct_rabi_params_target_normalization_none(self, mock_experiment, sample_rabi_param):
        """Test correct_rabi_params with targets=None uses all qubit_labels."""
        from qubex.experiment.experiment import Experiment
        
        # Setup
        mock_experiment.rabi_params = {"Q00": sample_rabi_param}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0}}
        
        # Call the actual method
        Experiment.correct_rabi_params(mock_experiment, targets=None, save=False)
        
        # Verify obtain_reference_points was called with all qubit_labels
        mock_experiment.obtain_reference_points.assert_called_once_with(targets=["Q00", "Q01", "Q02"])

    def test_correct_rabi_params_target_normalization_string(self, mock_experiment, sample_rabi_param):
        """Test correct_rabi_params with string target converts to list."""
        from qubex.experiment.experiment import Experiment
        
        # Setup
        mock_experiment.rabi_params = {"Q00": sample_rabi_param}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0}}
        
        # Call the actual method
        Experiment.correct_rabi_params(mock_experiment, targets="Q00", save=False)
        
        # Verify obtain_reference_points was called with converted list
        mock_experiment.obtain_reference_points.assert_called_once_with(targets=["Q00"])

    def test_correct_rabi_params_target_normalization_collection(self, mock_experiment, sample_rabi_param):
        """Test correct_rabi_params with collection target converts to list."""
        from qubex.experiment.experiment import Experiment
        
        # Setup
        mock_experiment.rabi_params = {"Q00": sample_rabi_param, "Q01": sample_rabi_param}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0, "Q01": 1.5}}
        
        # Call the actual method
        Experiment.correct_rabi_params(mock_experiment, targets=("Q00", "Q01"), save=False)
        
        # Verify obtain_reference_points was called with converted list
        mock_experiment.obtain_reference_points.assert_called_once_with(targets=["Q00", "Q01"])

    def test_correct_rabi_params_uses_provided_reference_phases(self, mock_experiment, sample_rabi_param):
        """Test correct_rabi_params uses provided reference_phases instead of obtaining them."""
        from qubex.experiment.experiment import Experiment
        
        # Setup
        mock_experiment.rabi_params = {"Q00": sample_rabi_param}
        reference_phases = {"Q00": 2.5}
        
        # Call the actual method
        Experiment.correct_rabi_params(
            mock_experiment, 
            targets=["Q00"], 
            reference_phases=reference_phases,
            save=False
        )
        
        # Verify obtain_reference_points was NOT called
        mock_experiment.obtain_reference_points.assert_not_called()
        
        # Verify the rabi_param.correct was called with the provided phase
        assert sample_rabi_param.reference_phase == 2.5

    def test_correct_rabi_params_missing_rabi_param(self, mock_experiment, capfd):
        """Test correct_rabi_params handles missing rabi parameters gracefully."""
        from qubex.experiment.experiment import Experiment
        
        # Setup - no rabi_params for the target
        mock_experiment.rabi_params = {}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0}}
        
        # Call the actual method
        Experiment.correct_rabi_params(mock_experiment, targets=["Q00"], save=False)
        
        # Verify warning message was printed
        captured = capfd.readouterr()
        assert "Rabi parameters for Q00 are not stored." in captured.out

    def test_correct_rabi_params_exception_handling(self, mock_experiment, capfd):
        """Test correct_rabi_params handles exceptions gracefully."""
        from qubex.experiment.experiment import Experiment
        
        # Setup - create a rabi_param that will raise an exception
        broken_rabi_param = Mock()
        broken_rabi_param.correct.side_effect = ValueError("Test error")
        mock_experiment.rabi_params = {"Q00": broken_rabi_param}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0}}
        
        # Call the actual method
        Experiment.correct_rabi_params(mock_experiment, targets=["Q00"], save=False)
        
        # Verify error message was printed
        captured = capfd.readouterr()
        assert "Failed to correct Rabi parameters for Q00: Test error" in captured.out

    def test_correct_rabi_params_updates_calib_note(self, mock_experiment, sample_rabi_param):
        """Test correct_rabi_params updates calibration note with corrected parameters."""
        from qubex.experiment.experiment import Experiment
        
        # Setup
        mock_experiment.rabi_params = {"Q00": sample_rabi_param}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0}}
        
        # Call the actual method
        Experiment.correct_rabi_params(mock_experiment, targets=["Q00"], save=False)
        
        # Verify calib_note.update_rabi_param was called with correct parameters
        mock_experiment.calib_note.update_rabi_param.assert_called_once_with(
            "Q00",
            {
                "target": sample_rabi_param.target,
                "frequency": sample_rabi_param.frequency,
                "amplitude": sample_rabi_param.amplitude,
                "phase": sample_rabi_param.phase,
                "offset": sample_rabi_param.offset,
                "noise": sample_rabi_param.noise,
                "angle": sample_rabi_param.angle,
                "distance": sample_rabi_param.distance,
                "r2": sample_rabi_param.r2,
                "reference_phase": sample_rabi_param.reference_phase,
            }
        )

    def test_correct_rabi_params_save_flag_true(self, mock_experiment, sample_rabi_param):
        """Test correct_rabi_params calls save_calib_note when save=True."""
        from qubex.experiment.experiment import Experiment
        
        # Setup
        mock_experiment.rabi_params = {"Q00": sample_rabi_param}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0}}
        
        # Call the actual method with save=True (default)
        Experiment.correct_rabi_params(mock_experiment, targets=["Q00"])
        
        # Verify save_calib_note was called
        mock_experiment.save_calib_note.assert_called_once()

    def test_correct_rabi_params_save_flag_false(self, mock_experiment, sample_rabi_param):
        """Test correct_rabi_params does not save when save=False."""
        from qubex.experiment.experiment import Experiment
        
        # Setup
        mock_experiment.rabi_params = {"Q00": sample_rabi_param}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0}}
        
        # Call the actual method with save=False
        Experiment.correct_rabi_params(mock_experiment, targets=["Q00"], save=False)
        
        # Verify save_calib_note was NOT called
        mock_experiment.save_calib_note.assert_not_called()

    def test_correct_rabi_params_multiple_targets(self, mock_experiment):
        """Test correct_rabi_params works with multiple targets."""
        from qubex.experiment.experiment import Experiment
        
        # Setup multiple rabi_params
        rabi_param_0 = RabiParam(
            target="Q00", amplitude=0.5, frequency=10.0, phase=1.5, offset=0.1,
            noise=0.02, angle=2.1, distance=0.8, r2=0.95, reference_phase=0.0
        )
        rabi_param_1 = RabiParam(
            target="Q01", amplitude=0.6, frequency=11.0, phase=1.6, offset=0.2,
            noise=0.03, angle=2.2, distance=0.9, r2=0.93, reference_phase=0.1
        )
        
        mock_experiment.rabi_params = {"Q00": rabi_param_0, "Q01": rabi_param_1}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0, "Q01": 1.5}}
        
        # Call the actual method
        Experiment.correct_rabi_params(mock_experiment, targets=["Q00", "Q01"], save=False)
        
        # Verify both params were corrected
        assert rabi_param_0.reference_phase == 1.0
        assert rabi_param_1.reference_phase == 1.5
        
        # Verify calib_note was updated for both
        assert mock_experiment.calib_note.update_rabi_param.call_count == 2

    def test_correct_rabi_params_partial_failure(self, mock_experiment, capfd):
        """Test correct_rabi_params continues processing other targets when one fails."""
        from qubex.experiment.experiment import Experiment
        
        # Setup - one good param, one that will fail
        good_param = RabiParam(
            target="Q00", amplitude=0.5, frequency=10.0, phase=1.5, offset=0.1,
            noise=0.02, angle=2.1, distance=0.8, r2=0.95, reference_phase=0.0
        )
        bad_param = Mock()
        bad_param.correct.side_effect = ValueError("Test error")
        
        mock_experiment.rabi_params = {"Q00": good_param, "Q01": bad_param}
        mock_experiment.obtain_reference_points.return_value = {"phase": {"Q00": 1.0, "Q01": 1.5}}
        
        # Call the actual method
        Experiment.correct_rabi_params(mock_experiment, targets=["Q00", "Q01"], save=False)
        
        # Verify good param was corrected
        assert good_param.reference_phase == 1.0
        
        # Verify error message was printed for bad param
        captured = capfd.readouterr()
        assert "Failed to correct Rabi parameters for Q01: Test error" in captured.out
        
        # Verify calib_note was still updated for the good param
        mock_experiment.calib_note.update_rabi_param.assert_called_once()


class TestExperimentValidateRabiParams:
    """Tests for Experiment.validate_rabi_params method."""

    @pytest.fixture
    def mock_experiment(self):
        """Create a mock experiment with rabi_params."""
        experiment = Mock()
        experiment.rabi_params = {
            "Q00": Mock(),
            "Q01": Mock(),
        }
        return experiment

    def test_validate_rabi_params_empty_params(self, mock_experiment):
        """Test validate_rabi_params raises error when no rabi_params stored."""
        from qubex.experiment.experiment import Experiment
        
        mock_experiment.rabi_params = {}
        
        with pytest.raises(ValueError, match="Rabi parameters are not stored."):
            Experiment.validate_rabi_params(mock_experiment)

    def test_validate_rabi_params_targets_none(self, mock_experiment):
        """Test validate_rabi_params passes when targets=None and params exist."""
        from qubex.experiment.experiment import Experiment
        
        # Should not raise any exception
        Experiment.validate_rabi_params(mock_experiment, targets=None)

    def test_validate_rabi_params_valid_targets(self, mock_experiment):
        """Test validate_rabi_params passes when all targets have params."""
        from qubex.experiment.experiment import Experiment
        
        # Should not raise any exception
        Experiment.validate_rabi_params(mock_experiment, targets=["Q00", "Q01"])

    def test_validate_rabi_params_invalid_target(self, mock_experiment):
        """Test validate_rabi_params raises error for missing target."""
        from qubex.experiment.experiment import Experiment
        
        with pytest.raises(ValueError, match="Rabi parameters for Q99 are not stored."):
            Experiment.validate_rabi_params(mock_experiment, targets=["Q00", "Q99"])

    def test_validate_rabi_params_duplicate_check(self, mock_experiment):
        """Test the duplicate check logic in validate_rabi_params."""
        from qubex.experiment.experiment import Experiment
        
        # Note: The original code has duplicate logic, our test verifies this behavior
        with pytest.raises(ValueError, match="Rabi parameters for Q99 are not stored."):
            Experiment.validate_rabi_params(mock_experiment, targets=["Q99"])


class TestExperimentStoreRabiParams:
    """Tests for Experiment.store_rabi_params method."""

    @pytest.fixture
    def mock_experiment(self):
        """Create a mock experiment with calib_note."""
        experiment = Mock()
        experiment.calib_note = Mock()
        return experiment

    def test_store_rabi_params_above_threshold(self, mock_experiment, capfd):
        """Test store_rabi_params stores params above r2 threshold."""
        from qubex.experiment.experiment import Experiment
        
        good_param = RabiParam(
            target="Q00", amplitude=0.5, frequency=10.0, phase=1.5, offset=0.1,
            noise=0.02, angle=2.1, distance=0.8, r2=0.95, reference_phase=0.0
        )
        
        Experiment.store_rabi_params(
            mock_experiment, 
            {"Q00": good_param}, 
            r2_threshold=0.5
        )
        
        # Verify calib_note.update_rabi_param was called
        mock_experiment.calib_note.update_rabi_param.assert_called_once_with(
            "Q00",
            {
                "target": good_param.target,
                "frequency": good_param.frequency,
                "amplitude": good_param.amplitude,
                "phase": good_param.phase,
                "offset": good_param.offset,
                "noise": good_param.noise,
                "angle": good_param.angle,
                "distance": good_param.distance,
                "r2": good_param.r2,
                "reference_phase": good_param.reference_phase,
            }
        )
        
        # Verify no warning message
        captured = capfd.readouterr()
        assert "not stored" not in captured.out

    def test_store_rabi_params_below_threshold(self, mock_experiment, capfd):
        """Test store_rabi_params does not store params below r2 threshold."""
        from qubex.experiment.experiment import Experiment
        
        bad_param = RabiParam(
            target="Q00", amplitude=0.5, frequency=10.0, phase=1.5, offset=0.1,
            noise=0.02, angle=2.1, distance=0.8, r2=0.3, reference_phase=0.0
        )
        
        Experiment.store_rabi_params(
            mock_experiment, 
            {"Q00": bad_param}, 
            r2_threshold=0.5
        )
        
        # Verify calib_note.update_rabi_param was NOT called
        mock_experiment.calib_note.update_rabi_param.assert_not_called()
        
        # Verify warning message was printed
        captured = capfd.readouterr()
        assert "Rabi parameters are not stored for qubits: ['Q00']" in captured.out

    def test_store_rabi_params_mixed_quality(self, mock_experiment, capfd):
        """Test store_rabi_params with mixed quality parameters."""
        from qubex.experiment.experiment import Experiment
        
        good_param = RabiParam(
            target="Q00", amplitude=0.5, frequency=10.0, phase=1.5, offset=0.1,
            noise=0.02, angle=2.1, distance=0.8, r2=0.95, reference_phase=0.0
        )
        bad_param = RabiParam(
            target="Q01", amplitude=0.5, frequency=10.0, phase=1.5, offset=0.1,
            noise=0.02, angle=2.1, distance=0.8, r2=0.3, reference_phase=0.0
        )
        
        Experiment.store_rabi_params(
            mock_experiment, 
            {"Q00": good_param, "Q01": bad_param}, 
            r2_threshold=0.5
        )
        
        # Verify only good param was stored
        mock_experiment.calib_note.update_rabi_param.assert_called_once()
        
        # Verify warning for bad param
        captured = capfd.readouterr()
        assert "Rabi parameters are not stored for qubits: ['Q01']" in captured.out
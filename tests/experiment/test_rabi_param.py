"""Tests for RabiParam class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.experiment.rabi_param import RabiParam


class TestRabiParam:
    """Tests for RabiParam class."""

    def test_init(self):
        """RabiParam should initialize with given parameters."""
        param = RabiParam(
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
        assert param.target == "Q00"
        assert param.amplitude == 0.5
        assert param.frequency == 10.0
        assert param.phase == 1.5
        assert param.offset == 0.1
        assert param.noise == 0.02
        assert param.angle == 2.1
        assert param.distance == 0.8
        assert param.r2 == 0.95
        assert param.reference_phase == 0.0

    def test_nan_class_method(self):
        """RabiParam.nan should create a param with NaN values."""
        param = RabiParam.nan("Q01")
        assert param.target == "Q01"
        assert np.isnan(param.amplitude)
        assert np.isnan(param.frequency)
        assert np.isnan(param.phase)
        assert np.isnan(param.offset)
        assert np.isnan(param.noise)
        assert np.isnan(param.angle)
        assert np.isnan(param.distance)
        assert np.isnan(param.r2)
        assert np.isnan(param.reference_phase)

    def test_endpoints_property(self):
        """RabiParam.endpoints should compute correct endpoint values."""
        param = RabiParam(
            target="Q00",
            amplitude=0.5,
            frequency=10.0,
            phase=1.5,
            offset=0.1,
            noise=0.02,
            angle=np.pi / 2,  # 90 degrees
            distance=0.8,
            r2=0.95,
            reference_phase=0.0,
        )
        
        iq_0, iq_1 = param.endpoints
        
        # With angle = π/2, rotation should swap real and imaginary parts
        # rotated_0 = complex(0.8, 0.6), rotated after exp(i*π/2) should be complex(-0.6, 0.8)
        # rotated_1 = complex(0.8, -0.4), rotated after exp(i*π/2) should be complex(0.4, 0.8)
        expected_0 = complex(0.8 + 0.5j, 0.1) * np.exp(1j * np.pi / 2)
        expected_1 = complex(0.8 - 0.5j, 0.1) * np.exp(1j * np.pi / 2)
        
        # More precise calculation
        rotated_0 = complex(param.distance, param.offset + param.amplitude)
        rotated_1 = complex(param.distance, param.offset - param.amplitude)
        expected_0 = rotated_0 * np.exp(1j * param.angle)
        expected_1 = rotated_1 * np.exp(1j * param.angle)
        
        assert_allclose(iq_0, expected_0, rtol=1e-10)
        assert_allclose(iq_1, expected_1, rtol=1e-10)

    def test_correct_method_first_time(self):
        """RabiParam.correct should set reference_phase if None."""
        param = RabiParam(
            target="Q00",
            amplitude=0.5,
            frequency=10.0,
            phase=1.5,
            offset=0.1,
            noise=0.02,
            angle=2.1,
            distance=0.8,
            r2=0.95,
            reference_phase=None,
        )
        
        original_angle = param.angle
        new_reference_phase = 1.0
        
        param.correct(new_reference_phase)
        
        assert param.reference_phase == new_reference_phase
        assert param.angle == original_angle  # Should not change on first correction

    def test_correct_method_subsequent_correction(self):
        """RabiParam.correct should adjust angle based on phase difference."""
        param = RabiParam(
            target="Q00",
            amplitude=0.5,
            frequency=10.0,
            phase=1.5,
            offset=0.1,
            noise=0.02,
            angle=2.1,
            distance=0.8,
            r2=0.95,
            reference_phase=0.5,
        )
        
        original_angle = param.angle
        original_ref_phase = param.reference_phase
        new_reference_phase = 1.2
        
        param.correct(new_reference_phase)
        
        expected_angle = original_angle + (new_reference_phase - original_ref_phase)
        assert param.reference_phase == new_reference_phase
        assert_allclose(param.angle, expected_angle, rtol=1e-10)

    def test_normalize_method(self):
        """RabiParam.normalize should correctly transform I/Q values."""
        param = RabiParam(
            target="Q00",
            amplitude=0.5,
            frequency=10.0,
            phase=1.5,
            offset=0.1,
            noise=0.02,
            angle=np.pi / 4,  # 45 degrees
            distance=0.8,
            r2=0.95,
            reference_phase=0.0,
        )
        
        # Test with some complex values
        values = np.array([1.0 + 0.5j, 0.2 + 0.8j, -0.3 + 0.1j], dtype=complex)
        
        normalized = param.normalize(values)
        
        # Manual calculation for verification
        rotated = values * np.exp(-1j * param.angle)
        expected = (np.imag(rotated) - param.offset) / param.amplitude
        
        assert_allclose(normalized, expected, rtol=1e-10)

    def test_normalize_with_zero_amplitude(self):
        """RabiParam.normalize should handle zero amplitude case."""
        param = RabiParam(
            target="Q00",
            amplitude=0.0,  # Zero amplitude
            frequency=10.0,
            phase=1.5,
            offset=0.1,
            noise=0.02,
            angle=np.pi / 4,
            distance=0.8,
            r2=0.95,
            reference_phase=0.0,
        )
        
        values = np.array([1.0 + 0.5j, 0.2 + 0.8j], dtype=complex)
        
        # Should result in inf values due to division by zero
        normalized = param.normalize(values)
        assert all(np.isinf(normalized))

    def test_normalize_with_array_inputs(self):
        """RabiParam.normalize should work with various array inputs."""
        param = RabiParam(
            target="Q00",
            amplitude=0.5,
            frequency=10.0,
            phase=1.5,
            offset=0.1,
            noise=0.02,
            angle=0.0,  # No rotation for simplicity
            distance=0.8,
            r2=0.95,
            reference_phase=0.0,
        )
        
        # Test with 1D array
        values_1d = np.array([0.0 + 0.6j, 0.0 + 0.1j], dtype=complex)
        normalized_1d = param.normalize(values_1d)
        expected_1d = (np.imag(values_1d) - param.offset) / param.amplitude
        assert_allclose(normalized_1d, expected_1d, rtol=1e-10)
        
        # Test with 2D array
        values_2d = np.array([[0.0 + 0.6j, 0.0 + 0.1j], [0.0 + 0.3j, 0.0 + 0.8j]], dtype=complex)
        normalized_2d = param.normalize(values_2d)
        expected_2d = (np.imag(values_2d) - param.offset) / param.amplitude
        assert_allclose(normalized_2d, expected_2d, rtol=1e-10)

    def test_correct_phase_progression(self):
        """Test multiple phase corrections work correctly."""
        param = RabiParam(
            target="Q00",
            amplitude=0.5,
            frequency=10.0,
            phase=1.5,
            offset=0.1,
            noise=0.02,
            angle=1.0,
            distance=0.8,
            r2=0.95,
            reference_phase=None,
        )
        
        # First correction
        param.correct(0.5)
        assert param.reference_phase == 0.5
        assert param.angle == 1.0
        
        # Second correction
        param.correct(1.0)
        assert param.reference_phase == 1.0
        assert_allclose(param.angle, 1.5, rtol=1e-10)
        
        # Third correction  
        param.correct(0.2)
        assert param.reference_phase == 0.2
        assert_allclose(param.angle, 0.7, rtol=1e-10)
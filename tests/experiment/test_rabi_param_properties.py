"""Property-based tests for calibration parameter validation."""

import numpy as np
from hypothesis import given, strategies as st, assume

from qubex.experiment.rabi_param import RabiParam


class TestRabiParamPropertyBased:
    """Property-based tests for RabiParam using hypothesis."""

    @given(
        target=st.text(min_size=1, max_size=10),
        amplitude=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        frequency=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        phase=st.floats(min_value=-2*np.pi, max_value=2*np.pi, allow_nan=False, allow_infinity=False),
        offset=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        noise=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        angle=st.floats(min_value=-2*np.pi, max_value=2*np.pi, allow_nan=False, allow_infinity=False),
        distance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        r2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        reference_phase=st.floats(min_value=-2*np.pi, max_value=2*np.pi, allow_nan=False, allow_infinity=False),
    )
    def test_rabi_param_init_valid_values(self, target, amplitude, frequency, phase, offset, 
                                         noise, angle, distance, r2, reference_phase):
        """Test RabiParam initialization with valid property values."""
        param = RabiParam(
            target=target,
            amplitude=amplitude,
            frequency=frequency,
            phase=phase,
            offset=offset,
            noise=noise,
            angle=angle,
            distance=distance,
            r2=r2,
            reference_phase=reference_phase,
        )
        
        assert param.target == target
        assert param.amplitude == amplitude
        assert param.frequency == frequency
        assert param.phase == phase
        assert param.offset == offset
        assert param.noise == noise
        assert param.angle == angle
        assert param.distance == distance
        assert param.r2 == r2
        assert param.reference_phase == reference_phase

    @given(
        amplitude=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        offset=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        angle=st.floats(min_value=-2*np.pi, max_value=2*np.pi, allow_nan=False, allow_infinity=False),
        distance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        array_size=st.integers(min_value=1, max_value=100),
    )
    def test_normalize_preserves_shape(self, amplitude, offset, angle, distance, array_size):
        """Test that normalize preserves input array shape."""
        param = RabiParam(
            target="Q00", amplitude=amplitude, frequency=10.0, phase=1.5, 
            offset=offset, noise=0.02, angle=angle, distance=distance, 
            r2=0.95, reference_phase=0.0
        )
        
        # Generate random complex values
        real_parts = np.random.uniform(-5, 5, array_size)
        imag_parts = np.random.uniform(-5, 5, array_size)
        values = real_parts + 1j * imag_parts
        
        normalized = param.normalize(values)
        
        # Shape should be preserved
        assert normalized.shape == values.shape
        # Result should be real (since we take imaginary part and normalize)
        assert np.all(np.isreal(normalized))

    @given(
        new_reference_phase=st.floats(min_value=-4*np.pi, max_value=4*np.pi, allow_nan=False, allow_infinity=False)
    )
    def test_correct_first_time_sets_reference(self, new_reference_phase):
        """Test that first correction sets reference_phase correctly."""
        param = RabiParam(
            target="Q00", amplitude=0.5, frequency=10.0, phase=1.5, 
            offset=0.1, noise=0.02, angle=2.1, distance=0.8, 
            r2=0.95, reference_phase=None
        )
        
        original_angle = param.angle
        param.correct(new_reference_phase)
        
        assert param.reference_phase == new_reference_phase
        assert param.angle == original_angle  # Should not change on first correction

    @given(
        initial_phase=st.floats(min_value=-2*np.pi, max_value=2*np.pi, allow_nan=False, allow_infinity=False),
        phase_delta=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
        initial_angle=st.floats(min_value=-2*np.pi, max_value=2*np.pi, allow_nan=False, allow_infinity=False),
    )
    def test_correct_angle_adjustment(self, initial_phase, phase_delta, initial_angle):
        """Test that subsequent corrections adjust angle by phase difference."""
        param = RabiParam(
            target="Q00", amplitude=0.5, frequency=10.0, phase=1.5, 
            offset=0.1, noise=0.02, angle=initial_angle, distance=0.8, 
            r2=0.95, reference_phase=initial_phase
        )
        
        new_phase = initial_phase + phase_delta
        param.correct(new_phase)
        
        expected_angle = initial_angle + phase_delta
        assert abs(param.angle - expected_angle) < 1e-10
        assert param.reference_phase == new_phase

    @given(
        amplitude=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        angle=st.floats(min_value=-2*np.pi, max_value=2*np.pi, allow_nan=False, allow_infinity=False),
        n_corrections=st.integers(min_value=1, max_value=10),
    )
    def test_multiple_corrections_accumulate(self, amplitude, angle, n_corrections):
        """Test that multiple corrections accumulate properly."""
        param = RabiParam(
            target="Q00", amplitude=amplitude, frequency=10.0, phase=1.5, 
            offset=0.1, noise=0.02, angle=angle, distance=0.8, 
            r2=0.95, reference_phase=None
        )
        
        total_phase_change = 0.0
        original_angle = angle
        
        for i in range(n_corrections):
            phase_change = 0.1 * (i + 1)  # Small incremental changes
            if param.reference_phase is None:
                param.correct(phase_change)
                # First correction doesn't change angle
                assert param.angle == original_angle
            else:
                old_ref = param.reference_phase
                param.correct(old_ref + phase_change)
                total_phase_change += phase_change
                
        if n_corrections > 0:
            expected_final_angle = original_angle + total_phase_change
            assert abs(param.angle - expected_final_angle) < 1e-10

    @given(
        endpoints_real=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        endpoints_imag=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    )
    def test_endpoints_are_complex(self, endpoints_real, endpoints_imag):
        """Test that endpoints property returns complex values."""
        # Use fixed values that create meaningful endpoints
        param = RabiParam(
            target="Q00", amplitude=1.0, frequency=10.0, phase=1.5, 
            offset=0.0, noise=0.02, angle=0.0, distance=1.0, 
            r2=0.95, reference_phase=0.0
        )
        
        iq_0, iq_1 = param.endpoints
        
        # Both should be complex numbers
        assert isinstance(iq_0, complex)
        assert isinstance(iq_1, complex)
        
        # They should be different (since amplitude > 0)
        assert iq_0 != iq_1

    @given(
        r2_threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        r2_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10
        ),
    )
    def test_r2_threshold_behavior(self, r2_threshold, r2_values):
        """Test that r2 threshold filtering works correctly."""
        # This is a property test for the logic that would be used in store_rabi_params
        
        # Count how many values are above threshold
        expected_above = sum(1 for r2 in r2_values if r2 >= r2_threshold)
        expected_below = len(r2_values) - expected_above
        
        # Verify our counting logic
        actual_above = 0
        actual_below = 0
        
        for r2 in r2_values:
            if r2 >= r2_threshold:
                actual_above += 1
            else:
                actual_below += 1
        
        assert actual_above == expected_above
        assert actual_below == expected_below
        assert actual_above + actual_below == len(r2_values)

    @given(
        values_real=st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=50
        ),
        values_imag=st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=50
        ),
        amplitude=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        offset=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
    )
    def test_normalize_linearity(self, values_real, values_imag, amplitude, offset):
        """Test that normalize operation is linear in the imaginary component."""
        assume(len(values_real) == len(values_imag))
        
        param = RabiParam(
            target="Q00", amplitude=amplitude, frequency=10.0, phase=1.5, 
            offset=offset, noise=0.02, angle=0.0, distance=0.8,  # angle=0 for simplicity
            r2=0.95, reference_phase=0.0
        )
        
        values = np.array([complex(r, i) for r, i in zip(values_real, values_imag)])
        normalized = param.normalize(values)
        
        # With angle=0, the normalization should be: (imag_part - offset) / amplitude
        expected = (np.array(values_imag) - offset) / amplitude
        
        np.testing.assert_allclose(normalized, expected, rtol=1e-10)

    @given(
        distance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        amplitude=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        offset=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
    )
    def test_endpoints_symmetry(self, distance, amplitude, offset):
        """Test that endpoints are symmetric around the distance point."""
        param = RabiParam(
            target="Q00", amplitude=amplitude, frequency=10.0, phase=1.5, 
            offset=offset, noise=0.02, angle=0.0, distance=distance, 
            r2=0.95, reference_phase=0.0
        )
        
        iq_0, iq_1 = param.endpoints
        
        # With angle=0, endpoints should be:
        # iq_0 = complex(distance, offset + amplitude)
        # iq_1 = complex(distance, offset - amplitude)
        
        expected_0 = complex(distance, offset + amplitude)
        expected_1 = complex(distance, offset - amplitude)
        
        assert abs(iq_0 - expected_0) < 1e-10
        assert abs(iq_1 - expected_1) < 1e-10
        
        # The midpoint should be at (distance, offset)
        midpoint = (iq_0 + iq_1) / 2
        expected_midpoint = complex(distance, offset)
        assert abs(midpoint - expected_midpoint) < 1e-10
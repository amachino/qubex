from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from qubex.experiment.experiment_context import ExperimentContext
from qubex.experiment.experiment_exceptions import CalibrationMissingError
from qubex.experiment.services.pulse_service import PulseService
from qubex.pulse import Drag, FlatTop, PulseArray, PulseSchedule, VirtualZ, Waveform


class TestPulseService:
    @pytest.fixture
    def mock_context(self):
        context = MagicMock(spec=ExperimentContext)
        # Setup basic params
        context._calibration_valid_days = 30
        context.params.get_control_amplitude.return_value = 0.5
        context.readout_duration = 1000.0
        context.readout_pre_margin = 10.0
        context.readout_post_margin = 10.0

        # Setup qubits
        q0 = MagicMock()
        q0.index = 0
        q1 = MagicMock()
        q1.index = 1
        context.qubits = {"Q0": q0, "Q1": q1}

        return context

    @pytest.fixture
    def pulse_service(self, mock_context):
        return PulseService(mock_context)

    def test_get_hpi_pulse_with_calibration(self, pulse_service, mock_context):
        mock_context.calib_note.get_hpi_param.return_value = {
            "duration": 40,
            "amplitude": 0.4,
            "tau": 10,
        }

        pulse = pulse_service.get_hpi_pulse("Q0")

        assert isinstance(pulse, FlatTop)
        assert pulse.duration == 40
        assert pulse.amplitude == 0.4

        mock_context.calib_note.get_hpi_param.assert_called_with("Q0", valid_days=30)

    def test_get_hpi_pulse_default(self, pulse_service, mock_context):
        mock_context.calib_note.get_hpi_param.return_value = None

        pulse = pulse_service.get_hpi_pulse("Q0")

        assert isinstance(pulse, FlatTop)
        assert pulse.amplitude == 0.5  # From params.get_control_amplitude

    def test_get_pi_pulse_missing(self, pulse_service, mock_context):
        mock_context.calib_note.get_pi_param.return_value = None

        with pytest.raises(CalibrationMissingError):
            pulse_service.get_pi_pulse("Q0")

    def test_get_drag_hpi_pulse(self, pulse_service, mock_context):
        mock_context.calib_note.get_drag_hpi_param.return_value = {
            "duration": 30,
            "amplitude": 0.3,
            "beta": 0.1,
        }

        pulse = pulse_service.get_drag_hpi_pulse("Q0")

        assert isinstance(pulse, Drag)
        assert pulse.duration == 30
        assert pulse.beta == 0.1

    def test_x90_uses_drag_if_available(self, pulse_service, mock_context):
        mock_context.calib_note.get_drag_hpi_param.return_value = {
            "duration": 30,
            "amplitude": 0.3,
            "beta": 0.1,
        }

        pulse = pulse_service.x90("Q0")
        assert isinstance(pulse, Drag)

    def test_x90_fallback_to_hpi(self, pulse_service, mock_context):
        mock_context.calib_note.get_drag_hpi_param.return_value = None
        mock_context.calib_note.get_hpi_param.return_value = {
            "duration": 40,
            "amplitude": 0.4,
            "tau": 10,
        }

        pulse = pulse_service.x90("Q0")
        assert isinstance(pulse, FlatTop)

    def test_gates_and_phases(self, pulse_service, mock_context):
        mock_context.calib_note.get_drag_hpi_param.return_value = {
            "duration": 30,
            "amplitude": 0.3,
            "beta": 0.1,
        }

        x90 = pulse_service.x90("Q0")
        x90m = pulse_service.x90m("Q0")
        y90 = pulse_service.y90("Q0")

        # Simple check for scaling and shifting
        # Note: Waveform equality check might depend on implementation
        # Here we just check they don't crash and return Waveforms
        assert isinstance(x90, Waveform)
        assert isinstance(x90m, Waveform)
        assert isinstance(y90, Waveform)

        z90 = pulse_service.z90()
        assert isinstance(z90, VirtualZ)
        # z90 delegates to VirtualZ(np.pi/2) which sets theta to -np.pi/2
        assert np.isclose(z90.theta, -np.pi / 2)

    def test_readout(self, pulse_service, mock_context):
        mock_context.measurement.readout_pulse.return_value = MagicMock(spec=Waveform)

        pulse = pulse_service.readout("Q0")

        mock_context.measurement.readout_pulse.assert_called_with(
            target="Q0",
            duration=1000.0,
            amplitude=None,
            ramptime=None,
            type=None,
            drag_coeff=None,
            pre_margin=10.0,
            post_margin=10.0,
        )
        assert pulse == mock_context.measurement.readout_pulse.return_value

    def test_hadamard(self, pulse_service, mock_context):
        mock_context.calib_note.get_drag_hpi_param.return_value = {
            "duration": 30,
            "amplitude": 0.3,
            "beta": 0.1,
        }
        # mock x180 via get_drag_pi_pulse
        mock_context.calib_note.get_drag_pi_param.return_value = {
            "duration": 30,
            "amplitude": 0.6,
            "beta": 0.1,
        }

        h_pulse = pulse_service.hadamard("Q0", decomposition="Y90-X180")
        assert isinstance(h_pulse, PulseArray)

    def test_zx90(self, pulse_service, mock_context):
        mock_context.calib_note.get_cr_param.return_value = {
            "cr_amplitude": 0.2,
            "duration": 100,
            "ramptime": 10,
            "cr_phase": 0,
            "cr_beta": 0,
            "cancel_amplitude": 0,
            "cancel_phase": 0,
            "cancel_beta": 0,
            "rotary_amplitude": 0,
        }
        # Mock pi pulse for echo
        mock_context.calib_note.get_drag_pi_param.return_value = {
            "duration": 30,
            "amplitude": 0.6,
            "beta": 0.1,
        }

        cr_schedule = pulse_service.zx90("Q0", "Q1")
        assert isinstance(cr_schedule, PulseSchedule)

    def test_cnot(self, pulse_service, mock_context):
        # Setup CR params
        mock_context.calib_note.get_cr_param.return_value = {
            "cr_amplitude": 0.2,
            "duration": 100,
            "ramptime": 10,
            "cr_phase": 0,
            "cr_beta": 0,
            "cancel_amplitude": 0,
            "cancel_phase": 0,
            "cancel_beta": 0,
            "rotary_amplitude": 0,
        }
        # Setup single qubit pulses
        mock_context.calib_note.get_drag_hpi_param.return_value = {
            "duration": 30,
            "amplitude": 0.3,
            "beta": 0.1,
        }
        mock_context.calib_note.get_drag_pi_param.return_value = {
            "duration": 30,
            "amplitude": 0.6,
            "beta": 0.1,
        }

        # Test low to high connection (often direct CR)
        # Q0 index 0, Q1 index 1 -> low to high
        mock_context.calib_note.cr_params = {"Q0-Q1": {}}

        cnot_schedule = pulse_service.cnot("Q0", "Q1")
        assert isinstance(cnot_schedule, PulseSchedule)

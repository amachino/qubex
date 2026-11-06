"""Compatibility contracts for resonator spectroscopy orchestration."""

from typing import Any, cast

import numpy as np

from qubex.experiment.services.characterization_service import CharacterizationService
from tests.experiment.test_characterization_service_electrical_delay import _FakeContext


def test_resonator_spectroscopy_scans_complete_frequency_range_per_power():
    """Each power runs the existing frequency scan and preserves its phase-difference row."""
    calls = []
    ctx = _FakeContext()
    service = CharacterizationService(
        experiment_context=cast(Any, ctx),
        measurement_service=cast(Any, None),
        calibration_service=cast(Any, None),
        pulse_service=cast(Any, None),
    )
    frequencies = np.array([5.0, 5.1, 5.4, 5.5])
    powers = np.array([-20.0, -10.0])

    def scan(target, **kwargs):
        calls.append((target, kwargs))
        return {"phases_diff": np.array([-1.0, 2.0, -3.0]) * len(calls)}

    service.scan_resonator_frequencies = cast(Any, scan)
    result = service.resonator_spectroscopy(
        "Q00",
        frequency_range=frequencies,
        power_range=powers,
        electrical_delay=0.0,
        shots=13,
        interval=2048,
        plot=False,
        save_image=False,
    )
    assert len(calls) == 2
    for (target, kwargs), power in zip(calls, powers, strict=True):
        assert target == "Q00"
        np.testing.assert_array_equal(kwargs["frequency_range"], frequencies)
        assert kwargs["readout_amplitude"] == np.sqrt(10 ** (power / 10))
        assert kwargs["electrical_delay"] == 0.0
        assert kwargs["shots"] == 13
        assert kwargs["interval"] == 2048
        assert kwargs["plot"] is False
        assert kwargs["save_image"] is False
    np.testing.assert_array_equal(result.data["data"], [[1, 2, 3, 3], [2, 4, 6, 6]])

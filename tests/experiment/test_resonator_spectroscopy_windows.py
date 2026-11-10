"""Compatibility contracts for resonator spectroscopy orchestration."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

from qubex.experiment.services.characterization_service import CharacterizationService
from tests.experiment.test_characterization_service_electrical_delay import _FakeContext


@pytest.mark.parametrize("subrange_width", ["omitted", None, 0.9])
def test_resonator_spectroscopy_scans_complete_frequency_range_per_power(
    subrange_width,
):
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
    kwargs = {} if subrange_width == "omitted" else {"subrange_width": subrange_width}
    result = service.resonator_spectroscopy(
        "Q00",
        frequency_range=frequencies,
        power_range=powers,
        electrical_delay=0.0,
        shots=13,
        interval=2048,
        plot=False,
        save_image=False,
        **kwargs,
    )
    assert len(calls) == 2
    for (target, kwargs), power in zip(calls, powers, strict=True):
        assert target == "Q00"
        np.testing.assert_array_equal(kwargs["frequency_range"], frequencies)
        assert kwargs["readout_amplitude"] == np.sqrt(10 ** (power / 10))
        assert kwargs["electrical_delay"] == 0.0
        assert kwargs["subrange_width"] == (
            None if subrange_width == "omitted" else subrange_width
        )
        assert kwargs["shots"] == 13
        assert kwargs["interval"] == 2048
        assert kwargs["plot"] is False
        assert kwargs["save_image"] is False
    np.testing.assert_array_equal(result.data["data"], [[1, 2, 3, 3], [2, 4, 6, 6]])


def test_resonator_spectroscopy_calibrates_delay_without_confirmation(monkeypatch):
    """Automatic delay calibration in resonator spectroscopy must not prompt."""
    ctx = _FakeContext()
    ctx.measurement = SimpleNamespace(
        measure=lambda *_args, **_kwargs: SimpleNamespace(
            data={
                "Q00": SimpleNamespace(
                    kerneled=np.exp(-1j * 2 * np.pi * ctx.current_frequency)
                )
            }
        )
    )
    service = CharacterizationService(
        experiment_context=cast(Any, ctx),
        measurement_service=cast(Any, None),
        calibration_service=cast(Any, None),
        pulse_service=cast(
            Any,
            SimpleNamespace(
                readout_duration=2000, readout_pre_margin=0, readout_post_margin=0
            ),
        ),
    )
    scan = Mock(return_value={"phases_diff": np.array([1.0, 2.0])})
    monkeypatch.setattr(service, "scan_resonator_frequencies", scan)
    confirm = Mock(side_effect=AssertionError("Spectroscopy must not prompt"))
    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.Confirm.ask", confirm
    )

    result = service.resonator_spectroscopy(
        "Q00",
        frequency_range=[1.0, 1.01, 1.02],
        power_range=[-20.0],
        plot=False,
        save_image=False,
    )

    confirm.assert_not_called()
    scan.assert_called_once()
    assert scan.call_args.kwargs["electrical_delay"] == pytest.approx(
        1.0, rel=0, abs=1e-9
    )
    np.testing.assert_array_equal(result.data["data"], [[1.0, 2.0, 2.0]])

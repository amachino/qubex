"""Tests for measurement configuration model."""

from __future__ import annotations

from qubex.measurement.measurement_defaults import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.measurement.models import MeasurementConfig


def test_create_applies_legacy_defaults() -> None:
    """Given omitted shots/interval, when building config, then legacy defaults are applied."""
    config = MeasurementConfig.create()

    assert config.mode == "avg"
    assert config.shots == DEFAULT_SHOTS
    assert config.interval == DEFAULT_INTERVAL
    assert config.readout.readout_duration is None
    assert config.dsp.enable_dsp_demodulation is True


def test_create_maps_dsp_and_line_params() -> None:
    """Given dsp args, when building config, then DSP and line params are set in dsp config."""
    config = MeasurementConfig.create(
        enable_dsp_demodulation=False,
        enable_dsp_sum=True,
        enable_dsp_classification=True,
        line_param0=(1.0, 2.0, 3.0),
        line_param1=(4.0, 5.0, 6.0),
    )

    assert config.dsp.enable_dsp_demodulation is False
    assert config.dsp.enable_dsp_sum is True
    assert config.dsp.enable_dsp_classification is True
    assert config.dsp.line_param0 == (1.0, 2.0, 3.0)
    assert config.dsp.line_param1 == (4.0, 5.0, 6.0)

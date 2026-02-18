"""Tests for MeasurementPulseFactory waveform generation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from qxpulse import Blank, FlatTop

from qubex.backend import ControlParams, Mux
from qubex.measurement import MeasurementPulseFactory
from qubex.measurement.measurement_defaults import (
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
    DEFAULT_READOUT_RAMPTIME,
)


def test_readout_pulse_uses_defaults_from_control_params() -> None:
    """Given default args, when readout pulse is built, then defaults are applied."""
    control_params = cast(
        ControlParams,
        SimpleNamespace(
            get_readout_amplitude=lambda qubit: 0.25 if qubit == "Q00" else 0.0
        ),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
    )

    pulse = factory.readout_pulse(target="RQ00")

    assert len(pulse.waveforms) == 2
    assert isinstance(pulse.waveforms[0], Blank)
    assert isinstance(pulse.waveforms[1], FlatTop)
    assert pulse.waveforms[0].duration == DEFAULT_READOUT_PRE_MARGIN
    assert (
        pulse.waveforms[1].duration
        == DEFAULT_READOUT_DURATION + DEFAULT_READOUT_POST_MARGIN
    )
    assert pulse.waveforms[1].amplitude == 0.25
    assert pulse.waveforms[1].tau == DEFAULT_READOUT_RAMPTIME


def test_pump_pulse_uses_mux_index_to_resolve_amplitude() -> None:
    """Given mux mapping, when pump pulse is built, then pump amplitude uses mux index."""
    mux = cast(Mux, SimpleNamespace(index=2, label="MX2"))
    control_params = cast(
        ControlParams,
        SimpleNamespace(get_pump_amplitude=lambda index: 0.1 * (index + 1)),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={"Q00": mux},
    )

    pulse = factory.pump_pulse(mux_index=mux.index)

    assert isinstance(pulse, FlatTop)
    assert pulse.duration == DEFAULT_READOUT_DURATION
    assert pulse.amplitude == pytest.approx(0.3)
    assert pulse.tau == DEFAULT_READOUT_RAMPTIME


def test_readout_pulse_uses_target_registry_when_label_is_custom() -> None:
    """Given custom target label, when registry is provided, then amplitude lookup uses resolved qubit."""
    control_params = cast(
        ControlParams,
        SimpleNamespace(
            get_readout_amplitude=lambda qubit: 0.4 if qubit == "Q17" else 0.0
        ),
    )

    class _TargetRegistry:
        @staticmethod
        def resolve_qubit_label(target: str) -> str:
            return "Q17" if target == "raw-readout-target" else "Q00"

    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
        target_registry=cast(Any, _TargetRegistry()),
    )

    pulse = factory.readout_pulse(target="raw-readout-target")

    assert isinstance(pulse.waveforms[1], FlatTop)
    assert pulse.waveforms[1].amplitude == pytest.approx(0.4)

"""Tests for MeasurementPulseFactory waveform generation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from qxpulse import Blank, FlatTop

from qubex.measurement import MeasurementPulseFactory
from qubex.measurement.measurement_defaults import (
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
    DEFAULT_READOUT_RAMP_TIME,
)
from qubex.system import ControlParameters, Mux


def test_readout_pulse_uses_defaults_from_control_params() -> None:
    """Given default args, when readout pulse is built, then defaults are applied."""
    control_params = cast(
        ControlParameters,
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
    assert pulse.waveforms[1].tau == DEFAULT_READOUT_RAMP_TIME


def test_readout_pulse_uses_configured_measurement_defaults() -> None:
    """Given measurement defaults overrides, when readout pulse is built, then configured timings are applied."""
    control_params = cast(
        ControlParameters,
        SimpleNamespace(
            get_readout_amplitude=lambda qubit: 0.25 if qubit == "Q00" else 0.0
        ),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
        measurement_defaults={
            "readout": {
                "duration_ns": 512.0,
                "ramp_time_ns": 24.0,
                "pre_margin_ns": 16.0,
                "post_margin_ns": 96.0,
            }
        },
    )

    pulse = factory.readout_pulse(target="RQ00")

    assert len(pulse.waveforms) == 2
    assert isinstance(pulse.waveforms[0], Blank)
    assert isinstance(pulse.waveforms[1], FlatTop)
    assert pulse.waveforms[0].duration == 16.0
    assert pulse.waveforms[1].duration == 512.0 + 96.0
    assert pulse.waveforms[1].amplitude == 0.25
    assert pulse.waveforms[1].tau == 24.0


def test_readout_pulse_accepts_renamed_ramp_parameters() -> None:
    """Given renamed ramp args, when readout pulse is built, then they are applied."""
    control_params = cast(
        ControlParameters,
        SimpleNamespace(
            get_readout_amplitude=lambda qubit: 0.25 if qubit == "Q00" else 0.0
        ),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
    )

    pulse = factory.readout_pulse(
        target="RQ00",
        ramp_time=16.0,
        ramp_type="Bump",
    )

    assert isinstance(pulse.waveforms[1], FlatTop)
    assert pulse.waveforms[1].tau == 16.0
    assert pulse.waveforms[1].type == "Bump"


def test_readout_pulse_accepts_legacy_ramp_aliases_with_warning() -> None:
    """Given legacy ramp args, when readout pulse is built, then deprecation warning is emitted."""
    control_params = cast(
        ControlParameters,
        SimpleNamespace(
            get_readout_amplitude=lambda qubit: 0.25 if qubit == "Q00" else 0.0
        ),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
    )

    with pytest.warns(DeprecationWarning, match="ramptime"):
        pulse = factory.readout_pulse(
            target="RQ00",
            ramptime=20.0,
            type="RaisedCosine",
        )

    assert isinstance(pulse.waveforms[1], FlatTop)
    assert pulse.waveforms[1].tau == 20.0
    assert pulse.waveforms[1].type == "RaisedCosine"


def test_readout_pulse_rejects_conflicting_ramp_aliases() -> None:
    """Given legacy and renamed ramp args, when values conflict, then ValueError is raised."""
    control_params = cast(
        ControlParameters,
        SimpleNamespace(
            get_readout_amplitude=lambda qubit: 0.25 if qubit == "Q00" else 0.0
        ),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
    )

    with (
        pytest.warns(DeprecationWarning, match="ramptime"),
        pytest.raises(ValueError, match="ramptime"),
    ):
        factory.readout_pulse(
            target="RQ00",
            ramp_time=16.0,
            ramptime=20.0,
        )


def test_pump_pulse_uses_mux_index_to_resolve_amplitude() -> None:
    """Given mux mapping, when pump pulse is built, then pump amplitude uses mux index."""
    mux = cast(Mux, SimpleNamespace(index=2, label="MX2"))
    control_params = cast(
        ControlParameters,
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
    assert pulse.tau == DEFAULT_READOUT_RAMP_TIME


def test_pump_pulse_uses_configured_measurement_defaults() -> None:
    """Given measurement defaults overrides, when pump pulse is built, then configured timings are applied."""
    control_params = cast(
        ControlParameters,
        SimpleNamespace(get_pump_amplitude=lambda index: 0.1 * (index + 1)),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
        measurement_defaults={
            "readout": {
                "duration_ns": 512.0,
                "ramp_time_ns": 24.0,
            }
        },
    )

    pulse = factory.pump_pulse(mux_index=2)

    assert isinstance(pulse, FlatTop)
    assert pulse.duration == 512.0
    assert pulse.amplitude == pytest.approx(0.3)
    assert pulse.tau == 24.0


def test_pump_pulse_accepts_renamed_ramp_parameters() -> None:
    """Given renamed ramp args, when pump pulse is built, then they are applied."""
    control_params = cast(
        ControlParameters,
        SimpleNamespace(get_pump_amplitude=lambda index: 0.1 * (index + 1)),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
    )

    pulse = factory.pump_pulse(
        mux_index=2,
        ramp_time=16.0,
        ramp_type="Bump",
    )

    assert isinstance(pulse, FlatTop)
    assert pulse.tau == 16.0
    assert pulse.type == "Bump"


def test_pump_pulse_accepts_legacy_ramp_aliases_with_warning() -> None:
    """Given legacy ramp args, when pump pulse is built, then deprecation warning is emitted."""
    control_params = cast(
        ControlParameters,
        SimpleNamespace(get_pump_amplitude=lambda index: 0.1 * (index + 1)),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
    )

    with pytest.warns(DeprecationWarning, match="ramptime"):
        pulse = factory.pump_pulse(
            mux_index=2,
            ramptime=20.0,
            type="RaisedCosine",
        )

    assert isinstance(pulse, FlatTop)
    assert pulse.tau == 20.0
    assert pulse.type == "RaisedCosine"


def test_pump_pulse_rejects_conflicting_ramp_aliases() -> None:
    """Given legacy and renamed ramp args, when values conflict, then ValueError is raised."""
    control_params = cast(
        ControlParameters,
        SimpleNamespace(get_pump_amplitude=lambda index: 0.1 * (index + 1)),
    )
    factory = MeasurementPulseFactory(
        control_params=control_params,
        mux_dict={},
    )

    with (
        pytest.warns(DeprecationWarning, match="ramptime"),
        pytest.raises(ValueError, match="ramptime"),
    ):
        factory.pump_pulse(
            mux_index=2,
            ramp_time=16.0,
            ramptime=20.0,
        )


def test_readout_pulse_uses_target_registry_when_label_is_custom() -> None:
    """Given custom target label, when registry is provided, then amplitude lookup uses resolved qubit."""
    control_params = cast(
        ControlParameters,
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

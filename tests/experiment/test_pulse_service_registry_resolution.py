"""Tests for target-registry resolution in pulse service."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.experiment.models.rabi_param import RabiParam
from qubex.experiment.services import pulse_service as pulse_service_module
from qubex.experiment.services.pulse_service import PulseService
from qubex.pulse import Arbitrary
from qubex.system.target_type import TargetType


def _make_rabi_param(target: str, frequency: float) -> RabiParam:
    return RabiParam(
        target=target,
        amplitude=1.0,
        frequency=frequency,
        phase=0.0,
        offset=0.0,
        noise=0.0,
        angle=0.0,
        distance=1.0,
        r2=1.0,
        reference_phase=0.0,
    )


def test_calc_control_amplitudes_resolves_qubits_via_target_registry() -> None:
    """Given custom targets, when inferring current amplitudes, then qubits are resolved via target registry."""

    class _TargetRegistry:
        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            return "Q17" if label == "custom-target" else label

    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_qubit_label=lambda label: _TargetRegistry.resolve_qubit_label(label),
    )

    ctx = SimpleNamespace(
        experiment_system=experiment_system,
        resolve_qubit_label=lambda label: experiment_system.resolve_qubit_label(label),
        params=SimpleNamespace(
            get_control_amplitude=lambda qubit: 0.25 if qubit == "Q17" else 0.0
        ),
    )
    service = PulseService(experiment_context=cast(Any, ctx))

    amplitudes = service.calc_control_amplitudes(
        rabi_rate=0.2,
        current_rabi_params={"custom-target": _make_rabi_param("custom-target", 0.1)},
        current_amplitudes={},
        print_result=False,
    )

    assert amplitudes == {"custom-target": 0.5}


def test_cr_pulse_uses_context_cr_pair_resolution(monkeypatch) -> None:
    """Given custom CR labels, when building CR pulses, then pair resolution is delegated to context."""
    captured: dict[str, object] = {}

    def _fake_cross_resonance(**kwargs: object) -> dict[str, object]:
        captured["kwargs"] = kwargs
        return dict(kwargs)

    monkeypatch.setattr(pulse_service_module, "CrossResonance", _fake_cross_resonance)

    ctx = SimpleNamespace(
        cr_targets=["CR_CUSTOM"],
        cr_pair=lambda _label: ("Q17", "Q18"),
        calib_note=SimpleNamespace(
            get_cr_param=lambda _label: {
                "cancel_amplitude": 0.1,
                "cancel_phase": 0.2,
                "rotary_amplitude": 0.3,
                "cr_amplitude": 0.4,
                "duration": 128.0,
                "ramptime": 16.0,
                "cr_phase": 0.0,
                "cr_beta": 0.0,
                "cancel_beta": 0.0,
            }
        ),
        calibration_valid_days=7,
    )
    service = PulseService(experiment_context=cast(Any, ctx))
    service.__dict__["x180"] = lambda _target: Arbitrary([])

    pulses = service.cr_pulse

    assert "CR_CUSTOM" in pulses
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs["control_qubit"] == "Q17"
    assert kwargs["target_qubit"] == "Q18"


def test_calc_control_amplitude_resolves_qubit_for_ef_target() -> None:
    """Given EF target, when resolving control amplitude, then registry-resolved qubit is used."""

    class _TargetRegistry:
        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            return "Q17" if label == "custom-ef-target" else label

    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_qubit_label=lambda label: _TargetRegistry.resolve_qubit_label(label),
    )

    ctx = SimpleNamespace(
        experiment_system=experiment_system,
        resolve_qubit_label=lambda label: experiment_system.resolve_qubit_label(label),
        ge_targets={},
        ef_targets={"custom-ef-target": object()},
        get_rabi_param=lambda target: _make_rabi_param(target, 0.1),
        targets={"custom-ef-target": SimpleNamespace(type=TargetType.CTRL_EF)},
        params=SimpleNamespace(
            get_control_amplitude=lambda _qubit: 0.0,
            get_ef_control_amplitude=lambda qubit: 0.5 if qubit == "Q17" else 0.0,
        ),
    )
    service = PulseService(experiment_context=cast(Any, ctx))

    amplitude = service.calc_control_amplitude("custom-ef-target", rabi_rate=0.2)

    assert amplitude == 1.0


def test_unavailable_target_rabi_params_still_support_control_amplitude() -> None:
    """Given unavailable GE target, when using stored Rabi params, then control amplitude is still derived."""
    ctx = SimpleNamespace(
        ge_targets={},
        ef_targets={},
        targets={
            "custom-target": SimpleNamespace(
                is_ge=True,
                is_ef=False,
                type=TargetType.CTRL_GE,
            )
        },
        get_rabi_param=lambda target: _make_rabi_param(target, 0.1),
        resolve_qubit_label=lambda label: "Q17" if label == "custom-target" else label,
        params=SimpleNamespace(
            get_control_amplitude=lambda qubit: 0.5 if qubit == "Q17" else None,
            get_ef_control_amplitude=lambda _qubit: None,
        ),
    )
    service = PulseService(experiment_context=cast(Any, ctx))

    assert service.rabi_params == {
        "custom-target": _make_rabi_param("custom-target", 0.1)
    }
    assert service.calc_control_amplitude("custom-target", rabi_rate=0.2) == 1.0


def test_ef_rabi_params_resolves_ge_labels_via_target_registry() -> None:
    """Given custom EF labels, when exposing EF Rabi params, then output keys use registry-resolved GE labels."""

    class _TargetRegistry:
        @staticmethod
        def resolve_ge_label(label: str) -> str:
            return "Q17" if label == "custom-ef-target" else label

    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_ge_label=lambda label: _TargetRegistry.resolve_ge_label(label),
    )

    ctx = SimpleNamespace(
        experiment_system=experiment_system,
        resolve_ge_label=lambda label: experiment_system.resolve_ge_label(label),
        ge_targets={},
        ef_targets={"custom-ef-target": object()},
        get_rabi_param=lambda target: _make_rabi_param(target, 0.1),
        targets={"custom-ef-target": SimpleNamespace(is_ef=True)},
    )
    service = PulseService(experiment_context=cast(Any, ctx))

    ef_params = service.ef_rabi_params

    assert list(ef_params.keys()) == ["Q17"]


def test_readout_accepts_renamed_ramp_parameters() -> None:
    """Given renamed ramp args, when readout is called, then pulse factory gets renamed args."""
    captured: dict[str, object] = {}
    expected = object()

    def _readout_pulse(**kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return expected

    ctx = SimpleNamespace(
        readout_duration=384.0,
        readout_pre_margin=32.0,
        readout_post_margin=128.0,
        measurement=SimpleNamespace(
            pulse_factory=SimpleNamespace(readout_pulse=_readout_pulse)
        ),
    )
    service = PulseService(experiment_context=cast(Any, ctx))

    result = service.readout(
        "RQ00",
        ramp_time=16.0,
        ramp_type="Bump",
    )

    kwargs = cast(dict[str, object], captured["kwargs"])
    assert result is expected
    assert kwargs["target"] == "RQ00"
    assert kwargs["ramp_time"] == 16.0
    assert kwargs["ramp_type"] == "Bump"


def test_readout_forwards_legacy_ramp_aliases() -> None:
    """Given legacy ramp args, when readout is called, then legacy args are forwarded to pulse factory."""
    captured: dict[str, object] = {}
    expected = object()

    def _readout_pulse(**kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return expected

    ctx = SimpleNamespace(
        readout_duration=384.0,
        readout_pre_margin=32.0,
        readout_post_margin=128.0,
        measurement=SimpleNamespace(
            pulse_factory=SimpleNamespace(readout_pulse=_readout_pulse)
        ),
    )
    service = PulseService(experiment_context=cast(Any, ctx))

    result = service.readout(
        "RQ00",
        ramptime=20.0,
        type="RaisedCosine",
    )

    kwargs = cast(dict[str, object], captured["kwargs"])
    assert result is expected
    assert kwargs["ramptime"] == 20.0
    assert kwargs["type"] == "RaisedCosine"


def test_readout_rejects_conflicting_ramp_aliases() -> None:
    """Given conflicting args, when readout is called, then pulse-factory error is propagated."""

    def _readout_pulse(**kwargs: object) -> object:
        if kwargs.get("ramp_time") is not None and kwargs.get("ramptime") is not None:
            raise ValueError("`ramptime` conflicts with `ramp_time`.")
        return object()

    ctx = SimpleNamespace(
        readout_duration=384.0,
        readout_pre_margin=32.0,
        readout_post_margin=128.0,
        measurement=SimpleNamespace(
            pulse_factory=SimpleNamespace(readout_pulse=_readout_pulse)
        ),
    )
    service = PulseService(experiment_context=cast(Any, ctx))

    with pytest.raises(ValueError, match="ramptime"):
        service.readout(
            "RQ00",
            ramp_time=16.0,
            ramptime=20.0,
        )

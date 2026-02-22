"""Tests for target-registry resolution in pulse service."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from qubex.backend import TargetType
from qubex.experiment.models.rabi_param import RabiParam
from qubex.experiment.services import pulse_service as pulse_service_module
from qubex.experiment.services.pulse_service import PulseService
from qubex.pulse import Arbitrary


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

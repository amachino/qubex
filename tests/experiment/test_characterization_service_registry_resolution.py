"""Tests for target-registry resolution in characterization service."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
from qxpulse import FlatTop

from qubex.experiment.models.result import Result
from qubex.experiment.services.characterization_service import CharacterizationService


def test_calibrate_ef_control_frequency_filters_targets_via_registry() -> None:
    """Given custom labels, when calibrating EF frequency, then EF availability is resolved via target registry."""

    class _TargetRegistry:
        @staticmethod
        def resolve_ef_label(label: str) -> str:
            return "Q17-ef" if label == "custom-target" else label

    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_ef_label=lambda label: _TargetRegistry.resolve_ef_label(label),
    )

    captured: dict[str, object] = {}

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q17"],
        experiment_system=experiment_system,
        resolve_ef_label=lambda label: experiment_system.resolve_ef_label(label),
        targets={"Q17-ef": object()},
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    def _obtain_freq_rabi_relation(
        self: CharacterizationService,
        *,
        targets: list[str],
        **_: object,
    ) -> Result:
        captured["targets"] = list(targets)
        return Result(
            data={
                "custom-target": SimpleNamespace(
                    fit=lambda: {"f_resonance": 5.123},
                )
            }
        )

    service.__dict__["obtain_freq_rabi_relation"] = MethodType(
        _obtain_freq_rabi_relation, service
    )

    result = service.calibrate_ef_control_frequency(
        targets=["custom-target"],
        plot=False,
        verbose=False,
    )

    assert captured["targets"] == ["custom-target"]
    assert result.data == {"custom-target": 5.123}


def test_measure_electrical_delay_resolves_labels_via_target_registry() -> None:
    """Given custom labels, when measuring electrical delay, then readout and qubit labels are resolved via target registry."""

    class _TargetRegistry:
        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            return "Q17" if label == "custom-target" else label

        @staticmethod
        def resolve_read_label(label: str) -> str:
            return "RQ17" if label == "custom-target" else label

    modified_calls: list[dict[str, float]] = []
    measure_calls: list[dict[str, object]] = []
    reset_calls: list[list[str] | None] = []

    @contextmanager
    def _modified_frequencies(
        frequencies: dict[str, float],
    ) -> Any:
        modified_calls.append(frequencies)
        yield

    def _measure(
        sequence: dict[str, np.ndarray],
        **kwargs: object,
    ) -> SimpleNamespace:
        measure_calls.append(
            {
                "sequence_labels": list(sequence.keys()),
                "readout_amplitudes": kwargs["readout_amplitudes"],
            }
        )
        return SimpleNamespace(
            data={"custom-target": SimpleNamespace(kerneled=np.complex128(1.0 + 0.0j))}
        )

    service = cast(Any, object.__new__(CharacterizationService))
    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_qubit_label=lambda label: _TargetRegistry.resolve_qubit_label(label),
        resolve_read_label=lambda label: _TargetRegistry.resolve_read_label(label),
        get_mux_by_qubit=lambda _qubit: SimpleNamespace(label="MUX0"),
        get_readout_box_for_qubit=lambda _qubit: SimpleNamespace(
            id="BOX0",
            traits=SimpleNamespace(readout_cnco_center=8.5e9),
        ),
    )
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=experiment_system,
        resolve_qubit_label=lambda label: experiment_system.resolve_qubit_label(label),
        resolve_read_label=lambda label: experiment_system.resolve_read_label(label),
        targets={
            "RQ17": SimpleNamespace(
                sideband="U",
                fine_frequency=6.0,
            )
        },
        reset_awg_and_capunits=lambda box_ids=None: reset_calls.append(box_ids),
        modified_frequencies=_modified_frequencies,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(measure=_measure)
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    tau = service.measure_electrical_delay(
        "custom-target",
        f_start=6.0,
        df=0.001,
        n_samples=3,
        plot=False,
        confirm=False,
    )

    assert np.isfinite(tau)
    assert reset_calls == [["BOX0"]]
    assert len(modified_calls) >= 3
    assert all(list(call.keys()) == ["RQ17"] for call in modified_calls)
    assert measure_calls[0]["sequence_labels"] == ["Q17"]
    assert measure_calls[0]["readout_amplitudes"] == {"Q17": 1.0}


def test_ckp_experiment_resolves_qubit_for_default_readout_amplitude() -> None:
    """Given custom labels, when CKP uses default readout amplitude, then qubit label resolution uses target registry."""

    class _TargetRegistry:
        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            return "Q17" if label == "custom-target" else label

    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_qubit_label=lambda label: _TargetRegistry.resolve_qubit_label(label),
    )

    captured: dict[str, object] = {}
    amplitude_calls: list[float] = []

    def _get_readout_amplitude(label: str) -> float:
        captured["qubit_label"] = label
        return 0.2

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=experiment_system,
        resolve_qubit_label=lambda label: experiment_system.resolve_qubit_label(label),
        params=SimpleNamespace(get_readout_amplitude=_get_readout_amplitude),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    def _ckp_measurement(
        self: CharacterizationService,
        *,
        qubit_initial_state: str,
        **kwargs: object,
    ) -> Result:
        amplitude_calls.append(cast(float, kwargs["resonator_drive_amplitude"]))
        fit_result = (
            {
                "popt": (1.0, 6.01, 0.02, 5.01),
                "gamma": 0.03,
                "A": 0.5,
                "C": 5.0,
                "f0": 6.01,
            }
            if qubit_initial_state == "0"
            else {
                "popt": (1.0, 6.03, 0.02, 5.03),
                "gamma": 0.03,
                "A": 0.5,
                "C": 5.02,
                "f0": 6.03,
            }
        )
        return Result(
            data={
                "resonator_frequency_range": np.array([6.0, 6.1]),
                "qubit_resonance_frequencies": np.array([5.0, 5.1]),
                "fit_result": fit_result,
            }
        )

    service.__dict__["ckp_measurement"] = MethodType(_ckp_measurement, service)

    _ = service.ckp_experiment(
        target="custom-target",
        qubit_pi_pulse=FlatTop(duration=64, amplitude=0.1, tau=16),
        plot=False,
        verbose=False,
        save_image=False,
    )

    assert captured["qubit_label"] == "Q17"
    assert amplitude_calls == [0.2, 0.2]

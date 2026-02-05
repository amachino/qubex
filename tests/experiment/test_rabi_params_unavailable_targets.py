from __future__ import annotations

from dataclasses import dataclass

import pytest

from qubex.backend.system_manager import SystemManager
from qubex.backend.target import TargetType
from qubex.experiment.experiment import Experiment


@dataclass
class DummyTarget:
    label: str
    type: TargetType
    is_available: bool

    @property
    def is_ge(self) -> bool:
        return self.type == TargetType.CTRL_GE

    @property
    def is_ef(self) -> bool:
        return self.type == TargetType.CTRL_EF

    @property
    def is_cr(self) -> bool:
        return self.type == TargetType.CTRL_CR

    def is_related_to_qubits(self, qubits: list[str]) -> bool:
        return self.label in qubits


class DummyControlParams:
    def __init__(self, control_amplitudes: dict[str, float]):
        self._control_amplitudes = control_amplitudes

    def get_control_amplitude(self, qubit: str) -> float | None:
        return self._control_amplitudes.get(qubit)

    def get_ef_control_amplitude(self, qubit: str) -> float | None:
        return self._control_amplitudes.get(qubit)


class DummyExperimentSystem:
    def __init__(self, targets: list[DummyTarget], control_params: DummyControlParams):
        self.targets = targets
        self.control_params = control_params


class DummyCalibrationNote:
    def __init__(self, rabi_params: dict[str, dict]):
        self._rabi_params = rabi_params

    def get_rabi_param(
        self, target: str, *, valid_days: int | None = None
    ) -> dict | None:
        _ = valid_days
        return self._rabi_params.get(target)


def test_calc_control_amplitude_works_for_unavailable_target(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression for #183.

    Even when a target is not "available" (e.g., AWG frequency window mismatch),
    stored Rabi parameters should still be usable for computing control amplitude.
    """

    # Isolate SystemManager singleton for this test.
    monkeypatch.setattr(SystemManager, "_instance", None)

    system_manager = SystemManager.shared()
    system_manager.__dict__["_experiment_system"] = DummyExperimentSystem(
        targets=[
            DummyTarget(label="Q057", type=TargetType.CTRL_GE, is_available=False),
        ],
        control_params=DummyControlParams({"Q057": 0.020}),
    )

    exp = Experiment.__new__(Experiment)
    exp.__dict__["_qubits"] = ["Q057"]
    exp.__dict__["_chip_id"] = "dummy"
    exp.__dict__["_calib_note"] = DummyCalibrationNote(
        {
            "Q057": {
                "target": "Q057",
                "frequency": 0.004,
                "amplitude": 0.03,
                "phase": 0.0,
                "offset": 0.0,
                "noise": 0.0,
                "angle": 0.0,
                "distance": 0.0,
                "r2": 0.99,
                "reference_phase": 0.0,
            }
        }
    )
    exp.__dict__["_calibration_valid_days"] = 365

    # With the dummy target set to unavailable, ge_targets is empty by design.
    assert exp.ge_targets == {}

    # But rabi_params should still pick up the stored params for related targets.
    assert "Q057" in exp.rabi_params

    amp = exp.calc_control_amplitude("Q057", 0.0125)

    # rabi_amplitude_ratio = rabi_param.frequency / default_amplitude = 0.004 / 0.020 = 0.2
    # amp = rabi_rate / ratio = 0.0125 / 0.2 = 0.0625
    assert amp == pytest.approx(0.0625)

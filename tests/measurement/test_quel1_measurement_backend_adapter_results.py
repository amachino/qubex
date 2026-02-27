"""Tests for QuEL-1 adapter result conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.testing import assert_allclose

from qubex.backend.quel1 import Quel1BackendExecutionResult
from qubex.measurement.adapters.backend_adapter import Quel1MeasurementBackendAdapter
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.typing import MeasurementMode


@dataclass
class _Target:
    sideband: str


@dataclass
class _ExperimentSystemStub:
    sideband_by_target: dict[str, str]

    def get_target(self, target: str) -> _Target:
        return _Target(sideband=self.sideband_by_target[target])


def _make_config(*, mode: MeasurementMode, shots: int) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=shots,
        shot_interval=100.0,
        shot_averaging=(mode == "avg"),
        time_integration=False,
        state_classification=False,
    )


def test_build_measurement_result_converts_single_mode_to_qubit_labels() -> None:
    """Given QuEL-1 raw result, when converting single mode, then qubit labels and normalization are applied."""
    norm_factor = 2 ** (-32)
    backend_result = Quel1BackendExecutionResult(
        status={},
        data={
            "RQ00": [
                np.array([9.0 + 3.0j], dtype=np.complex128),
                np.array([4.0 + 2.0j], dtype=np.complex128),
            ]
        },
        config={},
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, object()),
        experiment_system=cast(
            Any,
            _ExperimentSystemStub(sideband_by_target={"RQ00": "L"}),
        ),
    )

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(mode="single", shots=4),
        device_config={"kind": "quel1"},
        sampling_period=2.0,
    )

    assert result.measurement_config.shot_averaging is False
    assert result.device_config == {"kind": "quel1"}
    assert set(result.data.keys()) == {"Q00"}
    assert len(result.data["Q00"]) == 1
    assert result.data["Q00"][0].sampling_period == 2.0
    assert_allclose(
        result.data["Q00"][0],
        np.array([4.0 - 2.0j], dtype=np.complex128) * norm_factor,
    )


def test_build_measurement_result_converts_avg_mode_with_shot_scaling() -> None:
    """Given QuEL-1 raw result, when converting avg mode, then waveform is shot-normalized and squeezed."""
    norm_factor = 2 ** (-32)
    backend_result = Quel1BackendExecutionResult(
        status={},
        data={
            "RQ00": [
                np.array([[1.0 + 0.0j, 2.0 + 0.0j]], dtype=np.complex128),
                np.array([[8.0 + 4.0j, 12.0 + 6.0j]], dtype=np.complex128),
            ]
        },
        config={},
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, object()),
        experiment_system=cast(
            Any,
            _ExperimentSystemStub(sideband_by_target={"RQ00": "U"}),
        ),
    )

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(mode="avg", shots=4),
        device_config={"kind": "quel1"},
        sampling_period=2.0,
    )

    assert result.measurement_config.shot_averaging is True
    assert set(result.data.keys()) == {"Q00"}
    assert len(result.data["Q00"]) == 1
    assert_allclose(
        result.data["Q00"][0],
        np.array([2.0 + 1.0j, 3.0 + 1.5j], dtype=np.complex128) * norm_factor,
    )

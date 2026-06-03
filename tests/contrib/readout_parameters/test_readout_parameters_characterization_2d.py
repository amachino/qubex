"""Tests for readout parameter 2D characterization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from qubex.contrib.experiment import readout_parameters_characterization as rpc
from qubex.contrib.experiment.readout_parameters_characterization import (
    characterize_readout_parameters_2d,
)


class _ExperimentStub:
    def __init__(self) -> None:
        self.qubit_labels = ["Q00"]
        self.calls: list[dict[str, Any]] = []
        self.ctx = SimpleNamespace(
            params=SimpleNamespace(
                control_amplitude={"Q00": 0.1},
                readout_amplitude={"Q00": 0.2},
            ),
            resonators={"Q00": SimpleNamespace(label="RQ00", frequency=5.0)},
            resolve_qubit_label=lambda label: label,
        )

    def rabi_experiment(self, **kwargs: Any) -> SimpleNamespace:
        readout_amplitude = self.ctx.params.readout_amplitude["Q00"]
        readout_frequency = kwargs["frequencies"]["RQ00"]
        self.calls.append(
            {
                "kwargs": kwargs,
                "readout_amplitude": readout_amplitude,
                "readout_frequency": readout_frequency,
            }
        )
        response_range = readout_amplitude + 10 * (readout_frequency - 5.0)
        return SimpleNamespace(
            data={
                "Q00": SimpleNamespace(
                    data=np.array([0.0, response_range], dtype=np.float64)
                )
            }
        )


class _TqdmStub:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.updated = 0

    def __enter__(self) -> _TqdmStub:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def update(self, n: int = 1) -> None:
        self.updated += n


def test_characterize_readout_parameters_2d_builds_heatmap_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given amplitude and frequency ranges, when Rabi is measured, then heatmap data is returned."""
    exp = _ExperimentStub()
    frequency_range = np.array([5.0, 5.1, 5.2], dtype=np.float64)
    readout_amplitudes = np.array([0.01, 0.02], dtype=np.float64)
    progress_bars: list[_TqdmStub] = []

    def _tqdm_stub(**kwargs: Any) -> _TqdmStub:
        progress = _TqdmStub(**kwargs)
        progress_bars.append(progress)
        return progress

    monkeypatch.setattr(rpc, "tqdm", _tqdm_stub)

    result = characterize_readout_parameters_2d(
        exp,  # type: ignore[arg-type]
        frequency_range=frequency_range,
        readout_amplitudes=readout_amplitudes,
        time_range=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        n_shots=128,
        shot_interval=2.0,
        plot=False,
        save_image=False,
    )

    assert len(exp.calls) == 6
    assert len(progress_bars) == 1
    assert progress_bars[0].kwargs["total"] == 6
    assert progress_bars[0].kwargs["desc"] == "readout rabi Q00"
    assert progress_bars[0].updated == 6
    assert exp.calls[0]["readout_amplitude"] == 0.01
    assert exp.calls[0]["readout_frequency"] == 5.0
    assert exp.calls[0]["kwargs"]["amplitudes"] == {"Q00": 0.1}
    assert exp.calls[0]["kwargs"]["frequencies"] == {"RQ00": 5.0}
    assert exp.calls[0]["kwargs"]["n_shots"] == 128
    assert exp.calls[0]["kwargs"]["shot_interval"] == 2.0
    assert exp.calls[0]["kwargs"]["plot"] is False
    assert exp.ctx.params.readout_amplitude == {"Q00": 0.2}

    heatmap_data = cast(NDArray[np.float64], result.data["heatmap_data"])
    np.testing.assert_allclose(
        heatmap_data,
        [[0.01, 1.01, 2.01], [0.02, 1.02, 2.02]],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cast(NDArray[np.float64], result.data["frequency_range"]),
        frequency_range,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        cast(NDArray[np.float64], result.data["readout_amplitudes"]),
        readout_amplitudes,
        rtol=0,
        atol=0,
    )
    assert result.data["optimal_readout_frequency"] == 5.2
    assert result.data["optimal_readout_amplitude"] == 0.02
    np.testing.assert_allclose(
        result.data["optimal_response_range"],
        2.02,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.data["optimal_rabi_amplitude"],
        result.data["optimal_response_range"],
        rtol=0,
        atol=0,
    )
    assert result.figure is not None


def test_characterize_readout_parameters_2d_uses_current_settings_for_defaults() -> (
    None
):
    """Given no sweep ranges, when Rabi is measured, then current settings define the grid."""
    exp = _ExperimentStub()

    result = characterize_readout_parameters_2d(
        exp,  # type: ignore[arg-type]
        plot=False,
        save_image=False,
    )

    frequency_range = cast(NDArray[np.float64], result.data["frequency_range"])
    detuning_range = cast(NDArray[np.float64], result.data["detuning_range"])
    readout_amplitudes = cast(NDArray[np.float64], result.data["readout_amplitudes"])
    assert len(exp.calls) == 77
    assert exp.calls[0]["kwargs"]["n_shots"] == 256
    np.testing.assert_allclose(
        frequency_range,
        np.linspace(4.95, 5.05, 11),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        detuning_range,
        np.linspace(-0.05, 0.05, 11),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        readout_amplitudes,
        np.linspace(0.0, 0.2, 7),
        rtol=0,
        atol=1e-12,
    )
    assert result.data["optimal_readout_frequency"] == 5.05
    np.testing.assert_allclose(
        result.data["optimal_readout_amplitude"],
        0.2,
        rtol=0,
        atol=1e-12,
    )
    assert exp.ctx.params.readout_amplitude == {"Q00": 0.2}


def test_characterize_readout_parameters_2d_clips_default_amplitudes_to_unit_range() -> (
    None
):
    """Given high current amplitude, when defaults are used, then amplitudes stay within [0, 1]."""
    exp = _ExperimentStub()
    exp.ctx.params.readout_amplitude["Q00"] = 0.8

    result = characterize_readout_parameters_2d(
        exp,  # type: ignore[arg-type]
        frequency_range=np.array([5.0], dtype=np.float64),
        plot=False,
        save_image=False,
    )

    readout_amplitudes = cast(NDArray[np.float64], result.data["readout_amplitudes"])
    np.testing.assert_allclose(
        readout_amplitudes,
        np.linspace(0.0, 0.8, 7),
        rtol=0,
        atol=1e-12,
    )
    assert np.all(readout_amplitudes >= 0.0)
    assert np.all(readout_amplitudes <= 1.0)
    assert exp.ctx.params.readout_amplitude == {"Q00": 0.8}


def test_characterize_readout_parameters_2d_rejects_out_of_range_amplitudes() -> None:
    """Given out-of-range amplitudes, when scanning, then a value error is raised."""
    exp = _ExperimentStub()

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        characterize_readout_parameters_2d(
            exp,  # type: ignore[arg-type]
            frequency_range=np.array([5.0], dtype=np.float64),
            readout_amplitudes=np.array([-0.1, 0.5, 1.1], dtype=np.float64),
            plot=False,
            save_image=False,
        )

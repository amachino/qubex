"""Tests for Quel3 backend controller measurement execution."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from qubex.backend.quel3 import Quel3BackendController
from qubex.measurement.adapters import (
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3TargetTimeline,
)
from qubex.measurement.models.measurement_result import MeasurementResult


def _make_payload(*, mode: str = "avg", repeats: int = 2) -> Quel3ExecutionPayload:
    timeline = Quel3TargetTimeline(
        sampling_period_ns=0.4,
        waveform=np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
        capture_windows=(
            Quel3CaptureWindow(name="capture_0", start_offset_ns=0.4, length_ns=0.4),
        ),
        length_ns=0.8,
    )
    return Quel3ExecutionPayload(
        timelines={"RQ00": timeline},
        instrument_aliases={"RQ00": "alias-rq00"},
        output_target_labels={"RQ00": "Q00"},
        interval_ns=100.0,
        repeats=repeats,
        mode=mode,
        dsp_demodulation=True,
        enable_sum=False,
        enable_classification=False,
        line_param0=(1.0, 0.0, 0.0),
        line_param1=(0.0, 1.0, 0.0),
    )


def test_resolve_instrument_alias_uses_alias_map() -> None:
    """Given alias map, when resolving alias, then mapped alias is returned."""
    controller = Quel3BackendController(alias_map={"RQ00": "inst-00"})

    assert controller.resolve_instrument_alias("RQ00") == "inst-00"
    assert controller.resolve_instrument_alias("RQ01") == "RQ01"


def test_update_instrument_alias_map_overrides_target_alias() -> None:
    """Given alias-map update, when resolving alias, then updated alias is returned."""
    controller = Quel3BackendController(alias_map={"RQ00": "inst-00"})
    controller.update_instrument_alias_map({"RQ00": "inst-00-new", "RQ01": "inst-01"})

    assert controller.resolve_instrument_alias("RQ00") == "inst-00-new"
    assert controller.resolve_instrument_alias("RQ01") == "inst-01"


def test_execute_measurement_rejects_non_quel3_payload() -> None:
    """Given non-Quel3 payload, when executing, then TypeError is raised."""
    controller = Quel3BackendController()

    with pytest.raises(TypeError, match="Quel3ExecutionPayload"):
        controller.execute_measurement(payload=object())


def test_execute_measurement_surfaces_missing_quelware_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given missing quelware dependency, when executing, then RuntimeError is raised."""
    controller = Quel3BackendController()
    payload = _make_payload()

    monkeypatch.setattr(
        controller,
        "_load_quelware_api",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError("quelware_client")),
    )

    with pytest.raises(RuntimeError, match="quelware-client is not available"):
        controller.execute_measurement(payload=payload)


def test_build_measurement_result_averages_shot_samples() -> None:
    """Given shot samples, when mode is avg, then samples are averaged per capture."""
    payload = _make_payload(mode="avg", repeats=2)
    shot_samples = {
        "RQ00": {
            "capture_0": [
                np.array([1.0 + 1.0j, 3.0 + 3.0j], dtype=np.complex128),
                np.array([3.0 + 3.0j, 5.0 + 5.0j], dtype=np.complex128),
            ]
        }
    }

    result = cast(
        MeasurementResult,
        Quel3BackendController._build_measurement_result(  # noqa: SLF001
            payload=payload,
            shot_samples=shot_samples,
            sampling_period_ns=0.4,
        ),
    )

    assert result.mode == "avg"
    assert "Q00" in result.data
    assert np.array_equal(
        result.data["Q00"][0],
        np.array([2.0 + 2.0j, 4.0 + 4.0j], dtype=np.complex128),
    )
    assert result.sampling_period_ns == 0.4


def test_build_measurement_result_uses_output_target_labels() -> None:
    """Given explicit output map, when building result, then mapped label is used."""
    payload = _make_payload(mode="single", repeats=1)
    timeline = payload.timelines["RQ00"]
    payload = replace(
        payload,
        timelines={"raw-target": timeline},
        instrument_aliases={"raw-target": "alias-rq00"},
        output_target_labels={"raw-target": "Q17"},
    )
    shot_samples = {
        "raw-target": {
            "capture_0": [
                np.array([7.0 + 0.0j], dtype=np.complex128),
            ]
        }
    }

    result = cast(
        MeasurementResult,
        Quel3BackendController._build_measurement_result(  # noqa: SLF001
            payload=payload,
            shot_samples=shot_samples,
            sampling_period_ns=0.4,
        ),
    )

    assert "Q17" in result.data
    assert "raw-target" not in result.data


def test_constructor_uses_builtin_quelware_defaults_ignoring_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quelware env vars, constructor still uses builtin default values."""
    monkeypatch.setenv("QUBEX_QUELWARE_ENDPOINT", "env-host")
    monkeypatch.setenv("QUBEX_QUELWARE_PORT", "12345")
    monkeypatch.setenv("QUBEX_QUELWARE_TRIGGER_WAIT", "999")

    controller = Quel3BackendController()

    assert pytest.approx(0.4) == controller.DEFAULT_SAMPLING_PERIOD
    assert controller._quelware_endpoint == "localhost"  # noqa: SLF001
    assert controller._quelware_port == 50051  # noqa: SLF001
    assert controller._trigger_wait == 1_000_000  # noqa: SLF001

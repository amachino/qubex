# ruff: noqa: SLF001

"""Tests for QuEL-3 backend controller behavior."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from qubex.backend import BackendExecutionRequest
from qubex.backend.backend_controller import BackendController
from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import (
    Quel3BackendController,
    Quel3BackendExecutionResult,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
    Quel3WaveformEvent,
)
from qubex.backend.quel3.managers.execution_manager import Quel3ExecutionManager


def _make_payload(*, mode: str = "avg", repeats: int = 2) -> Quel3ExecutionPayload:
    waveform_name = "wf0"
    timeline = Quel3FixedTimeline(
        events=(
            Quel3WaveformEvent(
                waveform_name=waveform_name,
                start_offset_ns=0.0,
            ),
        ),
        capture_windows=(
            Quel3CaptureWindow(name="capture_0", start_offset_ns=0.4, length_ns=0.4),
        ),
        length_ns=0.8,
    )
    return Quel3ExecutionPayload(
        waveform_library={
            waveform_name: Quel3Waveform(
                iq_array=np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.4,
            )
        },
        fixed_timelines={"alias-rq00": timeline},
        interval_ns=100.0,
        repeats=repeats,
        mode=mode,
    )


def test_quel_controllers_implement_backend_controller_contract() -> None:
    """Given QuEL controllers, both satisfy BackendController protocol."""
    assert isinstance(Quel1BackendController(), BackendController)
    assert isinstance(Quel3BackendController(), BackendController)


def test_quel3_controller_is_not_quel1_subclass() -> None:
    """Given QuEL-3 controller, it is not a QuEL-1 subclass."""
    assert not isinstance(Quel3BackendController(), Quel1BackendController)


def test_quel3_constructor_rejects_config_path_argument() -> None:
    """Given legacy config_path kwarg, constructor raises TypeError."""
    with pytest.raises(TypeError, match="config_path"):
        cast(Any, Quel3BackendController)(config_path="dummy")


def test_quel3_constructor_rejects_sampling_period_override_argument() -> None:
    """Given legacy sampling-period override kwarg, constructor raises TypeError."""
    with pytest.raises(TypeError, match="sampling_period_ns"):
        cast(Any, Quel3BackendController)(sampling_period_ns=0.8)


def test_quel3_constructor_rejects_alias_map_argument() -> None:
    """Given legacy alias-map kwarg, constructor raises TypeError."""
    with pytest.raises(TypeError, match="alias_map"):
        cast(Any, Quel3BackendController)(alias_map={"RQ00": "inst-00"})


def test_execute_rejects_non_quel3_payload() -> None:
    """Given non-QuEL-3 payload, execute raises TypeError."""
    controller = Quel3BackendController()

    with pytest.raises(TypeError, match="Quel3ExecutionPayload"):
        asyncio.run(
            controller.execute(request=BackendExecutionRequest(payload=object()))
        )


def test_execute_surfaces_missing_quelware_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given missing quelware dependency, execute raises RuntimeError."""
    controller = Quel3BackendController()
    payload = _make_payload()

    monkeypatch.setattr(
        Quel3ExecutionManager,
        "_load_quelware_api",
        staticmethod(
            lambda: (_ for _ in ()).throw(ModuleNotFoundError("quelware_client"))
        ),
    )

    with pytest.raises(RuntimeError, match="quelware-client is not available"):
        asyncio.run(
            controller.execute(request=BackendExecutionRequest(payload=payload))
        )


def test_build_measurement_result_averages_shot_samples() -> None:
    """Given avg mode shots, result samples are averaged."""
    payload = _make_payload(mode="avg", repeats=2)
    shot_samples = {
        "alias-rq00": {
            "capture_0": [
                np.array([1.0 + 1.0j, 3.0 + 3.0j], dtype=np.complex128),
                np.array([3.0 + 3.0j, 5.0 + 5.0j], dtype=np.complex128),
            ]
        }
    }

    result = Quel3ExecutionManager._build_measurement_result(
        payload=payload,
        shot_samples=shot_samples,
        sampling_period_ns=0.4,
        backend_sampling_period=0.4,
        capture_decimation_factor=4,
    )

    assert isinstance(result, Quel3BackendExecutionResult)
    assert result.mode == "avg"
    assert "alias-rq00" in result.data
    assert np.array_equal(
        result.data["alias-rq00"][0],
        np.array([2.0 + 2.0j, 4.0 + 4.0j], dtype=np.complex128),
    )
    assert result.sampling_period_ns == 1.6


def test_build_measurement_result_keeps_backend_alias_labels() -> None:
    """Given backend flow result, measurement labels remain instrument aliases."""
    payload = _make_payload(mode="single", repeats=1)
    timeline = payload.fixed_timelines["alias-rq00"]
    payload = replace(
        payload,
        fixed_timelines={"alias-raw": timeline},
    )
    shot_samples = {
        "alias-raw": {
            "capture_0": [
                np.array([7.0 + 0.0j], dtype=np.complex128),
            ]
        }
    }

    result = Quel3ExecutionManager._build_measurement_result(
        payload=payload,
        shot_samples=shot_samples,
        sampling_period_ns=0.4,
        backend_sampling_period=0.4,
        capture_decimation_factor=4,
    )

    assert isinstance(result, Quel3BackendExecutionResult)
    assert "alias-raw" in result.data


def test_extract_capture_samples_from_waveform_result_container() -> None:
    """Given waveform-style iq_result, extraction returns latest waveform samples."""

    class _Waveform:
        def __init__(self, values: np.ndarray) -> None:
            self.iq_array = values

    class _Result:
        def __init__(self) -> None:
            self.iq_result = {
                "RQ00:0": [
                    _Waveform(np.array([1.0 + 0.0j], dtype=np.complex128)),
                    _Waveform(np.array([2.0 + 0.0j], dtype=np.complex128)),
                ]
            }

    values = Quel3ExecutionManager._extract_capture_samples(_Result(), "RQ00:0")

    assert values is not None
    assert np.array_equal(values, np.array([2.0 + 0.0j], dtype=np.complex128))


def test_extract_capture_samples_from_point_result_container() -> None:
    """Given point-style iq_result, extraction returns complex-point array."""

    class _Result:
        def __init__(self) -> None:
            self.iq_result = {
                "RQ00:0": [1.0 + 2.0j, 3.0 + 4.0j],
            }

    values = Quel3ExecutionManager._extract_capture_samples(_Result(), "RQ00:0")

    assert values is not None
    assert np.array_equal(
        values,
        np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex128),
    )


def test_constructor_uses_builtin_quelware_defaults_ignoring_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quelware env vars, constructor still uses builtin defaults."""
    monkeypatch.setenv("QUBEX_QUELWARE_ENDPOINT", "env-host")
    monkeypatch.setenv("QUBEX_QUELWARE_PORT", "12345")

    controller = Quel3BackendController()

    assert pytest.approx(0.4) == controller.sampling_period
    assert controller._connection_manager.quelware_endpoint == "localhost"
    assert controller._connection_manager.quelware_port == 50051


def test_execute_rejects_multiple_instrument_aliases() -> None:
    """Given multiple instrument timelines, execute raises NotImplementedError."""
    controller = Quel3BackendController()
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={
            "alias-0": payload.fixed_timelines["alias-rq00"],
            "alias-1": payload.fixed_timelines["alias-rq00"],
        },
    )

    with pytest.raises(NotImplementedError, match="single instrument alias"):
        asyncio.run(
            controller.execute(request=BackendExecutionRequest(payload=payload))
        )

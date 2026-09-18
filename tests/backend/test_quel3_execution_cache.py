"""Execution behavior for explicit QuEL-3 instrument caches."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.managers.execution_manager import Quel3ExecutionManager
from qubex.backend.quel3.models import (
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
)


@pytest.mark.parametrize("batched", [False, True])
def test_execution_rejects_missing_alias_before_loading_runtime(
    monkeypatch: pytest.MonkeyPatch,
    batched: bool,
) -> None:
    """Every requested alias should be cached before any batch payload starts."""
    manager = Quel3ExecutionManager(
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    payload = Quel3ExecutionPayload(
        waveform_library={},
        fixed_timelines={
            "RQ00": Quel3FixedTimeline(
                events=(),
                capture_windows=(
                    Quel3CaptureWindow(
                        name="readout", start_offset_ns=0.0, length_ns=0.4
                    ),
                ),
                length_ns=0.4,
            )
        },
        n_iterations=1,
        shot_interval_ns=100.0,
        capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
    )

    instrument_cache = InstrumentCache()
    instrument_cache.replace_all(
        instrument_infos=cast(
            Any,
            (
                SimpleNamespace(
                    id="unit-a:existing-instrument",
                    port_id="unit-a:trx_p00",
                    definition=SimpleNamespace(alias="RQ01"),
                ),
            ),
        )
    )

    def fail_runtime_load() -> None:
        pytest.fail("Runtime loading must follow cache validation.")

    monkeypatch.setattr(manager, "_load_quelware_api", fail_runtime_load)

    if batched:
        cached_payload = replace(
            payload, fixed_timelines={"RQ01": payload.fixed_timelines["RQ00"]}
        )
        execution = manager.execute_batch_async(
            requests=[
                BackendExecutionRequest(payload=cached_payload),
                BackendExecutionRequest(payload=payload),
            ],
            instrument_cache=instrument_cache,
        )
    else:
        execution = manager.execute_async(
            request=BackendExecutionRequest(payload=payload),
            instrument_cache=instrument_cache,
        )

    with pytest.raises(ValueError, match=r"RQ00.*not cached"):
        asyncio.run(execution)


def test_execution_rejects_duplicate_capture_names_before_loading_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate capture names on one alias should fail before creating a session."""
    instrument_cache = InstrumentCache()
    instrument_cache.replace_all(
        instrument_infos=cast(
            Any,
            (
                SimpleNamespace(
                    id="unit-a:instrument",
                    port_id="unit-a:trx_p00",
                    definition=SimpleNamespace(alias="RQ00"),
                ),
            ),
        )
    )
    timeline = Quel3FixedTimeline(
        events=(),
        capture_windows=(
            Quel3CaptureWindow(name="readout", start_offset_ns=0.0, length_ns=0.4),
            Quel3CaptureWindow(name="readout", start_offset_ns=0.4, length_ns=0.4),
        ),
        length_ns=0.8,
    )
    payload = Quel3ExecutionPayload(
        waveform_library={},
        fixed_timelines={"RQ00": timeline},
        n_iterations=1,
        shot_interval_ns=100.0,
        capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
    )
    manager = Quel3ExecutionManager(sampling_period_ns=0.4, capture_decimation_factor=4)

    def fail_runtime_load() -> None:
        pytest.fail("Capture validation must precede runtime loading.")

    monkeypatch.setattr(manager, "_load_quelware_api", fail_runtime_load)

    with pytest.raises(ValueError, match=r"Duplicate capture.*readout.*RQ00"):
        asyncio.run(
            manager.execute_async(
                request=BackendExecutionRequest(payload=payload),
                instrument_cache=instrument_cache,
            )
        )


def test_capture_names_are_scoped_to_instrument_aliases() -> None:
    """Separate instrument aliases should accept the same capture name."""
    instrument_cache = InstrumentCache()
    instrument_cache.replace_all(
        instrument_infos=cast(
            Any,
            tuple(
                SimpleNamespace(
                    id=f"unit-a:instrument-{index}",
                    port_id=f"unit-a:trx_p0{index}",
                    definition=SimpleNamespace(alias=alias),
                )
                for index, alias in enumerate(("RQ00", "RQ01"))
            ),
        )
    )
    window = Quel3CaptureWindow(name="readout", start_offset_ns=0.0, length_ns=0.4)
    timeline = Quel3FixedTimeline(events=(), capture_windows=(window,), length_ns=0.4)
    payload = Quel3ExecutionPayload(
        waveform_library={},
        fixed_timelines={"RQ00": timeline, "RQ01": timeline},
        n_iterations=1,
        shot_interval_ns=100.0,
        capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
    )

    plan = Quel3ExecutionManager._prepare_payload_execution_plan(  # noqa: SLF001
        payload=payload, instrument_cache=instrument_cache
    )

    assert plan.aliases == ("RQ00", "RQ01")
    assert plan.payload.fixed_timelines["RQ00"].capture_windows == (window,)
    assert plan.payload.fixed_timelines["RQ01"].capture_windows == (window,)


def test_cached_hardware_info_retains_actual_quelware_driver_configuration() -> None:
    """Cached hardware info should construct a real driver with its original config."""
    entities = pytest.importorskip("quelware_core.entities.instrument")
    drivers = pytest.importorskip("quelware_client.core.instrument_driver")
    config = entities.FixedTimelineConfig(
        sampling_period_fs=400_000,
        bitdepth=16,
        timeline_step_samples=64,
        samples_per_tick=8,
    )
    info = entities.InstrumentInfo(
        id="unit-a:readout-id",
        port_id="unit-a:trx_p00",
        definition=entities.InstrumentDefinition(
            alias="unit-a:RQ00",
            mode=entities.InstrumentMode.FIXED_TIMELINE,
            role=entities.InstrumentRole.TRANSCEIVER,
            profile=entities.FixedTimelineProfile(
                frequency_range_min=6.0e9,
                frequency_range_max=6.2e9,
            ),
        ),
        config=config,
    )
    instrument_cache = InstrumentCache()
    instrument_cache.replace_all(instrument_infos=(info,))
    requested_units: list[str] = []
    agent = object()

    def instrument_agent(unit_label: str) -> object:
        requested_units.append(unit_label)
        return agent

    session = SimpleNamespace(
        token="test-session",  # noqa: S106
        available_resource_ids={info.id},
        agent_container=SimpleNamespace(instrument=instrument_agent),
    )

    cached = instrument_cache.get("RQ00")
    driver = drivers.create_instrument_driver_fixed_timeline(session, cached)

    assert cached is info
    assert driver.instrument_config is config
    assert requested_units == ["unit-a"]

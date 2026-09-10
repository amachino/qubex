"""Single-shot batch acquisition contracts."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.contrib.experiment._single_shot_batch import measure_single_shot_batch
from qubex.pulse import Blank, PulseSchedule


@pytest.mark.parametrize("in_event_loop", [False, True])
def test_batch_preserves_shots_and_copies_input(in_event_loop, monkeypatch):
    """A synchronous batch should preserve shot order inside and outside notebooks."""
    monkeypatch.setattr(
        "qubex.contrib.experiment._single_shot_batch.DEFAULT_TIMEOUT_SECONDS", 2.0
    )
    calls = []
    iq = np.array([1 + 2j, 3 + 4j])

    async def sweep(schedule, **options):
        calls.append(options)
        point = schedule(0)
        point.add("Q0", Blank(10))
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    data={
                        "Q0": [SimpleNamespace(data=iq)],
                    }
                )
            ]
        )

    exp = SimpleNamespace(
        pulse=SimpleNamespace(
            readout_duration=10, readout_pre_margin=2, readout_post_margin=2
        ),
        measurement_service=SimpleNamespace(run_sweep_measurement=sweep),
    )
    schedule = PulseSchedule(["Q0"])
    schedule.add("Q0", Blank(4))

    def run():
        return measure_single_shot_batch(
            cast(Any, exp), [schedule], n_shots=2, shot_interval=100
        )

    async def notebook():
        return run()

    result = asyncio.run(notebook()) if in_event_loop else run()
    np.testing.assert_array_equal(result[0]["Q0"], iq)
    assert schedule.duration == 4
    assert len(calls) == 1
    assert calls[0]["shot_averaging"] is False
    assert calls[0]["time_integration"] is True
    assert calls[0]["state_classification"] is False
    assert calls[0]["enable_tqdm"] is False
    assert calls[0]["plot"] is False


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "number of points"),
        ([SimpleNamespace(data={"Q0": []})], "one capture"),
        (
            [SimpleNamespace(data={"Q0": [SimpleNamespace(data=np.array([1]))]})],
            "finite IQ shots",
        ),
        (
            [
                SimpleNamespace(
                    data={"Q0": [SimpleNamespace(data=np.array([1, np.nan]))]}
                )
            ],
            "finite IQ shots",
        ),
        (
            [SimpleNamespace(data={"Q0": [SimpleNamespace(data=np.ones((2, 2)))]})],
            "finite IQ shots",
        ),
    ],
)
def test_batch_rejects_incomplete_or_invalid_shots(payload, message):
    """Invalid acquisition payloads should fail before statistical analysis."""

    async def sweep(*args, **kwargs):
        return SimpleNamespace(results=payload)

    exp = SimpleNamespace(
        pulse=SimpleNamespace(
            readout_duration=0, readout_pre_margin=0, readout_post_margin=0
        ),
        measurement_service=SimpleNamespace(run_sweep_measurement=sweep),
    )
    with pytest.raises(ValueError, match=message):
        measure_single_shot_batch(
            cast(Any, exp),
            [PulseSchedule(["Q0"])],
            n_shots=2,
            shot_interval=100,
        )

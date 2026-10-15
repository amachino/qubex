"""Tests for classifier validation in randomized benchmarking."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from qxpulse import PulseSchedule

from qubex.core.async_bridge import AsyncBridge
from qubex.experiment.services import benchmarking_service as benchmarking_module
from qubex.experiment.services.benchmarking_service import BenchmarkingService


@pytest.mark.parametrize("missing_classifier", ["Q17", "Q18"])
def test_randomized_benchmarking_2q_checks_classifiers_before_measurement(
    missing_classifier: str,
) -> None:
    """Given a missing 2Q classifier, when RB starts, then it fails before measurement."""
    measurement_calls: list[object] = []
    classifiers = {
        qubit: object() for qubit in ("Q17", "Q18") if qubit != missing_classifier
    }

    def measure(**_kwargs: object) -> None:
        measurement_calls.append(object())
        raise ValueError(f"Classifier not found for {missing_classifier}.")

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        classifiers=classifiers,
        state_centers={"Q17": {0: 0j}, "Q18": {0: 0j}},
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(is_cr=True)
        ),
        calib_note=SimpleNamespace(cr_params={"CR17-18": object()}),
        cr_pair=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(measure=measure)
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["rb_sequence_2q"] = lambda **_kwargs: PulseSchedule(["Q17", "Q18"])

    with pytest.raises(
        ValueError,
        match=rf"^Classifier not found for {missing_classifier}\.$",
    ):
        service.randomized_benchmarking(
            targets="CR17-18",
            n_cliffords_range=[0],
            n_trials=1,
            seeds=[0],
            plot=False,
            save_image=False,
        )

    assert measurement_calls == []


@pytest.fixture
def rb_service(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Provide a real RB bridge with hardware operations replaced by rejecting mocks."""
    service = cast(Any, object.__new__(BenchmarkingService))
    pairs = {"CR17-18": ("Q17", "Q18"), "CR19-20": ("Q19", "Q20")}
    service.__dict__["_experiment_context"] = SimpleNamespace(
        classifiers={qubit: object() for pair in pairs.values() for qubit in pair},
        state_centers={},
        experiment_system=SimpleNamespace(
            get_target=lambda label: SimpleNamespace(is_cr=label in pairs)
        ),
        calib_note=SimpleNamespace(cr_params={}),
        cr_pair=pairs.__getitem__,
        reset_awg_and_capunits=Mock(),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        run_sweep_measurement=AsyncMock(
            side_effect=AssertionError(
                "Invalid RB inputs must fail before measurement."
            )
        )
    )
    with AsyncBridge() as bridge:
        monkeypatch.setattr(
            benchmarking_module, "get_shared_async_bridge", lambda *, key: bridge
        )
        yield service


@pytest.mark.parametrize("inside_loop", [False, True])
@pytest.mark.parametrize("in_parallel", [False, True])
@pytest.mark.parametrize(
    ("targets", "calibrated_targets", "has_classifiers", "message"),
    [
        pytest.param(
            ["CR17-18"],
            [],
            True,
            "CR parameters not found for CR17-18.",
            id="missing-cr-parameters",
        ),
        pytest.param(
            ["CR17-18"],
            [],
            False,
            "CR parameters not found for CR17-18.",
            id="missing-both",
        ),
        pytest.param(
            ["CR17-18", "CR19-20"],
            ["CR17-18"],
            True,
            "CR parameters not found for CR19-20.",
            id="partially-calibrated",
        ),
        pytest.param(
            ["CR17-18"],
            ["CR17-18"],
            False,
            "Classifier not found for Q17, Q18.",
            id="missing-classifiers",
        ),
    ],
)
def test_rb_rejects_missing_prerequisites_before_hardware_access(
    rb_service: Any,
    targets: list[str],
    calibrated_targets: list[str],
    has_classifiers: bool,
    message: str,
    in_parallel: bool,
    inside_loop: bool,
) -> None:
    """Missing 2Q prerequisites should raise before hardware access, including in notebooks."""
    rb_service.ctx.calib_note.cr_params = {
        target: object() for target in calibrated_targets
    }
    if not has_classifiers:
        rb_service.ctx.classifiers = {}

    def invoke() -> None:
        rb_service.randomized_benchmarking(
            targets=targets,
            in_parallel=in_parallel,
            n_cliffords_range=[0],
            n_trials=1,
            seeds=[0],
            plot=False,
            save_image=False,
        )

    async def invoke_inside_loop() -> None:
        invoke()

    run = (lambda: asyncio.run(invoke_inside_loop())) if inside_loop else invoke
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        run()

    rb_service.ctx.reset_awg_and_capunits.assert_not_called()
    rb_service.measurement_service.run_sweep_measurement.assert_not_called()


@pytest.mark.parametrize("targets", [["Q17"], ["CR17-18", "Q17"]])
def test_rb_2q_rejects_non_cr_targets(rb_service: Any, targets: list[str]) -> None:
    """2Q RB should reject non-CR targets instead of silently dropping them."""
    rb_service.ctx.calib_note.cr_params = {"CR17-18": object()}

    with pytest.raises(ValueError, match="`Q17` is not a 2Q target"):
        rb_service.rb_experiment_2q(targets=targets, n_cliffords_range=[0])

    rb_service.ctx.reset_awg_and_capunits.assert_not_called()


@pytest.mark.parametrize("method_name", ["randomized_benchmarking", "rb_experiment_2q"])
def test_rb_rejects_empty_targets(rb_service: Any, method_name: str) -> None:
    """RB should reject an empty target collection with an explicit input error."""
    with pytest.raises(ValueError, match=r"At least one.*target is required"):
        getattr(rb_service, method_name)(targets=[])

    rb_service.ctx.reset_awg_and_capunits.assert_not_called()

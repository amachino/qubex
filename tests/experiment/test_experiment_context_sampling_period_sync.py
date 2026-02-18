"""Tests for sampling-period synchronization in experiment context."""

from __future__ import annotations

from types import SimpleNamespace

from qubex.experiment.experiment_context import ExperimentContext


def test_connect_applies_measurement_sampling_period_to_pulse_library(
    monkeypatch,
) -> None:
    """Given measurement dt, when connecting, then pulse sampling period matches measurement."""
    captured: dict[str, float] = {}
    context = object.__new__(ExperimentContext)
    context.__dict__["_measurement"] = SimpleNamespace(
        sampling_period=0.4,
        connect=lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "qubex.experiment.experiment_context.set_sampling_period",
        lambda dt: captured.__setitem__("sampling_period", float(dt)),
    )

    ExperimentContext.connect(context)

    assert captured["sampling_period"] == 0.4


def test_connect_syncs_pulse_sampling_period(monkeypatch) -> None:
    """Given connect call, when backend connects, then context synchronizes pulse sampling period."""
    calls: list[tuple[str, dict]] = []
    context = object.__new__(ExperimentContext)
    context.__dict__["_measurement"] = SimpleNamespace(
        connect=lambda **kwargs: calls.append(("connect", kwargs)),
    )
    monkeypatch.setattr(
        ExperimentContext,
        "_sync_pulse_sampling_period",
        lambda self: calls.append(("sync", {})) or 0.4,
    )

    ExperimentContext.connect(context, sync_clocks=False, parallel=True)

    assert calls == [
        ("connect", {"sync_clocks": False, "parallel": True}),
        ("sync", {}),
    ]


def test_reload_syncs_pulse_sampling_period(monkeypatch) -> None:
    """Given reload call, when backend reloads, then context synchronizes pulse sampling period."""
    calls: list[tuple[str, dict]] = []
    context = object.__new__(ExperimentContext)
    context.__dict__["_measurement"] = SimpleNamespace(
        reload=lambda **kwargs: calls.append(("reload", kwargs)),
    )
    context.__dict__["_configuration_mode"] = "ge-cr-cr"
    monkeypatch.setattr(
        ExperimentContext,
        "_sync_pulse_sampling_period",
        lambda self: calls.append(("sync", {})) or 0.4,
    )

    ExperimentContext.reload(context)

    assert calls == [
        ("reload", {"configuration_mode": "ge-cr-cr"}),
        ("sync", {}),
    ]

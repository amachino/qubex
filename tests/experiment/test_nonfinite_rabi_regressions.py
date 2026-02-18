"""Regression tests for non-finite Rabi parameter handling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qubex.experiment.experiment_context import ExperimentContext
from qubex.experiment.models.experiment_result import SweepData
from qubex.experiment.models.rabi_param import RabiParam


@dataclass
class _CalibrationNoteStub:
    updates: list[tuple[str, dict[str, float | str]]] = field(default_factory=list)

    def update_rabi_param(self, label: str, data: dict[str, float | str]) -> None:
        self.updates.append((label, data))


def test_store_rabi_params_skips_nonfinite_r2() -> None:
    """Given non-finite r2, when storing params, then calibration note is not updated."""
    context = object.__new__(ExperimentContext)
    context.__dict__["_calib_note"] = _CalibrationNoteStub()

    ExperimentContext.store_rabi_params(
        context,
        {
            "Q00": RabiParam.nan(target="Q00"),
            "Q01": RabiParam(
                target="Q01",
                amplitude=1.0,
                frequency=1.0,
                phase=0.0,
                offset=0.0,
                noise=0.1,
                angle=0.0,
                distance=0.0,
                r2=0.9,
                reference_phase=0.0,
            ),
        },
    )

    calib_note = context.__dict__["_calib_note"]
    assert [label for label, _ in calib_note.updates] == ["Q01"]


def test_store_rabi_params_logs_skip_reason_at_info(caplog) -> None:
    """Given skipped params, when storing, then info logs include the skip reason."""
    context = object.__new__(ExperimentContext)
    context.__dict__["_calib_note"] = _CalibrationNoteStub()

    with caplog.at_level("INFO"):
        ExperimentContext.store_rabi_params(
            context,
            {
                "Q00": RabiParam.nan(target="Q00"),
                "Q01": RabiParam(
                    target="Q01",
                    amplitude=1.0,
                    frequency=1.0,
                    phase=0.0,
                    offset=0.0,
                    noise=0.1,
                    angle=0.0,
                    distance=0.0,
                    r2=0.3,
                    reference_phase=0.0,
                ),
            },
            r2_threshold=0.5,
        )

    assert "Q00" in caplog.text
    assert "non-finite r2" in caplog.text
    assert "Q01" in caplog.text
    assert "below threshold" in caplog.text


def test_sweep_plot_normalize_omits_nonfinite_error_bar(monkeypatch) -> None:
    """Given non-finite noise ratio, when plotting normalized sweep, then it does not pass an invalid error bar value."""
    monkeypatch.setattr(
        "qubex.experiment.models.experiment_result.go.Figure.show",
        lambda *_args, **_kwargs: None,
    )

    data = SweepData(
        target="Q00",
        data=np.array([1 + 0j, 2 + 0j]),
        sweep_range=np.array([0.0, 1.0]),
        rabi_param=RabiParam.nan(target="Q00"),
    )

    fig = data.plot(normalize=True, return_figure=True)

    assert fig is not None
    figure_data = fig.to_plotly_json()["data"]
    assert len(figure_data) == 1
    assert "value" not in figure_data[0].get("error_y", {})


def test_sweep_plot_normalize_prints_reason_when_rabi_param_missing(
    monkeypatch, capsys
) -> None:
    """Given missing rabi_param, when plotting normalized sweep, then it prints the reason with a hint."""
    monkeypatch.setattr(
        "qubex.experiment.models.experiment_result.go.Figure.show",
        lambda *_args, **_kwargs: None,
    )

    data = SweepData(
        target="Q00",
        data=np.array([1 + 0j, 2 + 0j]),
        sweep_range=np.array([0.0, 1.0]),
        rabi_param=None,
    )

    data.plot(normalize=True)

    captured = capsys.readouterr()
    assert "Q00" in captured.out
    assert "rabi_param is missing" in captured.out
    assert "obtain_rabi_params" in captured.out
    assert "Experiment instance" in captured.out

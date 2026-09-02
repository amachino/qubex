"""Tests for `qubex.contrib.experiment.warmup_characterization`."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.contrib import (
    check_mux_isolation,
    effective_temperature,
    load_warmup_log,
    plot_warmup_log,
    preflight_check,
    warmup_campaign,
)
from qubex.contrib.experiment import warmup_characterization
from qubex.experiment.experiment_exceptions import CalibrationMissingError
from qubex.experiment.models.experiment_result import ExperimentResult
from qubex.experiment.models.result import Result

_PLANCK = 6.62607015e-34
_BOLTZMANN = 1.380649e-23


class _FakeExperiment:
    def __init__(self) -> None:
        labels = ["Q32", "Q33"]
        targets: dict[str, Any] = {}
        for index, label in enumerate(labels):
            targets[label] = SimpleNamespace(frequency=8.0 + 0.1 * index)
            targets[f"R{label}"] = SimpleNamespace(frequency=10.0 + 0.1 * index)
        self.ctx = SimpleNamespace(
            qubit_labels=labels,
            targets=targets,
            resolve_ge_label=lambda target: target,
            resolve_ef_label=lambda target: f"{target}-ef",
            resolve_read_label=lambda target: f"R{target}",
            resolve_qubit_label=lambda target: target,
        )
        self.box_map = {"Q32": "BOX07", "Q33": "BOX07", "Q40": "BOX08", "Q41": "BOX08"}
        self.mux_map = {7: ["Q32", "Q33"], 8: ["Q40", "Q41"]}
        self.ctx.box_ids = ["BOX07"]
        self.ctx.experiment_system = SimpleNamespace(
            get_mux=self._get_mux,
            get_boxes_for_qubits=self._get_boxes_for_qubits,
        )
        self.calibrated_pulses = set(labels) | {f"{label}-ef" for label in labels}
        self.pulse = SimpleNamespace(
            rabi_params={label: object() for label in labels},
            x180=self._x180,
        )
        self.modified_frequency_calls: list[dict[str, float] | None] = []
        self.reflection_calls: list[dict[str, Any]] = []
        self.ramsey_shift = 0.001
        self.coherence_value = 10_000.0
        self.reflection_shift = 0.002

    def _get_mux(self, mux: int | str) -> SimpleNamespace:
        qubits = self.mux_map[int(mux)]
        return SimpleNamespace(
            index=int(mux),
            resonators=[SimpleNamespace(qubit=qubit) for qubit in qubits],
        )

    def _get_boxes_for_qubits(self, qubits: Any) -> list[SimpleNamespace]:
        box_ids = {self.box_map[qubit] for qubit in qubits}
        return [SimpleNamespace(id=box_id) for box_id in sorted(box_ids)]

    def _x180(self, label: str) -> str:
        if label not in self.calibrated_pulses:
            raise CalibrationMissingError(f"missing {label}")
        return f"x180:{label}"

    @contextmanager
    def modified_frequencies(
        self, frequencies: dict[str, float] | None
    ) -> Iterator[None]:
        self.modified_frequency_calls.append(frequencies)
        yield

    def obtain_reference_points(self, targets: Any, **kwargs: Any) -> Result:
        return Result(data={})

    def obtain_rabi_params(self, targets: Any, **kwargs: Any) -> ExperimentResult[Any]:
        return ExperimentResult(data={})

    def ramsey_experiment(self, targets: Any, **kwargs: Any) -> ExperimentResult[Any]:
        return ExperimentResult(
            data={
                target: SimpleNamespace(
                    t2=self.coherence_value,
                    bare_freq=self.ctx.targets[target].frequency + self.ramsey_shift,
                )
                for target in targets
            }
        )

    def t1_experiment(self, targets: Any, **kwargs: Any) -> ExperimentResult[Any]:
        return ExperimentResult(
            data={
                target: SimpleNamespace(t1=self.coherence_value) for target in targets
            }
        )

    def t2_experiment(self, targets: Any, **kwargs: Any) -> ExperimentResult[Any]:
        return ExperimentResult(
            data={
                target: SimpleNamespace(t2=self.coherence_value) for target in targets
            }
        )

    def measure_state_distribution(
        self, targets: Any, **kwargs: Any
    ) -> list[SimpleNamespace]:
        shots = np.array([0.1 + 0.2j, 0.3 - 0.1j])
        return [
            SimpleNamespace(
                data={target: SimpleNamespace(kerneled=shots) for target in targets}
            )
            for _ in range(2)
        ]

    def measure_reflection_coefficient(self, target: str, **kwargs: Any) -> Result:
        self.reflection_calls.append({"target": target, **kwargs})
        f_read = self.ctx.targets[f"R{target}"].frequency
        return Result(
            data={
                "f_r": f_read + self.reflection_shift,
                "kappa_ex": 0.002,
                "kappa_in": 0.0001,
            }
        )


def _fake_thermal(exp: Any, target: str, **kwargs: Any) -> Result:
    return Result(data={"p_ex": 0.02})


def test_warmup_helpers_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then warm-up helpers are available."""
    assert callable(warmup_campaign)
    assert callable(effective_temperature)
    assert callable(load_warmup_log)
    assert callable(plot_warmup_log)


def test_effective_temperature_matches_boltzmann_relation() -> None:
    """Given a small excited population, when converted, then the two-level Boltzmann relation holds."""
    p_ex = 0.01
    frequency = 8.0
    expected = _PLANCK * frequency * 1e9 / (_BOLTZMANN * np.log((1 - p_ex) / p_ex))
    assert effective_temperature(p_ex, frequency) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("p_ex", "frequency"),
    [
        (0.0, 8.0),
        (0.5, 8.0),
        (0.9, 8.0),
        (-0.1, 8.0),
        (0.01, 0.0),
        (0.01, float("nan")),
    ],
)
def test_effective_temperature_rejects_unphysical_inputs(
    p_ex: float,
    frequency: float,
) -> None:
    """Given unphysical inputs, when converted, then NaN is returned."""
    assert np.isnan(effective_temperature(p_ex, frequency))


def test_warmup_campaign_runs_cycles_and_tracks_frequencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a healthy experiment, when two cycles run, then metrics, tracking, and files are produced."""
    monkeypatch.setattr(
        warmup_characterization, "measure_thermal_excitation", _fake_thermal
    )
    exp = _FakeExperiment()
    output_dir = tmp_path / "run"

    result = warmup_campaign(cast(Any, exp), output_dir=output_dir, max_cycles=2)

    assert result.data["stop_reason"] == "max_cycles"
    assert result.data["n_cycles"] == 2
    assert result.data["qubit_alive"] == {"Q32": True, "Q33": True}
    assert result.data["resonator_alive"] == {"Q32": True, "Q33": True}

    assert result.data["tracked_frequencies"]["Q32"] == pytest.approx(8.001)
    assert result.data["tracked_frequencies"]["Q33"] == pytest.approx(8.101)
    assert exp.modified_frequency_calls[0] is None
    second_call = exp.modified_frequency_calls[1]
    assert second_call is not None
    assert second_call["Q32"] == pytest.approx(8.001)

    assert exp.reflection_calls[0]["center_frequency"] is None
    assert exp.reflection_calls[2]["center_frequency"] == pytest.approx(10.002)

    records = result.data["records"]
    ok_steps = {
        record["step"]
        for record in records
        if record["status"] == "ok" and record.get("target") == "Q32"
    }
    assert {
        "ramsey",
        "t1",
        "t2_echo",
        "thermal",
        "single_shot",
        "reflection",
    } <= ok_steps

    thermal_record = next(
        record
        for record in records
        if record["step"] == "thermal" and record.get("target") == "Q32"
    )
    assert thermal_record["values"]["p_ex"] == pytest.approx(0.02)
    assert thermal_record["values"]["frequency"] == pytest.approx(8.001)
    assert thermal_record["values"]["t_eff"] == pytest.approx(
        effective_temperature(0.02, 8.001)
    )

    assert (output_dir / "warmup_log.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    single_shot_file = output_dir / "single_shot" / "cycle_00001.npz"
    assert single_shot_file.exists()
    with np.load(single_shot_file) as arrays:
        assert "Q32_state0" in arrays
        assert "Q33_state1" in arrays


def test_warmup_campaign_falls_back_to_resonator_tracking(tmp_path: Path) -> None:
    """Given failing coherence fits, when the failure budget is spent, then resonator tracking continues alone."""
    exp = _FakeExperiment()
    exp.coherence_value = float("nan")

    result = warmup_campaign(
        cast(Any, exp),
        output_dir=tmp_path / "run",
        max_cycles=4,
        max_consecutive_failures=2,
        steps=["ramsey", "t1", "t2_echo", "reflection"],
    )

    assert result.data["stop_reason"] == "max_cycles"
    assert result.data["qubit_alive"] == {"Q32": False, "Q33": False}
    assert result.data["resonator_alive"] == {"Q32": True, "Q33": True}

    records = result.data["records"]
    lost_cycles = [
        record["cycle"] for record in records if record["step"] == "qubit_lost"
    ]
    assert lost_cycles == [2, 2]
    reflection_ok = [
        record
        for record in records
        if record["step"] == "reflection" and record["status"] == "ok"
    ]
    assert len(reflection_ok) == 2 * 4


def test_warmup_campaign_honors_stop_file(tmp_path: Path) -> None:
    """Given an existing stop file, when the campaign starts, then it stops before any cycle."""
    exp = _FakeExperiment()
    stop_file = tmp_path / "STOP"
    stop_file.write_text("stop", encoding="utf-8")

    result = warmup_campaign(
        cast(Any, exp),
        output_dir=tmp_path / "run",
        stop_file=stop_file,
        steps=["reflection"],
    )

    assert result.data["stop_reason"] == "stop_file"
    assert result.data["n_cycles"] == 0


def test_warmup_campaign_survives_keyboard_interrupt(tmp_path: Path) -> None:
    """Given a keyboard interrupt mid-cycle, when it fires, then the log is finalized gracefully."""

    class _InterruptingExperiment(_FakeExperiment):
        def t1_experiment(self, targets: Any, **kwargs: Any) -> ExperimentResult[Any]:
            raise KeyboardInterrupt

    exp = _InterruptingExperiment()

    result = warmup_campaign(
        cast(Any, exp),
        output_dir=tmp_path / "run",
        max_cycles=3,
        steps=["t1", "reflection"],
    )

    assert result.data["stop_reason"] == "keyboard_interrupt"
    assert result.data["records"][-1]["step"] == "campaign_end"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_duration": 0.0}, "max_duration must be positive"),
        ({"max_cycles": 0}, "max_cycles must be at least 1"),
        (
            {"max_consecutive_failures": 0},
            "max_consecutive_failures must be at least 1",
        ),
        ({"refresh_rabi_every": 0}, "refresh_rabi_every must be at least 1"),
        ({"steps": ["bogus"]}, "Unknown warm-up step"),
        ({"steps": ["t1", "T1"]}, "Duplicate warm-up step"),
        ({"steps": []}, "At least one warm-up step"),
    ],
)
def test_warmup_campaign_validates_inputs(
    tmp_path: Path,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Given invalid options, when called, then a clear ValueError is raised."""
    with pytest.raises(ValueError, match=match):
        warmup_campaign(
            cast(Any, _FakeExperiment()),
            output_dir=tmp_path / "run",
            **kwargs,
        )


def test_plot_warmup_log_builds_figures_from_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a campaign log, when plotted, then per-metric figures are built and saved."""
    monkeypatch.setattr(
        warmup_characterization, "measure_thermal_excitation", _fake_thermal
    )
    exp = _FakeExperiment()
    output_dir = tmp_path / "run"
    warmup_campaign(cast(Any, exp), output_dir=output_dir, max_cycles=1)

    records = load_warmup_log(output_dir)
    assert any(record["step"] == "campaign_start" for record in records)

    save_dir = tmp_path / "figures"
    figures = plot_warmup_log(output_dir, plot=False, save_dir=save_dir)

    assert {"t1", "t2_echo", "t2_star", "f_ge", "p_ex", "t_eff", "f_r"} <= set(figures)
    assert (save_dir / "warmup_t1.html").exists()

    with pytest.raises(ValueError, match="Unknown metric"):
        plot_warmup_log(records, metrics=["bogus"], plot=False)


def test_preflight_check_reports_all_ready_for_calibrated_targets() -> None:
    """Given fully calibrated targets, when checked offline, then every step is ready."""
    exp = _FakeExperiment()

    result = preflight_check(cast(Any, exp), verbose=False)

    assert result.data["all_ready"] is True
    entry = result.data["targets"]["Q32"]
    assert entry["missing_steps"] == []
    assert set(entry["ready_steps"]) == set(warmup_characterization.WARMUP_STEPS)
    assert all(entry["checks"].values())


def test_preflight_check_reports_missing_calibrations() -> None:
    """Given missing ef and Rabi calibrations, when checked, then the dependent steps are reported."""
    exp = _FakeExperiment()
    exp.calibrated_pulses.discard("Q33-ef")
    del exp.pulse.rabi_params["Q32"]

    result = preflight_check(cast(Any, exp), verbose=True)

    assert result.data["all_ready"] is False
    q32 = result.data["targets"]["Q32"]
    q33 = result.data["targets"]["Q33"]
    assert q32["checks"]["ge_rabi_params"] is False
    assert q32["missing_steps"] == ["ramsey", "t1", "t2_echo"]
    assert q33["checks"]["ef_pi_pulse"] is False
    assert q33["missing_steps"] == ["thermal"]


def test_warmup_campaign_invokes_cycle_callback_and_survives_its_errors(
    tmp_path: Path,
) -> None:
    """Given a cycle callback that fails once, when cycles run, then it is called each cycle and the failure is logged."""
    calls: list[int] = []

    def _callback(info: dict[str, Any]) -> None:
        calls.append(info["cycle"])
        assert info["qubit_alive"] == {"Q32": False, "Q33": False}
        if info["cycle"] == 1:
            raise RuntimeError("boom")

    result = warmup_campaign(
        cast(Any, _FakeExperiment()),
        output_dir=tmp_path / "run",
        max_cycles=2,
        steps=["reflection"],
        on_cycle_end=_callback,
    )

    assert calls == [1, 2]
    statuses = [
        record["status"]
        for record in result.data["records"]
        if record["step"] == "cycle_callback"
    ]
    assert statuses == ["failed", "ok"]
    assert result.data["stop_reason"] == "max_cycles"


def test_check_mux_isolation_passes_when_nothing_is_shared() -> None:
    """Given a forbidden mux on another box, when checked, then the experiment is isolated."""
    exp = _FakeExperiment()

    result = check_mux_isolation(cast(Any, exp), [8], verbose=False)

    assert result.data["isolated"] is True
    assert result.data["forbidden_qubits"] == ["Q40", "Q41"]
    assert result.data["forbidden_boxes"] == ["BOX08"]
    assert result.data["selected_boxes"] == ["BOX07"]
    assert result.data["shared_qubits"] == []
    assert result.data["shared_boxes"] == []


def test_check_mux_isolation_detects_shared_box() -> None:
    """Given a forbidden mux wired to a selected box, when checked, then isolation fails."""
    exp = _FakeExperiment()
    exp.box_map["Q40"] = "BOX07"

    result = check_mux_isolation(cast(Any, exp), [8], verbose=True)

    assert result.data["isolated"] is False
    assert result.data["shared_boxes"] == ["BOX07"]
    assert result.data["shared_qubits"] == []


def test_check_mux_isolation_detects_shared_qubit() -> None:
    """Given a selected qubit inside a forbidden mux, when checked, then isolation fails."""
    exp = _FakeExperiment()
    exp.mux_map[8] = ["Q32", "Q41"]

    result = check_mux_isolation(cast(Any, exp), [8], verbose=False)

    assert result.data["isolated"] is False
    assert result.data["shared_qubits"] == ["Q32"]


def test_check_mux_isolation_fails_closed_for_unknown_mux() -> None:
    """Given a forbidden mux missing from the configuration, when checked, then a ValueError is raised."""
    with pytest.raises(ValueError, match="not found"):
        check_mux_isolation(cast(Any, _FakeExperiment()), [12], verbose=False)

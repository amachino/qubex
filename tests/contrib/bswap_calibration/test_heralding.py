"""Offline paired-readout tests with the real schedule builder and fake capture data."""

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Blank, PulseArray, PulseSchedule, Rect

from qubex.contrib.experiment.bswap_calibration.heralding import (
    acquire_heralded,
    build_heralded_schedule,
)
from qubex.contrib.experiment.bswap_calibration.measurements import CampaignMeasurements
from qubex.contrib.experiment.bswap_calibration.pulses import (
    compile_campaign,
    make_squad_pulse,
)
from qubex.experiment.experiment import Experiment
from qubex.measurement.adapters import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.measurement.models.capture_data import CaptureData
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.system import TargetRegistry


class _Classifier:
    def __init__(self, failure: bool = False, fractional: bool = False) -> None:
        self.failure, self.fractional = failure, fractional

    def predict(self, iq: np.ndarray) -> np.ndarray:
        if self.failure:
            raise RuntimeError("synthetic classifier failure")
        if self.fractional:
            return np.full(iq.shape, 0.5)
        return iq.real.astype(int)


class _Experiment:
    def __init__(self) -> None:
        self.profile = MeasurementConstraintProfile.quel1()
        self.ctx = SimpleNamespace(
            measurement=SimpleNamespace(constraint_profile=self.profile),
            experiment_system=SimpleNamespace(target_registry=TargetRegistry()),
        )
        self.classifiers = {q: _Classifier() for q in ("Q00", "Q01")}
        self.pulse = SimpleNamespace(
            get_drag_hpi_pulse=lambda _: Rect(duration=16, amplitude=0.1),
            get_drag_pi_pulse=lambda _: Rect(duration=24, amplitude=0.2),
            rabi_params={"Q00": SimpleNamespace(frequency=0.04)},
            readout=lambda _: PulseArray(
                [Blank(8), Rect(duration=64, amplitude=0.1).padded(80)]
            ),
        )
        self.params = SimpleNamespace(get_control_amplitude=lambda _: 0.5)
        self.targets = {
            label: SimpleNamespace(
                frequency=freq, is_pump=False, is_read=label.startswith("R")
            )
            for label, freq in {
                "Q00": 5.0,
                "Q01": 4.4,
                "D": 4.7,
                "C": 4.71,
                "RQ00": 6.0,
                "RQ01": 6.1,
            }.items()
        }
        self.builder = MeasurementScheduleBuilder(
            control_params=cast(Any, SimpleNamespace(readout_amplitude={})),
            pulse_factory=cast(Any, SimpleNamespace()),
            targets=cast(Any, self.targets),
            mux_dict={},
            constraint_profile=self.profile,
        )
        self.initial = np.array([[0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]])
        final_states = np.array([3, 0, 1, 2, 3, 0])
        self.final = np.stack([final_states // 2, final_states % 2])
        self.capture_count = 2
        self.averaged = False
        self.calls: list[dict[str, Any]] = []
        self.build_calls: list[dict[str, Any]] = []

    def build_measurement_schedule(
        self, pulse_schedule: PulseSchedule, **options: Any
    ) -> MeasurementSchedule:
        self.build_calls.append(deepcopy(options))
        return self.builder.build(schedule=pulse_schedule, **options)

    async def run_measurement(
        self, schedule: MeasurementSchedule, **options: Any
    ) -> MeasurementResult:
        self.calls.append({"schedule": schedule, **deepcopy(options)})
        config = MeasurementConfig(
            n_shots=options["n_shots"],
            shot_interval=options["shot_interval"],
            shot_averaging=self.averaged,
            time_integration=True,
            state_classification=False,
        )
        data = {}
        for qi, q in enumerate(("Q00", "Q01")):
            arrays = [self.initial[qi], self.final[qi], np.ones(3)]
            data[q] = [
                CaptureData.from_primary_data(
                    target=q,
                    data=np.asarray(0.5j if self.averaged else array, dtype=complex),
                    config=config.model_copy(update={"n_shots": len(array)}),
                    sampling_period=2.0,
                )
                for array in arrays[: self.capture_count]
            ]
        return MeasurementResult(data=data, measurement_config=config)


def _session(tmp_path: Path) -> tuple[CampaignMeasurements, _Experiment]:
    exp = _Experiment()
    recipe = dict(
        gate_kind="sqrt_bswap",
        amplitude=0.5,
        frequency_ghz=4.7006,
        duration_ns=116.0,
        ramp_ns=16.0,
        cd_strength=0.5,
        design_delta_scale=1.0,
        window={"type": "hann"},
        gate_start_ns=24.0,
        phase_calibration=dict(
            pre_active_rad=0.2, post_active_rad=0.1, post_passive_rad=-0.2
        ),
    )
    session = CampaignMeasurements(
        cast(Experiment, exp),
        tmp_path,
        {"sqrt_bswap": recipe},
        qubits=("Q00", "Q01"),
        drive_label="D",
        cancel_label="C",
        shots=6,
    )
    return session, exp


def test_paired_iq_preserves_shots_and_selects_only_initial_gg(tmp_path: Path) -> None:
    """Initial gg selection retains excited final outcomes and an all-shot companion."""
    session, exp = _session(tmp_path)
    recipes_before = deepcopy(session.recipes)
    targets_before = deepcopy(session.targets)
    references_before = deepcopy(session.references)
    readout_before = exp.pulse.readout("Q00").values.copy()
    row = asyncio.run(
        acquire_heralded(
            session, ["RAW_SQRT_BSWAP"], tmp_path, "paired", settle_ns=1000
        )
    )
    assert row["counts_allshots"] == [2, 1, 1, 2]
    assert row["counts_initial_gg"] == [1, 0, 1, 1]
    assert row["accepted_shots"] == 3
    assert row["acceptance_fraction"] == 0.5
    assert row["selection_uses_final_outcome"] is False
    assert row["physical_ground_state_purity_verified"] is False
    with np.load(row["iq_file"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["iq_initial"], exp.initial)
        np.testing.assert_array_equal(saved["iq_final"], exp.final)
        np.testing.assert_array_equal(saved["shot_index"], np.arange(6))
    with np.load(row["classification_file"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["initial_gg_mask"], [1, 0, 0, 1, 0, 1])
    call = exp.calls[0]
    assert call["shot_averaging"] is False
    assert call["state_classification"] is False
    assert call["time_integration"] is True
    assert call["final_measurement"] is False
    assert exp.build_calls[0]["frequencies"] == session.targets
    assert exp.build_calls[0]["capture_placement"] == "pulse_aligned"
    assert session.serial == 0
    assert session.recipes == recipes_before
    assert session.targets == targets_before
    assert session.references == references_before
    np.testing.assert_array_equal(exp.pulse.readout("Q00").values, readout_before)


def test_schedule_uses_real_prefix_once_and_preserves_carrier_phase(
    tmp_path: Path,
) -> None:
    """Herald delay enters logical time and the backend prefix enters detuning once."""
    session, exp = _session(tmp_path)
    built, record = build_heralded_schedule(
        session, ["RAW_SQRT_BSWAP"], settle_ns=1000, prepared=("+", "0")
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, SimpleNamespace()),
        experiment_system=cast(Any, exp.ctx.experiment_system),
        constraint_profile=exp.profile,
    )
    adapter.validate_schedule(built)
    assert record["herald_prefix_ns"] == 1088.0
    assert record["compiled"]["global_start_ns"] == 1088.0
    assert record["compiled"]["backend_preamble_ns"] == 40.0
    captures = built.capture_schedule.channels["RQ00"]
    assert [c.is_workaround for c in captures] == [True, False, False]
    assert captures[1].start_time == 48.0
    assert captures[1].duration == 80.0
    event = next(e for e in record["compiled"]["events"] if e["kind"] == "sqrt_bswap")
    assert event["start_ns"] == 1112.0
    physical_start = event["start_ns"] + 40
    pulse = make_squad_pulse(
        session.recipes["sqrt_bswap"],
        rabi_ghz_per_amplitude=session.rabi_scale,
        transition_frequency_ghz=session.references["Q00"],
    )
    offset = 4.7006 - session.targets["D"]
    expected = pulse.values * np.exp(
        1j * event["logical_drive_phase_rad"]
        - 2j * np.pi * offset * (physical_start + pulse.times)
    )
    values = built.pulse_schedule.get_sampled_sequences()["D"]
    start = int(physical_start / 2)
    np.testing.assert_allclose(
        values[start : start + len(expected)], expected, atol=1e-12
    )
    assert built.pulse_schedule.get_frequencies()["D"] == 4.7


def test_empty_acceptance_is_reported_without_false_probability(tmp_path: Path) -> None:
    """Zero accepted shots retain all raw data and never fabricate a conditional result."""
    session, exp = _session(tmp_path)
    exp.initial[:] = 1
    row = asyncio.run(acquire_heralded(session, [], tmp_path, "empty", settle_ns=1000))
    assert row["counts_initial_gg"] == [0, 0, 0, 0]
    assert row["raw_probabilities_initial_gg"] is None
    assert row["accepted_shots"] == 0
    assert row["status"] == "no_accepted_shots"
    assert Path(row["iq_file"]).exists()


@pytest.mark.parametrize("extra_delay", [0.0, 6.0])
def test_existing_acquire_delay_matches_herald_control_samples(
    tmp_path: Path, extra_delay: float
) -> None:
    """Using herald_prefix_ns as the ordinary delay produces identical control IQ."""
    session, _ = _session(tmp_path)
    gates = ["RAW_SQRT_BSWAP", ("XY", 0.3, -0.7), "RAW_SQRT_BSWAP"]
    built, record = build_heralded_schedule(
        session,
        gates,
        settle_ns=1000,
        prepared=("+", "+i"),
        basis="YX",
        delay_ns=extra_delay,
    )
    reference, reference_record = compile_campaign(
        gates,
        recipes=session.recipes,
        qubits=session.qubits,
        drive_label=session.drive_label,
        cancel_label=session.cancel_label,
        target_frequencies_ghz=session.targets,
        reference_frequencies_ghz=session.references,
        rabi_ghz_per_amplitude=session.rabi_scale,
        x90=session.x90,
        xpi=session.xpi,
        prepared=("+", "+i"),
        basis="YX",
        delay_ns=record["herald_prefix_ns"] + extra_delay,
        backend_preamble_ns=session.backend_preamble_ns,
    )
    assert record["compiled"]["events"] == reference_record["events"]
    physical = built.pulse_schedule.get_sampled_sequences()
    offset = int(session.backend_preamble_ns / 2)
    for label, expected in reference.get_sampled_sequences().items():
        np.testing.assert_allclose(
            physical[label][offset : offset + len(expected)], expected, atol=1e-12
        )


@pytest.mark.parametrize("fractional", [False, True])
def test_iq_is_saved_before_classifier_failure(
    tmp_path: Path, fractional: bool
) -> None:
    """Classifier exceptions and invalid labels cannot discard acquired paired IQ."""
    session, exp = _session(tmp_path)
    exp.classifiers["Q00"] = _Classifier(failure=not fractional, fractional=fractional)
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(acquire_heralded(session, [], tmp_path, "bad", settle_ns=1000))
    raw = list(tmp_path.glob("*_iq.npz"))
    assert len(raw) == 1
    with np.load(raw[0], allow_pickle=False) as saved:
        assert saved["iq_initial"].shape == (2, 6)
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["status"] == "classification_failed"


@pytest.mark.parametrize("count", [1, 3])
def test_malformed_capture_count_saves_each_available_array(
    tmp_path: Path, count: int
) -> None:
    """Missing or extra captures remain inspectable without guessing their identity."""
    session, exp = _session(tmp_path)
    exp.capture_count = count
    with pytest.raises(ValueError, match="two"):
        asyncio.run(acquire_heralded(session, [], tmp_path, "count", settle_ns=1000))
    with np.load(next(tmp_path.glob("*_iq.npz")), allow_pickle=False) as saved:
        assert saved[f"q0_capture{count - 1}"].size == (3 if count == 3 else 6)
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["status"] == "invalid_capture_data"


def test_averaged_payload_is_not_accepted_as_paired_iq(tmp_path: Path) -> None:
    """Averaged IQ is retained as evidence but rejected as per-shot heralding input."""
    session, exp = _session(tmp_path)
    exp.averaged = True
    with pytest.raises(ValueError, match="iq_series"):
        asyncio.run(acquire_heralded(session, [], tmp_path, "averaged", settle_ns=1000))
    assert len(list(tmp_path.glob("*_iq.npz"))) == 1


def test_mismatched_shot_lengths_are_retained_without_truncation(
    tmp_path: Path,
) -> None:
    """A short final capture cannot silently change the initial-to-final shot pairing."""
    session, exp = _session(tmp_path)
    exp.final = exp.final[:, :5]
    with pytest.raises(ValueError, match="requested length"):
        asyncio.run(acquire_heralded(session, [], tmp_path, "short", settle_ns=1000))
    with np.load(next(tmp_path.glob("*_iq.npz")), allow_pickle=False) as saved:
        assert saved["q0_capture0"].shape == (6,)
        assert saved["q0_capture1"].shape == (5,)


@pytest.mark.parametrize("shots", [True, 0, -1, 6.5])
def test_noninteger_or_nonpositive_shots_do_not_acquire(
    tmp_path: Path, shots: Any
) -> None:
    """Invalid shot requests are rejected instead of rounded or coerced."""
    session, exp = _session(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(
            acquire_heralded(
                session, [], tmp_path, "shots", settle_ns=1000, shots=shots
            )
        )
    assert exp.calls == []


@pytest.mark.parametrize("settle", [-2, 1, float("nan")])
def test_invalid_settle_time_stops_before_acquisition(
    tmp_path: Path, settle: float
) -> None:
    """Readout-to-control idle must be finite, nonnegative, and on the native grid."""
    session, exp = _session(tmp_path)
    with pytest.raises(ValueError, match="settle_ns"):
        asyncio.run(
            acquire_heralded(session, [], tmp_path, "invalid", settle_ns=settle)
        )
    assert exp.calls == []


def test_deadline_stops_before_schedule_and_hardware(tmp_path: Path) -> None:
    """A reservation deadline prevents even one additional paired acquisition."""
    session, exp = _session(tmp_path)
    session.deadline = datetime.now(timezone.utc) + timedelta(seconds=30)
    with pytest.raises(TimeoutError):
        asyncio.run(acquire_heralded(session, [], tmp_path, "late", settle_ns=1000))
    assert exp.calls == []
    assert exp.build_calls == []

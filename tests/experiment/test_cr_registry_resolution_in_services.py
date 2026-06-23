"""Tests for CR label resolution through experiment context in services."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Blank, PulseArray, PulseSchedule

from qubex.clifford.clifford import Clifford
from qubex.experiment.services.benchmarking_service import BenchmarkingService
from qubex.experiment.services.calibration_service import CalibrationService
from qubex.system import TargetType


def test_calibrate_2q_uses_context_cr_pair_resolution() -> None:
    """Given custom CR labels, when calibrating 2Q gates, then pair resolution uses context mapping."""
    service = cast(Any, object.__new__(CalibrationService))
    captured: list[tuple[str, str]] = []

    service.__dict__["_experiment_context"] = SimpleNamespace(
        cr_labels=["CR_CUSTOM"],
        cr_pair=lambda _label: ("Q17", "Q18"),
        save_calib_note=lambda: None,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["obtain_cr_params"] = lambda **kwargs: (
        captured.append((kwargs["control_qubit"], kwargs["target_qubit"]))
        or {"ok": True}
    )
    service.__dict__["calibrate_zx90"] = lambda **kwargs: {"ok": kwargs}

    service.calibrate_2q(targets=["CR_CUSTOM"], plot=False)

    assert captured == [("Q17", "Q18")]


def test_rb_sequence_2q_uses_context_cr_pair_resolution() -> None:
    """Given custom CR labels, when building 2Q RB sequence, then pair resolution uses context mapping."""
    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(
                is_2q=True,
                is_bswap=False,
                type=TargetType.CTRL_CR,
            )
        ),
        resolve_2q_qubits=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        x90=lambda _target: PulseArray([]),
        zx90=lambda control, target: PulseSchedule([control, target]),
    )
    service.__dict__["_clifford_generator_dict"] = {
        "default": SimpleNamespace(create_rb_sequences=lambda **_kwargs: ([], []))
    }

    schedule = service.rb_sequence_2q(target="CR_CUSTOM", n=0)

    assert list(schedule.labels) == ["Q17", "CR_CUSTOM", "Q18"]


def test_rb_sequence_2q_requires_native_gate_for_generic_2q_target() -> None:
    """Given generic 2Q target, when native gate is omitted, then the gate kind is not guessed."""
    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(
                is_2q=True,
                is_bswap=False,
                type=TargetType.CTRL_2Q,
            )
        ),
        resolve_2q_qubits=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        x90=lambda _target: PulseArray([]),
    )

    with pytest.raises(ValueError, match="native_2q_gate must be provided"):
        service.rb_sequence_2q(target="Q17-Q18-CUSTOM", n=0)


def test_rb_sequence_2q_uses_bswap_native_path() -> None:
    """Given bSWAP target, when building RB, then the bSWAP Clifford table and waveform are used."""
    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(
                is_2q=True,
                is_bswap=True,
                type=TargetType.CTRL_2Q,
            )
        ),
        calib_note=SimpleNamespace(
            get_bswap_param=lambda _target: {
                "post_z_offsets": {},
                "post_z_update_rates": {},
            }
        ),
        resolve_2q_qubits=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        x90=lambda _target: PulseArray([]),
    )
    calls: list[dict[str, object]] = []
    service.__dict__["_clifford_generator_dict"] = {
        "clifford_list_2q_bswap": SimpleNamespace(
            create_rb_sequences=lambda **kwargs: (
                calls.append(kwargs) or ([["BSWAP"]], [])
            )
        )
    }
    bswap_waveform = PulseSchedule(["Q17", "Q17-Q18-BSWAP", "Q18"])

    schedule = service.rb_sequence_2q(
        target="Q17-Q18-BSWAP",
        n=1,
        native_2q_waveform=bswap_waveform,
    )

    assert calls == [{"n": 1, "type": "2Q", "seed": None}]
    assert list(schedule.labels) == ["Q17", "Q17-Q18-BSWAP", "Q18"]


def test_rb_sequence_2q_bswap_defaults_interleaved_waveform_to_native() -> None:
    """Given BSWAP IRB, when interleaved waveform is omitted, then native waveform is reused."""
    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(
                is_2q=True,
                is_bswap=True,
                type=TargetType.CTRL_2Q,
            )
        ),
        calib_note=SimpleNamespace(
            get_bswap_param=lambda _target: {
                "post_z_offsets": {},
                "post_z_update_rates": {},
            }
        ),
        resolve_2q_qubits=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        x90=lambda _target: PulseArray([]),
    )
    calls: list[dict[str, object]] = []
    service.__dict__["_clifford_generator_dict"] = {
        "clifford_list_2q_bswap": SimpleNamespace(
            create_irb_sequences=lambda **kwargs: calls.append(kwargs) or ([[]], [])
        )
    }
    with PulseSchedule(["Q17", "Q17-Q18-BSWAP", "Q18"]) as bswap_waveform:
        bswap_waveform.add("Q17-Q18-BSWAP", Blank(8.0))

    schedule = service.rb_sequence_2q(
        target="Q17-Q18-BSWAP",
        n=1,
        native_2q_waveform=bswap_waveform,
        interleaved_clifford=Clifford.BSWAP(),
    )

    interleave = calls[0]["interleave"]
    assert isinstance(interleave, Clifford)
    assert interleave.name == "BSWAP"
    assert schedule.duration == pytest.approx(8.0)


def test_rb_sequence_2q_bswap_requires_calibration_note_param() -> None:
    """Given bSWAP target, when calibration data is absent, then RB does not silently use zero post-Z."""
    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(
                is_2q=True,
                is_bswap=True,
                type=TargetType.CTRL_2Q,
            )
        ),
        calib_note=SimpleNamespace(get_bswap_param=lambda _target: None),
        resolve_2q_qubits=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        x90=lambda _target: PulseArray([]),
    )
    bswap_waveform = PulseSchedule(["Q17", "Q17-Q18-BSWAP", "Q18"])

    with pytest.raises(ValueError, match="bSWAP calibration parameters are missing"):
        service.rb_sequence_2q(
            target="Q17-Q18-BSWAP",
            n=1,
            native_2q_waveform=bswap_waveform,
        )


def test_rb_sequence_2q_bswap_emits_final_pending_z() -> None:
    """Given pending Z crossing bSWAP, when sequence ends, then the output Z is explicit."""
    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(
                is_2q=True,
                is_bswap=True,
                type=TargetType.CTRL_2Q,
            )
        ),
        calib_note=SimpleNamespace(
            get_bswap_param=lambda _target: {
                "post_z_offsets": {},
                "post_z_update_rates": {},
            }
        ),
        resolve_2q_qubits=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace(
        x90=lambda _target: PulseArray([]),
    )
    service.__dict__["_clifford_generator_dict"] = {
        "clifford_list_2q_bswap": SimpleNamespace(
            create_rb_sequences=lambda **_kwargs: ([["ZI90", "BSWAP"]], [])
        )
    }
    bswap_waveform = PulseSchedule(["Q17", "Q17-Q18-BSWAP", "Q18"])

    schedule = service.rb_sequence_2q(
        target="Q17-Q18-BSWAP",
        n=1,
        native_2q_waveform=bswap_waveform,
    )

    assert schedule.get_final_frame_shift("Q17") == 0.0
    assert schedule.get_final_frame_shift("Q18") == pytest.approx(np.pi / 2)


def test_rb_sequence_2q_bswap_uses_dynamic_z_update_time() -> None:
    """Given bSWAP post-Z rate, when a bSWAP ends, then update uses global end time."""
    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(
                is_2q=True,
                is_bswap=True,
                type=TargetType.CTRL_2Q,
            )
        ),
        calib_note=SimpleNamespace(
            get_bswap_param=lambda _target: {
                "post_z_update_rates": {"Q17": 0.1, "Q18": 0.2},
            }
        ),
        resolve_2q_qubits=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["_clifford_generator_dict"] = {
        "clifford_list_2q_bswap": SimpleNamespace(
            create_rb_sequences=lambda **_kwargs: ([["XI90", "BSWAP"]], [])
        )
    }
    with PulseSchedule(["Q17", "Q17-Q18-BSWAP", "Q18"]) as bswap_waveform:
        bswap_waveform.add("Q17-Q18-BSWAP", Blank(8.0))

    schedule = service.rb_sequence_2q(
        target="Q17-Q18-BSWAP",
        n=1,
        x90={"Q17": Blank(4.0), "Q18": Blank(4.0)},
        native_2q_waveform=bswap_waveform,
    )

    assert schedule.duration == pytest.approx(12.0)
    assert schedule.get_final_frame_shift("Q17") == pytest.approx(-1.2)
    assert schedule.get_final_frame_shift("Q18") == pytest.approx(-2.4)

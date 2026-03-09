"""Tests for QuEL-1 adapter loopback-capture behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
from numpy.testing import assert_allclose
from qxpulse import Blank, Gaussian, PulseSchedule

from qubex.backend.quel1 import SAMPLING_PERIOD_NS, Quel1BackendExecutionResult
from qubex.measurement.adapters.backend_adapter import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.typing import MeasurementMode


@dataclass
class _FakeRange:
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass
class _FakePulseSchedule:
    duration: float
    length: int
    ranges: dict[str, list[_FakeRange]]

    def is_valid(self) -> bool:
        return True

    def get_pulse_ranges(
        self,
        labels: list[str] | None = None,
    ) -> dict[str, list[_FakeRange]]:
        if labels is None:
            return self.ranges
        return {label: self.ranges.get(label, []) for label in labels}


def _make_config(*, mode: MeasurementMode, shots: int) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=shots,
        shot_interval=100.0,
        shot_averaging=(mode == "avg"),
        time_integration=False,
        state_classification=False,
    )


def test_validate_schedule_accepts_full_span_captures_without_pulse_ranges() -> None:
    """Given full-span mode, when captures have no pulse ranges, then validation succeeds."""
    profile = MeasurementConstraintProfile.quel1(sampling_period_ns=SAMPLING_PERIOD_NS)
    block_duration = cast(float, profile.block_duration_ns)
    start_offset = profile.extra_capture_duration_ns
    pulse_schedule = _FakePulseSchedule(
        duration=2 * block_duration,
        length=round((2 * block_duration) / SAMPLING_PERIOD_NS),
        ranges={},
    )
    capture_schedule = CaptureSchedule(
        captures=[
            Capture(
                channels=["RQ00"],
                start_time=0.0,
                duration=profile.workaround_capture_duration_ns,
            ),
            Capture(
                channels=["RQ00"],
                start_time=start_offset,
                duration=(2 * block_duration) - start_offset,
            ),
            Capture(
                channels=["B0.MNTR0.IN"],
                start_time=0.0,
                duration=profile.workaround_capture_duration_ns,
            ),
            Capture(
                channels=["B0.MNTR0.IN"],
                start_time=start_offset,
                duration=(2 * block_duration) - start_offset,
            ),
        ]
    )
    schedule = MeasurementSchedule.model_construct(
        pulse_schedule=pulse_schedule,
        capture_schedule=capture_schedule,
    )

    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=object(),  # type: ignore[arg-type]
        experiment_system=object(),  # type: ignore[arg-type]
        constraint_profile=profile,
    )

    result = adapter.validate_schedule(schedule)
    assert result is None


def test_build_measurement_result_keeps_monitor_labels() -> None:
    """Given monitor targets in backend result, when converting, then non-readout labels are preserved."""
    norm_factor = 2 ** (-32)
    backend_result = Quel1BackendExecutionResult(
        status={},
        data={
            "RQ00": [
                np.array([1.0 + 0.0j], dtype=np.complex128),
                np.array([4.0 + 2.0j], dtype=np.complex128),
            ],
            "B0.MNTR0.IN": [
                np.array([9.0 + 3.0j], dtype=np.complex128),
                np.array([3.0 + 1.0j], dtype=np.complex128),
            ],
        },
        config={},
    )

    class _TargetRegistry:
        @staticmethod
        def measurement_output_label(target: str) -> str:
            return "Q00" if target == "RQ00" else target

    @dataclass
    class _Target:
        sideband: str

    class _ExperimentSystemStub:
        target_registry = _TargetRegistry()

        @staticmethod
        def get_target(target: str) -> _Target:
            if target == "RQ00":
                return _Target(sideband="U")
            raise KeyError(target)

    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, object()),
        experiment_system=cast(Any, _ExperimentSystemStub()),
    )

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(mode="single", shots=4),
        device_config={"kind": "quel1"},
        sampling_period=2.0,
    )

    assert set(result.data.keys()) == {"Q00", "B0.MNTR0.IN"}
    assert_allclose(
        result.data["Q00"][0].data,
        np.array([4.0 + 2.0j], dtype=np.complex128) * norm_factor,
    )
    assert_allclose(
        result.data["B0.MNTR0.IN"][0].data,
        np.array([3.0 + 1.0j], dtype=np.complex128) * norm_factor,
    )


def test_create_sampled_sequences_accepts_monitor_capture_targets() -> None:
    """Given monitor capture labels, when building sampled sequences, then monitor channels are included."""
    profile = MeasurementConstraintProfile.quel1(sampling_period_ns=SAMPLING_PERIOD_NS)

    class _ExperimentSystemStub:
        control_params = SimpleNamespace(capture_delay_word={0: 1})

        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            if label == "Q00":
                return "Q00"
            raise ValueError(label)

        @staticmethod
        def get_mux_by_qubit(qubit: str) -> SimpleNamespace:
            assert qubit == "Q00"
            return SimpleNamespace(index=0)

        @staticmethod
        def get_diff_frequency(target: str) -> float:
            _ = target
            return 0.0

        @staticmethod
        def get_target(target: str) -> SimpleNamespace:
            if target == "Q00":
                return SimpleNamespace(sideband="U")
            raise KeyError(target)

        @staticmethod
        def get_awg_frequency(target: str) -> float:
            if target == "Q00":
                return 100e6
            raise KeyError(target)

        control_system = SimpleNamespace(
            get_port_by_id=lambda _label: SimpleNamespace(
                channels=(SimpleNamespace(ndelay=2),)
            )
        )

    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, object()),
            experiment_system=cast(Any, _ExperimentSystemStub()),
            constraint_profile=profile,
        ),
    )

    def _gen(
        self: Quel1MeasurementBackendAdapter,
        *,
        target_name: str,
        real: Any,
        imag: Any,
        modulation_frequency: float,
    ) -> dict[str, object]:
        _ = (self, real, imag)
        return {
            "target_name": target_name,
            "modulation_frequency": modulation_frequency,
        }

    def _cap(
        self: Quel1MeasurementBackendAdapter,
        *,
        target_name: str,
        modulation_frequency: float,
        capture_delay: int,
        capture_slots: list[tuple[int, int]],
    ) -> dict[str, object]:
        _ = self
        return {
            "target_name": target_name,
            "modulation_frequency": modulation_frequency,
            "capture_delay": capture_delay,
            "capture_slots": capture_slots,
        }

    adapter._create_gen_sampled_sequence = MethodType(_gen, adapter)  # noqa: SLF001
    adapter._create_cap_sampled_sequence = MethodType(_cap, adapter)  # noqa: SLF001

    with PulseSchedule(["Q00"]) as pulse_schedule:
        pulse_schedule.add("Q00", Blank(16))

    measurement_schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(channels=["Q00"], start_time=0.0, duration=8.0),
                Capture(channels=["B0.MNTR0.IN"], start_time=0.0, duration=8.0),
            ]
        ),
    )

    gen_sequences, cap_sequences = adapter._create_sampled_sequences(  # noqa: SLF001
        schedule=measurement_schedule
    )

    assert set(gen_sequences.keys()) == {"Q00"}
    assert set(cap_sequences.keys()) == {"Q00", "B0.MNTR0.IN"}


def test_create_sampled_sequences_uses_schedule_frequency_for_modulation_and_phase() -> (
    None
):
    """Given schedule frequency metadata, when building sampled sequences, then modulation and phase use that frequency."""
    profile = MeasurementConstraintProfile.quel1(sampling_period_ns=SAMPLING_PERIOD_NS)

    class _ExperimentSystemStub:
        control_params = SimpleNamespace(capture_delay_word={0: 0})

        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            if label == "Q00":
                return "Q00"
            raise ValueError(label)

        @staticmethod
        def get_mux_by_qubit(qubit: str) -> SimpleNamespace:
            assert qubit == "Q00"
            return SimpleNamespace(index=0)

        @staticmethod
        def get_diff_frequency(target: str) -> float:
            _ = target
            return 0.0

        @staticmethod
        def get_target(target: str) -> SimpleNamespace:
            if target == "Q00":
                return SimpleNamespace(sideband="U")
            raise KeyError(target)

        @staticmethod
        def get_awg_frequency(target: str) -> float:
            if target == "Q00":
                return 77e6
            raise KeyError(target)

        @staticmethod
        def get_nco_frequency(target: str) -> float:
            if target == "Q00":
                return 20e6
            raise KeyError(target)

        control_system = SimpleNamespace(
            get_port_by_id=lambda _label: SimpleNamespace(
                channels=(SimpleNamespace(ndelay=0, fnco_freq=20e6),)
            )
        )

    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, object()),
            experiment_system=cast(Any, _ExperimentSystemStub()),
            constraint_profile=profile,
        ),
    )

    recorded: dict[str, Any] = {}

    def _gen(
        self: Quel1MeasurementBackendAdapter,
        *,
        target_name: str,
        real: Any,
        imag: Any,
        modulation_frequency: float,
    ) -> dict[str, object]:
        _ = self
        recorded["target_name"] = target_name
        recorded["modulation_frequency"] = modulation_frequency
        recorded["real"] = np.asarray(real, dtype=np.float64)
        recorded["imag"] = np.asarray(imag, dtype=np.float64)
        return {"target_name": target_name}

    def _cap(
        self: Quel1MeasurementBackendAdapter,
        *,
        target_name: str,
        modulation_frequency: float,
        capture_delay: int,
        capture_slots: list[tuple[int, int]],
    ) -> dict[str, object]:
        _ = (self, target_name, modulation_frequency, capture_delay, capture_slots)
        return {}

    adapter._create_gen_sampled_sequence = MethodType(_gen, adapter)  # noqa: SLF001
    adapter._create_cap_sampled_sequence = MethodType(_cap, adapter)  # noqa: SLF001

    with PulseSchedule(["Q00"]) as pulse_schedule:
        pulse_schedule.add("Q00", Blank(8))
        pulse_schedule.add("Q00", Gaussian(duration=16, amplitude=0.2, sigma=4))
    pulse_schedule.set_frequency("Q00", 140e6)

    measurement_schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(
            captures=[Capture(channels=["Q00"], start_time=8.0, duration=16.0)]
        ),
    )

    _ = adapter._create_sampled_sequences(schedule=measurement_schedule)  # noqa: SLF001

    expected_modulation_frequency = 120e6
    expected = pulse_schedule.get_sampled_sequences()["Q00"].copy()
    for rng in pulse_schedule.get_pulse_ranges(["Q00"])["Q00"]:
        offset = rng.start * SAMPLING_PERIOD_NS
        expected[rng] *= np.exp(1j * 2 * np.pi * expected_modulation_frequency * offset)
    expected = np.conj(expected)

    assert recorded["target_name"] == "Q00"
    assert recorded["modulation_frequency"] == expected_modulation_frequency
    assert_allclose(recorded["real"], np.real(expected))
    assert_allclose(recorded["imag"], np.imag(expected))


def test_create_sampled_sequences_uses_zero_delay_for_entire_schedule() -> None:
    """Given full-span captures, when building sampled sequences, then capture delay is zero."""
    profile = MeasurementConstraintProfile.quel1(sampling_period_ns=SAMPLING_PERIOD_NS)

    class _ExperimentSystemStub:
        control_params = SimpleNamespace(capture_delay_word={0: 1})

        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            if label == "Q00":
                return "Q00"
            raise ValueError(label)

        @staticmethod
        def get_mux_by_qubit(qubit: str) -> SimpleNamespace:
            assert qubit == "Q00"
            return SimpleNamespace(index=0)

        @staticmethod
        def get_diff_frequency(target: str) -> float:
            _ = target
            return 0.0

        @staticmethod
        def get_target(target: str) -> SimpleNamespace:
            if target == "Q00":
                return SimpleNamespace(sideband="U")
            raise KeyError(target)

        @staticmethod
        def get_awg_frequency(target: str) -> float:
            if target == "Q00":
                return 100e6
            raise KeyError(target)

        control_system = SimpleNamespace(
            get_port_by_id=lambda _label: SimpleNamespace(
                channels=(SimpleNamespace(ndelay=2),)
            )
        )

    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, object()),
            experiment_system=cast(Any, _ExperimentSystemStub()),
            constraint_profile=profile,
        ),
    )

    def _gen(
        self: Quel1MeasurementBackendAdapter,
        *,
        target_name: str,
        real: Any,
        imag: Any,
        modulation_frequency: float,
    ) -> dict[str, object]:
        _ = (self, target_name, real, imag, modulation_frequency)
        return {}

    def _cap(
        self: Quel1MeasurementBackendAdapter,
        *,
        target_name: str,
        modulation_frequency: float,
        capture_delay: int,
        capture_slots: list[tuple[int, int]],
    ) -> dict[str, object]:
        _ = (self, target_name, modulation_frequency, capture_slots)
        return {"capture_delay": capture_delay}

    adapter._create_gen_sampled_sequence = MethodType(_gen, adapter)  # noqa: SLF001
    adapter._create_cap_sampled_sequence = MethodType(_cap, adapter)  # noqa: SLF001

    with PulseSchedule(["Q00"]) as pulse_schedule:
        pulse_schedule.add("Q00", Blank(128))

    capture_start = profile.extra_capture_duration_ns
    measurement_schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(
                    channels=["Q00"],
                    start_time=0.0,
                    duration=profile.workaround_capture_duration_ns,
                ),
                Capture(
                    channels=["Q00"],
                    start_time=capture_start,
                    duration=pulse_schedule.duration - capture_start,
                ),
            ]
        ),
    )

    _, cap_sequences = adapter._create_sampled_sequences(  # noqa: SLF001
        schedule=measurement_schedule
    )

    assert cap_sequences["Q00"]["capture_delay"] == 0


def test_create_sampled_sequences_skips_readout_phase_shift_when_capture_targets_do_not_overlap_pulse_labels() -> (
    None
):
    """Given no overlap between capture targets and pulse labels, when building sequences, then it does not look up missing capture delays."""
    profile = MeasurementConstraintProfile.quel1(sampling_period_ns=SAMPLING_PERIOD_NS)

    class _ExperimentSystemStub:
        control_params = SimpleNamespace(capture_delay_word={0: 1})

        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            if label == "RQ08":
                return "Q08"
            raise ValueError(label)

        @staticmethod
        def get_mux_by_qubit(qubit: str) -> SimpleNamespace:
            assert qubit == "Q08"
            return SimpleNamespace(index=0)

        @staticmethod
        def get_diff_frequency(target: str) -> float:
            _ = target
            return 0.0

        @staticmethod
        def get_target(target: str) -> SimpleNamespace:
            if target == "Q08":
                return SimpleNamespace(sideband="U")
            raise KeyError(target)

        @staticmethod
        def get_awg_frequency(target: str) -> float:
            if target == "Q08":
                return 100e6
            raise KeyError(target)

        control_system = SimpleNamespace(
            get_port_by_id=lambda _label: SimpleNamespace(
                channels=(SimpleNamespace(ndelay=2, fnco_freq=100e6),)
            )
        )

    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, object()),
            experiment_system=cast(Any, _ExperimentSystemStub()),
            constraint_profile=profile,
        ),
    )

    def _gen(
        self: Quel1MeasurementBackendAdapter,
        *,
        target_name: str,
        real: Any,
        imag: Any,
        modulation_frequency: float,
    ) -> dict[str, object]:
        _ = (self, target_name, real, imag, modulation_frequency)
        return {}

    def _cap(
        self: Quel1MeasurementBackendAdapter,
        *,
        target_name: str,
        modulation_frequency: float,
        capture_delay: int,
        capture_slots: list[tuple[int, int]],
    ) -> dict[str, object]:
        _ = (self, modulation_frequency, capture_slots)
        return {"target_name": target_name, "capture_delay": capture_delay}

    adapter._create_gen_sampled_sequence = MethodType(_gen, adapter)  # noqa: SLF001
    adapter._create_cap_sampled_sequence = MethodType(_cap, adapter)  # noqa: SLF001

    with PulseSchedule(["Q08"]) as pulse_schedule:
        pulse_schedule.add("Q08", Gaussian(duration=32, amplitude=0.2, sigma=8))

    measurement_schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(
            captures=[
                Capture(channels=["RQ08"], start_time=0.0, duration=32.0),
                Capture(channels=["B0.MNTR0.IN"], start_time=0.0, duration=32.0),
            ]
        ),
    )

    gen_sequences, cap_sequences = adapter._create_sampled_sequences(  # noqa: SLF001
        schedule=measurement_schedule
    )

    assert set(gen_sequences.keys()) == {"Q08"}
    assert set(cap_sequences.keys()) == {"RQ08", "B0.MNTR0.IN"}


def test_build_execution_request_resolves_read_in_port_id_for_resource_map() -> None:
    """Given loopback capture with READ_IN port IDs, building request should resolve resource-map lookup to read-in target labels."""

    class _BackendControllerStub:
        def __init__(self) -> None:
            self.targets: list[str] = []

        def get_resource_map(
            self, targets: list[str]
        ) -> dict[str, list[dict[str, str]]]:
            self.targets = list(targets)
            return {target: [{"resolved_target": target}] for target in targets}

    backend = _BackendControllerStub()
    read_in_target = SimpleNamespace(
        label="RQ00",
        channel=SimpleNamespace(port=SimpleNamespace(id="B0.READ0.IN")),
    )

    class _ExperimentSystemStub:
        def __init__(self) -> None:
            self.read_in_targets = [read_in_target]

        @staticmethod
        def get_target(target: str) -> SimpleNamespace:
            if target == "Q00":
                return SimpleNamespace(sideband="U")
            raise KeyError(target)

        @staticmethod
        def get_cap_target(target: str) -> SimpleNamespace:
            if target == "RQ00":
                return read_in_target
            raise KeyError(target)

    adapter = cast(
        Any,
        Quel1MeasurementBackendAdapter(
            backend_controller=cast(Any, backend),
            experiment_system=cast(Any, _ExperimentSystemStub()),
        ),
    )

    def _sampled_sequences(
        self: Quel1MeasurementBackendAdapter,
        *,
        schedule: MeasurementSchedule,
    ) -> tuple[dict[str, object], dict[str, object]]:
        _ = (self, schedule)
        return {}, {"B0.READ0.IN": object(), "B0.MNTR0.IN": object()}

    adapter._create_sampled_sequences = MethodType(  # noqa: SLF001
        _sampled_sequences,
        adapter,
    )

    with PulseSchedule(["Q00"]) as pulse_schedule:
        pulse_schedule.add("Q00", Blank(128.0))

    schedule = MeasurementSchedule(
        pulse_schedule=pulse_schedule,
        capture_schedule=CaptureSchedule(captures=[]),
    )

    request = adapter.build_execution_request(
        schedule=schedule,
        config=_make_config(mode="single", shots=1),
    )

    payload = request.payload
    assert backend.targets == ["RQ00", "B0.MNTR0.IN"]
    assert payload.resource_map["B0.READ0.IN"][0]["resolved_target"] == "RQ00"
    assert payload.resource_map["B0.MNTR0.IN"][0]["resolved_target"] == "B0.MNTR0.IN"

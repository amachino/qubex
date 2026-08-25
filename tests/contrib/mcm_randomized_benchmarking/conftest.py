"""Test fixtures for measurement-crosstalk randomized benchmarking."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qubex.pulse import Rect


class FakeCliffordGenerator:
    """Return deterministic one-qubit Clifford decompositions."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int | None]] = []

    def create_rb_sequences(
        self,
        n: int,
        type: str,
        seed: int | None,
    ) -> tuple[list[list[str]], list[str]]:
        """Return a reproducible gate list with variable Clifford durations."""
        self.calls.append((n, type, seed))
        assert type == "1Q"
        cliffords = [
            ["X90"] if index % 2 == 0 else ["Z90", "X90", "X90"] for index in range(n)
        ]
        return cliffords, ["Z90", "X90"]


class FakeRabiParam:
    """Normalize a synthetic IQ value to a Bloch-Z expectation value."""

    def normalize(self, iq: Any) -> float:
        """Return twice the encoded ground-state probability minus one."""
        return 2.0 * float(np.real(np.mean(np.asarray(iq)))) - 1.0


class FakePulseService:
    """Provide fixed-duration test pulses."""

    def __init__(self, *, measurement_duration: float = 64.0) -> None:
        self.measurement_duration = measurement_duration
        self.rabi_params = {f"Q{index}": FakeRabiParam() for index in range(5)}
        self.validated_targets: list[str] | None = None

    def validate_rabi_params(self, targets: list[str]) -> None:
        """Record the validated targets."""
        self.validated_targets = list(targets)

    def x90(self, target: str) -> Rect:
        """Return an 8 ns pi/2 pulse."""
        del target
        return Rect(duration=8.0, amplitude=0.5, sampling_period=2.0)

    def x180(self, target: str) -> Rect:
        """Return a 16 ns pi pulse."""
        del target
        return Rect(duration=16.0, amplitude=1.0, sampling_period=2.0)

    def readout(self, target: str) -> Rect:
        """Return a fixed-duration readout pulse."""
        del target
        return Rect(
            duration=self.measurement_duration,
            amplitude=0.25,
            sampling_period=2.0,
        )


class FakeExperimentSystem:
    """Resolve test qubit and readout labels."""

    @staticmethod
    def resolve_qubit_label(label: str) -> str:
        """Return the qubit corresponding to a test label."""
        return label.removeprefix("R")

    @staticmethod
    def resolve_read_label(label: str) -> str:
        """Return the readout label corresponding to a test label."""
        return label if label.startswith("R") else f"R{label}"


class FakeContext:
    """Provide the experiment context used by the public helpers."""

    def __init__(self) -> None:
        self.measurement = SimpleNamespace(sampling_period=2.0)
        self.experiment_system = FakeExperimentSystem()
        self.reset_calls: list[set[str]] = []

    def resolve_qubit_label(self, label: str) -> str:
        """Resolve a qubit label."""
        return self.experiment_system.resolve_qubit_label(label)

    def reset_awg_and_capunits(self, *, qubits: set[str]) -> None:
        """Record one hardware-reset request."""
        self.reset_calls.append(set(qubits))


@pytest.fixture
def fake_experiment() -> Any:
    """Provide a hardware-free experiment double for sequence tests."""
    pulse = FakePulseService()
    context = FakeContext()
    return SimpleNamespace(
        pulse=pulse,
        ctx=context,
        experiment_system=context.experiment_system,
        benchmarking_service=SimpleNamespace(
            clifford_generator=FakeCliffordGenerator()
        ),
        measurement_service=SimpleNamespace(),
    )

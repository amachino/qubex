"""Tests for CR crosstalk decomposition."""

import pytest

from qubex.contrib.experiment.cr_xt_decomposition import decompose_cr_crosstalk
from qubex.experiment.models import Result


class DummyQubit:
    """Minimal qubit model for testing."""

    def __init__(
        self,
        *,
        frequency: float,
        anharmonicity: float,
        control_frequency_ef: float,
    ) -> None:
        self.frequency = frequency
        self.anharmonicity = anharmonicity
        self.control_frequency_ef = control_frequency_ef


class DummyCtx:
    """Minimal context containing qubit information."""

    def __init__(self) -> None:
        self.qubits = {
            "Q0": DummyQubit(
                frequency=5.0,
                anharmonicity=-0.25,
                control_frequency_ef=4.7,
            ),
            "Q1": DummyQubit(
                frequency=4.8,
                anharmonicity=-0.22,
                control_frequency_ef=4.6,
            ),
        }


class DummyPulse:
    """Minimal pulse interface for testing."""

    def calc_control_amplitude(self, target: str, rabi_rate: float) -> float:
        """Return a fixed control amplitude."""
        return 0.5


class DummyCalibrationService:
    """Mock calibration service returning fixed CR tomography data."""

    def cr_hamiltonian_tomography(self, **kwargs) -> Result:
        """Return predefined Hamiltonian coefficients."""
        return Result(
            data={
                "coeffs": {
                    "IX": 0.010,
                    "IY": 0.020,
                    "ZX": 0.030,
                    "ZY": 0.040,
                }
            }
        )


class DummyExperiment:
    """Minimal experiment object combining required components."""

    def __init__(self) -> None:
        self.ctx = DummyCtx()
        self.pulse = DummyPulse()
        self.calibration_service = DummyCalibrationService()


def test_decompose_cr_crosstalk_computes_quantum_and_classical_components():
    """Compute quantum and classical crosstalk components correctly."""
    exp = DummyExperiment()

    result = decompose_cr_crosstalk(
        exp,  # type: ignore[arg-type]
        control_qubit="Q0",
        target_qubit="Q1",
        plot=False,
    )

    f_delta = 5.0 - 4.8
    alpha_c = -0.25

    expected_ix_quantum = 0.030 * f_delta / alpha_c
    expected_iy_quantum = 0.040 * f_delta / alpha_c

    expected_ix_classical = 0.010 - expected_ix_quantum
    expected_iy_classical = 0.020 - expected_iy_quantum

    assert result["IX_quantum"] == pytest.approx(expected_ix_quantum)
    assert result["IY_quantum"] == pytest.approx(expected_iy_quantum)
    assert result["IX_classical"] == pytest.approx(expected_ix_classical)
    assert result["IY_classical"] == pytest.approx(expected_iy_classical)

    assert result["IX_total"] == pytest.approx(
        result["IX_quantum"] + result["IX_classical"]
    )
    assert result["IY_total"] == pytest.approx(
        result["IY_quantum"] + result["IY_classical"]
    )

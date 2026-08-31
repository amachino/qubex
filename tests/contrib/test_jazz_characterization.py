"""Signed JAZZ analysis and model-inferred exchange without hardware access."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from qxpulse import Arbitrary

from qubex.contrib.experiment.jazz_characterization import (
    analyze_jazz_counts,
    build_jazz_schedule,
    fit_jazz_quadratures,
    infer_exchange_from_static_zz,
)


def _rotation(qubit: int, phase: float, angle: float) -> np.ndarray:
    axis = np.array([[0, np.exp(-1j * phase)], [np.exp(1j * phase), 0]])
    single = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * axis
    return np.kron(single, np.eye(2)) if qubit == 0 else np.kron(np.eye(2), single)


def _matrix_quadratures(
    total_ns: np.ndarray, zz_ghz: float, reference_ghz: float, target: int
) -> tuple[np.ndarray, np.ndarray]:
    """Independent ideal pulse matrices retain the service's emitted phase signs."""
    energies = np.array([0, 0.00043, 0.00031, 0.00074 + zz_ghz])
    initial = np.array([1, 0, 0, 0], complex)
    z = np.diag([1, -1])
    observable = np.kron(z, np.eye(2)) if target == 0 else np.kron(np.eye(2), z)
    echo = _rotation(0, 0, np.pi) @ _rotation(1, 0, np.pi)
    quadratures = []
    for axis in ("X", "Y"):
        values = []
        for total in total_ns:
            free = np.diag(np.exp(-2j * np.pi * energies * total / 2))
            before_analysis = (
                free @ echo @ free @ _rotation(target, 0, np.pi / 2) @ initial
            )
            phase = (
                np.pi if axis == "X" else -np.pi / 2
            ) - 2 * np.pi * reference_ghz * total
            final = _rotation(target, phase, np.pi / 2) @ before_analysis
            values.append(float(np.real(final.conj() @ observable @ final)))
        quadratures.append(np.array(values))
    return quadratures[0], quadratures[1]


@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize("zz_ghz", [-0.00012, 0.00012, 0.0005])
def test_signed_complex_fringe_recovers_energy_zz_in_both_directions(
    target: int, zz_ghz: float
) -> None:
    """X/Y phases recover signed energy ZZ rather than the cosine magnitude alias."""
    total = 2 * np.arange(0, 20001, 400, dtype=float)
    reference = 0.0002
    x, y = _matrix_quadratures(total, zz_ghz, reference, target)
    np.testing.assert_allclose(
        -x + 1j * y,
        np.exp(2j * np.pi * (zz_ghz / 2 - reference) * total),
        atol=2e-14,
        rtol=0,
    )
    result = fit_jazz_quadratures(total, x, y, rotation_frequency_ghz=reference)
    assert result["qualified"]
    assert result["zz_energy_ghz"] == pytest.approx(zz_ghz, abs=2e-9)
    assert result["signed_beat_frequency_ghz"] == pytest.approx(
        zz_ghz / 2 - reference, abs=1e-9
    )


@pytest.mark.parametrize("axis", ["X", "Y"])
def test_schedule_uses_saved_drag_samples_and_exact_analysis_phase(
    axis: Literal["X", "Y"],
) -> None:
    """The pure schedule keeps two tau waits and rotates the supplied DRAG waveform."""
    samples = np.linspace(0.02, 0.05, 12) + 0.01j * np.sin(np.arange(12))
    x90 = {q: Arbitrary(samples, sampling_period=2) for q in ("A", "P")}
    xpi = {q: Arbitrary(2 * samples, sampling_period=2) for q in ("A", "P")}
    tau, reference = 400.0, 0.0002
    schedule = build_jazz_schedule(
        "A",
        "P",
        tau,
        x90=x90,
        xpi=xpi,
        analysis_axis=axis,
        rotation_frequency_ghz=reference,
    )
    a = schedule.get_sequence("A").get_values(apply_frame_shifts=True)
    p = schedule.get_sequence("P").get_values(apply_frame_shifts=True)
    phase = (np.pi if axis == "X" else -np.pi / 2) - 2 * np.pi * reference * 2 * tau
    np.testing.assert_allclose(a[:12], samples, atol=1e-15, rtol=0)
    np.testing.assert_allclose(a[212:224], 2 * samples, atol=1e-15, rtol=0)
    np.testing.assert_allclose(p[212:224], 2 * samples, atol=1e-15, rtol=0)
    np.testing.assert_allclose(
        a[-12:], samples * np.exp(1j * phase), atol=1e-15, rtol=0
    )
    assert schedule.duration == 2 * tau + 72


@pytest.mark.parametrize("tau", [-2, 1, np.nan])
def test_schedule_rejects_invalid_half_wait(tau: float) -> None:
    """JAZZ half waits must be finite nonnegative values on the native 2 ns grid."""
    pulses = {q: Arbitrary(np.ones(12) * 0.1, sampling_period=2) for q in ("A", "P")}
    with pytest.raises(ValueError, match="tau_ns"):
        build_jazz_schedule("A", "P", tau, x90=pulses, xpi=pulses)


def test_schedule_rejects_unequal_echo_durations() -> None:
    """Unequal echo lengths cannot silently redefine the common free-evolution time."""
    x90 = {q: Arbitrary(np.ones(12) * 0.1, sampling_period=2) for q in ("A", "P")}
    xpi = {"A": x90["A"], "P": Arbitrary(np.ones(16) * 0.1, sampling_period=2)}
    with pytest.raises(ValueError, match="echo durations"):
        build_jazz_schedule("A", "P", 400, x90=x90, xpi=xpi)


@pytest.mark.parametrize("zz_ghz", [-0.00012, 0.0005])
def test_damped_noisy_complex_fit_retains_signed_frequency(zz_ghz: float) -> None:
    """Complex offsets and decay do not erase the sign of the fitted beat."""
    rng = np.random.default_rng(408)
    total = 2 * np.arange(0, 20001, 400, dtype=float)
    beat = zz_ghz / 2 - 0.0002
    observed = (0.04 - 0.03j) + 0.72 * np.exp(0.23j) * np.exp(-total / 25000) * np.exp(
        2j * np.pi * beat * total
    )
    observed += 0.008 * (rng.normal(size=total.size) + 1j * rng.normal(size=total.size))
    result = fit_jazz_quadratures(
        total,
        -observed.real,
        observed.imag,
        rotation_frequency_ghz=0.0002,
        standard_error_x=np.full(total.size, 0.008),
        standard_error_y=np.full(total.size, 0.008),
    )
    assert result["qualified"]
    assert result["zz_energy_ghz"] == pytest.approx(zz_ghz, abs=2e-6)
    assert result["zz_energy_standard_error_ghz"] > 0
    assert result["decay_time_ns"] == pytest.approx(25000, rel=0.1)


def test_reference_change_has_negative_signed_frequency_winding() -> None:
    """Changing programmed rotation changes the signed beat without changing energy ZZ."""
    total = 2 * np.arange(0, 20001, 400, dtype=float)
    results = []
    for reference in (0.0, 0.0002):
        x, y = _matrix_quadratures(total, -0.00012, reference, 0)
        results.append(
            fit_jazz_quadratures(total, x, y, rotation_frequency_ghz=reference)
        )
    assert results[1]["signed_beat_frequency_ghz"] - results[0][
        "signed_beat_frequency_ghz"
    ] == pytest.approx(-0.0002, abs=2e-9)
    assert results[0]["zz_energy_ghz"] == pytest.approx(
        results[1]["zz_energy_ghz"], abs=2e-9
    )


def test_high_r2_does_not_hide_weighted_model_lack_of_fit() -> None:
    """A weak second mode fails the weighted lack-of-fit diagnostic despite high R2."""
    total = 2 * np.arange(0, 20001, 400, dtype=float)
    observed = 0.72 * np.exp(-total / 30000) * np.exp(-2j * np.pi * 0.00026 * total)
    observed += 0.015 * np.exp(-2j * np.pi * 0.00019 * total)
    result = fit_jazz_quadratures(
        total,
        -observed.real,
        observed.imag,
        rotation_frequency_ghz=0.0002,
        standard_error_x=np.full(total.size, 0.0002),
        standard_error_y=np.full(total.size, 0.0002),
    )
    assert result["r2"] > 0.99
    assert not result["qualified"]
    assert result["reduced_chi2"] > 5
    assert result["nominal_gaussian_lack_of_fit_p_value"] < 1e-8
    assert "lack of fit" in " ".join(result["reasons"])


@pytest.mark.parametrize("target_index", [0, 1])
def test_raw_joint_counts_are_retained_with_target_bit_order(
    target_index: Literal[0, 1],
) -> None:
    """Count analysis keeps all raw counts while forming the requested target marginal."""
    tau = np.arange(0, 20001, 400, dtype=float)
    x, y = _matrix_quadratures(2 * tau, -0.00012, 0.0002, target_index)
    counts = []
    for values in (x, y):
        excited = np.rint(4096 * (1 - values) / 2).astype(int)
        data = np.zeros((len(tau), 4), int)
        data[:, 0] = 4096 - excited
        data[:, 2 if target_index == 0 else 1] = excited
        counts.append(data)
    result = analyze_jazz_counts(
        tau,
        counts[0],
        counts[1],
        target_index=target_index,
        rotation_frequency_ghz=0.0002,
    )
    np.testing.assert_array_equal(result["raw_counts_x"], counts[0])
    np.testing.assert_array_equal(result["raw_counts_y"], counts[1])
    np.testing.assert_array_equal(result["total_free_time_ns"], 2 * tau)
    assert result["zz_energy_ghz"] == pytest.approx(-0.00012, abs=1e-7)


def test_invalid_counts_and_alias_search_bounds_are_rejected() -> None:
    """Negative counts and frequency search beyond Nyquist are invalid input."""
    tau = np.arange(0, 20001, 400, dtype=float)
    bad = np.full((len(tau), 4), 100)
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="counts"):
        analyze_jazz_counts(
            tau, bad, bad, target_index=0, rotation_frequency_ghz=0.0002
        )
    with pytest.raises(ValueError, match="Nyquist"):
        fit_jazz_quadratures(
            2 * tau,
            np.sin(tau),
            np.cos(tau),
            rotation_frequency_ghz=0.0002,
            frequency_bounds_ghz=(-0.001, 0.001),
        )


def test_flat_quadratures_are_unqualified_not_zero_coupling() -> None:
    """No observed oscillation cannot silently become a zero-ZZ measurement."""
    total = 2 * np.arange(0, 20001, 400, dtype=float)
    result = fit_jazz_quadratures(
        total, np.ones_like(total), np.zeros_like(total), rotation_frequency_ghz=0.0002
    )
    assert not result["qualified"]
    assert result["zz_energy_ghz"] is None


def test_exchange_inversion_is_signed_and_invariant_under_pair_reversal() -> None:
    """The signed two-Duffing energy shift recovers the same coupling magnitude in both directions."""
    f1, f2, a1, a2, g = 4.41735, 5.006861, -0.22, -0.225, 0.008
    delta = f1 - f2
    zz = 2 * g**2 * (a1 + a2) / ((delta + a1) * (delta - a2))
    for fa, fp, aa, ap in ((f1, f2, a1, a2), (f2, f1, a2, a1)):
        result = infer_exchange_from_static_zz(
            zz,
            frequency_1_ghz=fa,
            frequency_2_ghz=fp,
            anharmonicity_1_ghz=aa,
            anharmonicity_2_ghz=ap,
            zz_energy_standard_error_ghz=abs(zz) / 100,
        )
        assert result["qualified"]
        assert result["g_magnitude_ghz"] == pytest.approx(g, rel=1e-12)
        assert result["g_standard_error_ghz"] == pytest.approx(g / 200, rel=1e-12)


def test_wrong_zz_sign_is_not_repaired_with_absolute_value() -> None:
    """A sign-incompatible static ZZ returns no coupling estimate."""
    result = infer_exchange_from_static_zz(
        0.0001,
        frequency_1_ghz=4.4,
        frequency_2_ghz=5.0,
        anharmonicity_1_ghz=-0.22,
        anharmonicity_2_ghz=-0.22,
    )
    assert not result["qualified"]
    assert result["g_magnitude_ghz"] is None
    assert "sign" in " ".join(result["reasons"])


def test_exchange_formula_matches_weak_coupling_exact_energy_diagonalization() -> None:
    """The inversion agrees with the independent two-excitation Duffing matrix."""
    f1, f2, a1, a2, g = 4.41735, 5.006861, -0.22, -0.225, 0.001
    two_excitation = np.array(
        [
            [2 * f1 + a1, np.sqrt(2) * g, 0],
            [np.sqrt(2) * g, f1 + f2, np.sqrt(2) * g],
            [0, np.sqrt(2) * g, 2 * f2 + a2],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(two_excitation)
    energy_11 = eigenvalues[np.argmax(np.abs(eigenvectors[1]) ** 2)]
    # The complete one-excitation pair has trace f1+f2, independent of mixing.
    zz = float(energy_11 - f1 - f2)
    result = infer_exchange_from_static_zz(
        zz,
        frequency_1_ghz=f1,
        frequency_2_ghz=f2,
        anharmonicity_1_ghz=a1,
        anharmonicity_2_ghz=a2,
    )
    assert zz < 0
    assert result["qualified"]
    assert result["g_magnitude_ghz"] == pytest.approx(g, rel=3e-5)


def test_nonfinite_or_nonuniform_time_data_are_rejected() -> None:
    """Invalid time grids and NaN quadratures fail before a numerical fit."""
    total = np.arange(12, dtype=float) * 800
    bad = np.ones_like(total)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_jazz_quadratures(
            total, bad, np.ones_like(total), rotation_frequency_ghz=0.0002
        )
    total[3] += 4
    with pytest.raises(ValueError, match="uniform"):
        fit_jazz_quadratures(
            total,
            np.ones_like(total),
            np.ones_like(total),
            rotation_frequency_ghz=0.0002,
        )


def test_small_zz_and_breakdown_of_dispersive_model_remain_unqualified() -> None:
    """Unresolved sign or resonant hybridization cannot qualify a dispersive coupling estimate."""
    arguments = dict(
        frequency_1_ghz=4.4,
        frequency_2_ghz=5.0,
        anharmonicity_1_ghz=-0.22,
        anharmonicity_2_ghz=-0.22,
    )
    uncertain = infer_exchange_from_static_zz(
        -1e-7, **arguments, zz_energy_standard_error_ghz=1e-6
    )
    assert not uncertain["qualified"]
    assert uncertain["g_standard_error_ghz"] is None
    large = infer_exchange_from_static_zz(-0.05, **arguments)
    assert not large["qualified"]
    assert large["maximum_dispersive_ratio"] > 0.1

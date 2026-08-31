"""Offline tests for a failure-only diagnostic; no device connection is used."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qxpulse import Arbitrary

from qubex.contrib.experiment.bswap_calibration import parity_diagnostics as module
from qubex.contrib.experiment.bswap_calibration.tomography import BASES, PAULI


class CountPort:
    """Compile real pulses but synthesize counts from declared diagnostic states."""

    def __init__(self) -> None:
        self.qubits = ("A", "P")
        self.drive_label, self.cancel_label = "D", "C"
        self.targets = {"D": 4.613, "C": 4.613}
        self.references = {"A": 4.41735, "P": 5.006861}
        self.rabi_scale = 0.6358463897962201
        self.backend_preamble_ns = 40.0
        self.shot_interval_ns = 1_000_000.0
        self.session_id = "synthetic-diagnostic"
        self.assignment_source = "synthetic-current-baseline"
        self.classifiers: dict[str, Any] = {}
        self.x90 = {
            q: Arbitrary(np.full(12, 0.08 + 0.01j), sampling_period=2.0)
            for q in self.qubits
        }
        self.xpi = {
            q: Arbitrary(np.full(12, 0.16 + 0.02j), sampling_period=2.0)
            for q in self.qubits
        }
        self.recipes: dict[str, Any] = {
            "bswap": dict(
                gate_kind="bswap",
                amplitude=0.989,
                frequency_ghz=4.61250206847794,
                duration_ns=262.0,
                ramp_ns=16.0,
                gate_start_ns=24.0,
                cd_strength=1.0,
                design_delta_scale=1.0,
                window={"type": "hann"},
                cancel_amplitude_ratio=0.023,
                cancel_phase_rad=1.3,
                zz_phase_rad=0.015,
                phase_calibration=dict(
                    pre_active_rad=0.0,
                    post_active_rad=-0.03173099811937686,
                    post_passive_rad=-2.5297480273023645,
                ),
            )
        }
        self.calls: list[dict[str, Any]] = []
        self.timeout = self.malformed = self.mutate_scale = False

    def acquire(
        self,
        gates: Any,
        directory: Path,
        label: str,
        *,
        prepared: tuple[str, str],
        basis: str,
        shots: int,
        recipes: Any,
    ) -> dict[str, Any]:
        """Generate reproducible multinomial counts and retain synthetic IQ."""
        self.calls.append(
            dict(gates=gates, label=label, prepared=prepared, basis=basis, shots=shots)
        )
        if self.timeout:
            raise TimeoutError("synthetic deadline")
        if self.mutate_scale:
            self.rabi_scale += 0.01
        variant = label.rsplit("_", 1)[-1]
        p = {"standard": 0.6, "parity": 0.8, "idle": 0.9}[variant]
        rho = np.diag([1 - p, p, 0, 0])
        projectors = [
            np.kron(
                (PAULI["I"] + sa * PAULI[basis[0]]) / 2,
                (PAULI["I"] + sp * PAULI[basis[1]]) / 2,
            )
            for sa, sp in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ]
        probabilities = np.array([np.trace(rho @ op).real for op in projectors])
        counts = np.random.default_rng(len(self.calls)).multinomial(
            shots, probabilities
        )
        directory.mkdir(parents=True, exist_ok=True)
        iq_file = directory / f"{label}.npz"
        np.savez(iq_file, iq=np.zeros((2, shots), dtype=complex))
        return dict(
            counts=counts[:2].tolist() if self.malformed else counts.tolist(),
            shots=shots,
            iq_file=str(iq_file),
            basis=basis,
            prepared=list(prepared),
        )


def test_exact_compilation_and_adjacent_three_variant_acquisition(
    tmp_path: Path,
) -> None:
    """All actual schedules pass preflight before balanced adjacent measurements."""
    port = CountPort()
    before = deepcopy(port.recipes)
    result = module.acquire_bswap_odd_sector_diagnostic(port, tmp_path)
    assert len(port.calls) == 54
    assert result["requested_shots"] == 55_296
    assert result["diagnostic_only"]
    assert not result["scientific_qualified"]
    assert port.recipes == before
    assert result["preflight"]["gate_body_duration_ns"] == 1048
    assert result["preflight"]["expected_tone_signs"] == [1, -1, 1, -1]
    assert result["preflight"]["maximum_ge_difference"] < 1e-12
    assert result["preflight"]["maximum_tone_difference"] < 1e-12
    for offset in range(0, 54, 3):
        rows = port.calls[offset : offset + 3]
        assert len({row["basis"] for row in rows}) == 1
        assert {row["label"].rsplit("_", 1)[-1] for row in rows} == {
            "standard",
            "parity",
            "idle",
        }
        assert all(row["prepared"] == ("0", "1") for row in rows)
    orders = [
        tuple(row["label"].rsplit("_", 1)[-1] for row in port.calls[n : n + 3])
        for n in range(0, 54, 3)
    ]
    assert len(set(orders)) == 6
    assert all(orders.count(order) == 3 for order in set(orders))
    for variant, expected in (("standard", 0.6), ("parity", 0.8), ("idle", 0.9)):
        summary = result["variant_summaries"][variant]
        assert summary["raw_probabilities"][1] == pytest.approx(expected, abs=0.04)
        assert np.asarray(summary["rho_real"]).shape == (4, 4)
        assert len(summary["replicates"]) == 2
    assert result["comparisons"]["parity_minus_standard"]["P01"]["difference"] > 0.1
    assert Path(result["summary_path"]).exists()
    with np.load(tmp_path / "counts_and_density.npz") as arrays:
        assert arrays["counts"].shape == (3, 2, 9, 4)
        assert arrays["raw_linear_density"].shape == (3, 2, 4, 4)
    assert len(list((tmp_path / "shots").glob("*.npz"))) == 54


@pytest.mark.parametrize("failure", ["headroom", "timing", "cancel", "frame"])
def test_preflight_failure_requests_no_shots(tmp_path: Path, failure: str) -> None:
    """Unsafe or non-equivalent emitted schedules stop before any acquisition."""
    port = CountPort()
    recipe = port.recipes["bswap"]
    if failure == "headroom":
        recipe["amplitude"] = 1.1
    elif failure == "timing":
        recipe["duration_ns"] = 263.0
    elif failure == "cancel":
        recipe["cancel_amplitude_ratio"] = 0.0
    else:
        recipe["phase_calibration"].update(post_active_rad=0.7, post_passive_rad=-0.8)
    with pytest.raises(ValueError, match=r"headroom|amplitude|grid|tone|sign|GE"):
        module.acquire_bswap_odd_sector_diagnostic(port, tmp_path)
    assert port.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [{"max_total_shots": 55_295}, {"shots": 0}, {"shots": 1.5}, {"replicates": True}],
)
def test_invalid_budget_or_sampling_fails_before_shots(
    tmp_path: Path, kwargs: Any
) -> None:
    """An insufficient total budget or malformed count setting is never acquired."""
    port = CountPort()
    with pytest.raises(ValueError, match=r"shots|replicates"):
        module.acquire_bswap_odd_sector_diagnostic(port, tmp_path, **kwargs)
    assert not port.calls


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("timeout", TimeoutError),
        ("malformed", ValueError),
        ("mutate_scale", RuntimeError),
    ],
)
def test_failure_is_saved_and_propagated(
    tmp_path: Path, failure: str, error: Any
) -> None:
    """Deadline, malformed counts and identity changes leave evidence and stop."""
    port = CountPort()
    setattr(port, failure, True)
    with pytest.raises(error):
        module.acquire_bswap_odd_sector_diagnostic(port, tmp_path)
    assert len(port.calls) == 1
    assert (tmp_path / "failure.json").exists()
    assert not (tmp_path / "summary.json").exists()


def test_existing_protocol_is_not_overwritten(tmp_path: Path) -> None:
    """A previous diagnostic directory cannot silently become a new acquisition."""
    (tmp_path / "protocol.json").write_text("keep")
    port = CountPort()
    with pytest.raises(FileExistsError):
        module.acquire_bswap_odd_sector_diagnostic(port, tmp_path)
    assert not port.calls
    assert (tmp_path / "protocol.json").read_text() == "keep"


def test_count_statistics_retain_coherences_without_physical_projection() -> None:
    """Raw linear conditional coherences retain complex signs and analytic shot errors."""
    vector = np.array([1, 1j, 0, 0]) / np.sqrt(2)
    rho = np.outer(vector, vector.conj())
    counts = [
        [
            4096
            * np.trace(
                rho
                @ np.kron(
                    (PAULI["I"] + sa * PAULI[basis[0]]) / 2,
                    (PAULI["I"] + sp * PAULI[basis[1]]) / 2,
                )
            ).real
            for sa, sp in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ]
        for basis in BASES
    ]
    # A pure analytic covariance check is independent of the acquisition fixture.
    summary = module._count_summary(np.asarray(counts))  # noqa: SLF001
    assert summary["conditional_coherences"]["passive_given_active_0"][
        "imag"
    ] == pytest.approx(0.5)
    np.testing.assert_allclose(
        np.asarray(summary["rho_real"]) + 1j * np.asarray(summary["rho_imag"]),
        rho,
        atol=1e-12,
    )
    assert (
        summary["conditional_coherences"]["passive_given_active_0"]["imag_shot_se"] >= 0
    )

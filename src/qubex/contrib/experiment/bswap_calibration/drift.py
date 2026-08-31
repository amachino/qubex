"""
Fixed-waveform bidirectional transfer-peak anchors for a bSWAP campaign.

Only the supplied measurement callback owns hardware access. These functions
never connect, configure, retune calibration recipes, or schedule repeated work.
A moving transfer optimum is not proof of resonance drift or its amplitude cause.
"""

import json
import pickle
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from qxpulse import Waveform
from scipy.stats import chi2


class AnchorMeasurements(Protocol):
    """Read-only calibration context and count-preserving acquisition callback."""

    @property
    def qubits(self) -> Sequence[str]:
        """Return active/passive qubit labels in measurement order."""
        ...

    @property
    def recipes(self) -> dict[str, Any]:
        """Return frozen full and root pulse recipes."""
        ...

    @property
    def references(self) -> Mapping[str, float]:
        """Return qubit reference frequencies in GHz."""
        ...

    @property
    def targets(self) -> Mapping[str, float]:
        """Return drive-target reference frequencies in GHz."""
        ...

    @property
    def rabi_scale(self) -> float:
        """Return the fixed GHz-per-command Rabi conversion."""
        ...

    @property
    def session_id(self) -> str:
        """Return the owning connection identifier."""
        ...

    @property
    def x90(self) -> Mapping[str, Waveform]:
        """Return calibrated production X90 waveforms."""
        ...

    @property
    def xpi(self) -> Mapping[str, Waveform]:
        """Return calibrated production Xpi waveforms."""
        ...

    @property
    def classifiers(self) -> Mapping[str, Any]:
        """Return serializable classifier states for baseline fingerprints."""
        ...

    def acquire(
        self,
        gates: Sequence[str],
        directory: str | Path,
        label: str,
        *,
        prepared: tuple[str, str],
        basis: str,
        shots: int,
        recipes: dict[str, Any],
    ) -> dict[str, Any]:
        """Acquire and retain raw counts using the caller's existing connection."""
        ...


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _fingerprint(value: Any) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_encode
    )
    return sha256(serialized.encode()).hexdigest()


def _save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=_encode))
    temporary.replace(path)


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _pulse_identity(collection: Mapping[str, Waveform]) -> dict[str, Any]:
    return {
        q: dict(
            duration_ns=float(pulse.duration),
            sampling_period_ns=float(pulse.sampling_period),
            samples_sha256=sha256(
                np.asarray(pulse.values, dtype="<c16").tobytes()
            ).hexdigest(),
        )
        for q, pulse in collection.items()
    }


def _baseline_context(
    measurements: AnchorMeasurements, recipes: dict[str, Any]
) -> dict[str, Any]:
    """Identify comparable settings, not certify that the physical state is stable."""
    context = dict(
        recipe_fingerprint=_fingerprint(recipes["bswap"]),
        qubits=list(measurements.qubits),
        references_ghz=deepcopy(measurements.references),
        targets_ghz=deepcopy(measurements.targets),
        rabi_ghz_per_command=float(measurements.rabi_scale),
        connection_session_id=str(measurements.session_id),
        preparation_window_ns=max(float(r["gate_start_ns"]) for r in recipes.values()),
        x90=_pulse_identity(measurements.x90),
        xpi=_pulse_identity(measurements.xpi),
        classifiers_sha256={
            q: sha256(pickle.dumps(classifier, protocol=5)).hexdigest()
            for q, classifier in measurements.classifiers.items()
        },
    )
    return context


def _quadratic_peak(
    offsets: ArrayLike, probabilities: ArrayLike, variances: ArrayLike
) -> dict[str, Any]:
    """Fit one predeclared window and retain an unqualified vertex as diagnostic only."""
    x = np.asarray(offsets, dtype=float)
    y = np.asarray(probabilities, dtype=float)
    variance = np.asarray(variances, dtype=float)
    design = np.column_stack((np.ones(len(x)), x, x * x))
    weighted = design / np.sqrt(variance[:, None])
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted, y / np.sqrt(variance), rcond=None
    )
    if rank != 3:
        raise ValueError("Frequency window does not identify a quadratic")
    residual = y - design @ coefficients
    statistic = float(np.sum(residual * residual / variance))
    dof = len(x) - 3
    covariance = np.linalg.inv(weighted.T @ weighted) * max(1.0, statistic / dof)
    errors = np.sqrt(np.diag(covariance))
    _, slope, curvature = coefficients
    reasons = []
    if curvature + 1.96 * errors[2] >= 0:
        reasons.append("concave_curvature_not_resolved")
    vertex = None
    sem = None
    interval = None
    if curvature != 0:
        vertex = float(-slope / (2 * curvature))
        gradient = np.array([0.0, -1 / (2 * curvature), slope / (2 * curvature**2)])
        sem = float(np.sqrt(max(0.0, gradient @ covariance @ gradient)))
        interval = [vertex - 1.96 * sem, vertex + 1.96 * sem]
        if not np.isfinite([vertex, sem, *interval]).all():
            vertex = sem = interval = None
    if vertex is None or interval is None:
        reasons.append("peak_uncertainty_unidentified")
    elif not (x.min() < interval[0] <= vertex <= interval[1] < x.max()):
        reasons.append("peak_not_interior_with_uncertainty")
    lack_of_fit_p = float(chi2.sf(statistic, dof))
    if lack_of_fit_p < 0.01:
        reasons.append("quadratic_lack_of_fit")
    return dict(
        qualified=not reasons,
        reasons=reasons,
        coefficients=coefficients.tolist(),
        covariance=covariance.tolist(),
        curvature_per_mhz2=float(curvature),
        curvature_sem=float(errors[2]),
        diagnostic_vertex_offset_mhz=vertex,
        center_sem_mhz=sem,
        diagnostic_interval_95_mhz=interval,
        reduced_chi2=statistic / dof,
        lack_of_fit_p=lack_of_fit_p,
        probabilities=y.tolist(),
        residuals=residual.tolist(),
    )


def fit_resonance_anchor(
    frequencies_ghz: ArrayLike, counts: ArrayLike
) -> dict[str, Any]:
    """
    Estimate a local transfer optimum with shot and fit-window diagnostics.

    Parameters
    ----------
    frequencies_ghz : array_like
        At least seven strictly increasing carrier frequencies in GHz.
    counts : array_like
        Integer counts with shape (2, frequency, 4), without postselection.

    Returns
    -------
    dict
        Qualified carrier center and interval in GHz, or explicit rejection
        reasons with diagnostic fits and no qualified center.

    Notes
    -----
    The approximate normal interval uses binomial shot uncertainty and a local
    quadratic; it does not cover unknown SPAM or physical line-model bias.
    """
    frequencies = np.asarray(frequencies_ghz, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if (
        frequencies.ndim != 1
        or len(frequencies) < 7
        or not np.isfinite(frequencies).all()
        or np.any(np.diff(frequencies) <= 0)
    ):
        raise ValueError("Need at least seven increasing finite frequencies")
    if (
        counts.shape != (2, len(frequencies), 4)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError(
            "Need complete nonnegative integer bidirectional four-outcome counts"
        )
    totals = counts.sum(axis=-1)
    if np.any(totals <= 0):
        raise ValueError("Each direction/frequency needs shots")
    successes = np.stack((counts[0, :, 3], counts[1, :, 0]))
    probabilities = successes / totals
    # Jeffreys binomial variance stays finite at zero/all-success counts.
    variance = (
        (successes + 0.5)
        * (totals - successes + 0.5)
        / ((totals + 1) ** 2 * (totals + 2))
    )
    origin = float((frequencies[0] + frequencies[-1]) / 2)
    offsets = (frequencies - origin) * 1e3
    directions = [
        _quadratic_peak(offsets, p, v)
        for p, v in zip(probabilities, variance, strict=True)
    ]
    mean = probabilities.mean(axis=0)
    mean_variance = variance.sum(axis=0) / 4
    pooled = _quadratic_peak(offsets, mean, mean_variance)
    inner = _quadratic_peak(offsets[1:-1], mean[1:-1], mean_variance[1:-1])
    reasons = [
        f"direction_{i}:{reason}"
        for i, fit in enumerate(directions)
        for reason in fit["reasons"]
    ]
    reasons += [f"pooled:{reason}" for reason in pooled["reasons"]]
    reasons += [f"inner_window:{reason}" for reason in inner["reasons"]]
    if all(fit["qualified"] for fit in directions):
        difference = abs(
            directions[0]["diagnostic_vertex_offset_mhz"]
            - directions[1]["diagnostic_vertex_offset_mhz"]
        )
        uncertainty = np.hypot(*(fit["center_sem_mhz"] for fit in directions))
        if difference > 1.96 * uncertainty:
            reasons.append("bidirectional_centers_disagree")
    if pooled["qualified"] and inner["qualified"]:
        difference = abs(
            pooled["diagnostic_vertex_offset_mhz"]
            - inner["diagnostic_vertex_offset_mhz"]
        )
        # Nested windows are correlated: this is a conservative sensitivity
        # sentinel, not an independent-hypothesis-test p value.
        uncertainty = np.hypot(pooled["center_sem_mhz"], inner["center_sem_mhz"])
        if difference > 1.96 * uncertainty:
            reasons.append("fit_window_sensitive_center")
    qualified = not reasons
    center = (
        origin + pooled["diagnostic_vertex_offset_mhz"] / 1e3 if qualified else None
    )
    interval = (
        [origin + value / 1e3 for value in pooled["diagnostic_interval_95_mhz"]]
        if qualified
        else None
    )
    return dict(
        qualified=qualified,
        reasons=reasons,
        center_ghz=center,
        center_sem_mhz=pooled["center_sem_mhz"] if qualified else None,
        center_interval_95_ghz=interval,
        reference_carrier_ghz=origin,
        directions=directions,
        pooled=pooled,
        inner_window=inner,
        claim="fixed-waveform bidirectional transfer-peak anchor, not a unique Hamiltonian resonance or amplitude-drift diagnosis",
        uncertainty_scope="conditional independent-binomial shot/fit uncertainty, not natural temporal variability; excludes unknown SPAM drift and line-model systematics",
    )


def _validated_anchor_counts(counts: ArrayLike, shots: int) -> NDArray[np.float64]:
    observed = np.asarray(counts, dtype=float)
    if (
        observed.shape != (4,)
        or not np.isfinite(observed).all()
        or np.any(observed < 0)
        or np.any(observed != np.floor(observed))
        or observed.sum() != shots
    ):
        raise ValueError("Anchor callback returned invalid counts or shot total")
    return observed


def record_resonance_anchor(
    measurements: AnchorMeasurements,
    directory: str | Path,
    *,
    shots: int = 256,
    span_mhz: float = 1.5,
    npoints: int = 9,
) -> dict[str, Any]:
    """
    Acquire one bounded full-bSWAP anchor through the current session callback.

    Parameters
    ----------
    measurements : AnchorMeasurements
        Existing measurement owner; this function never connects or configures.
    directory : str or Path
        Parent directory for a new timestamped checkpoint directory.
    shots : int, default 256
        Positive shot count per frequency and preparation direction.
    span_mhz : float, default 1.5
        Positive frequency half-width in MHz.
    npoints : int, default 9
        Odd number of frequencies, at least seven.

    Returns
    -------
    dict
        Fit qualification, baseline identity, timestamps and saved data paths.

    Notes
    -----
    Writes raw-count checkpoints through the supplied callback and files.
    No calibration recipe is updated and this
    function does not arrange subsequent anchors. All pulse design parameters
    remain frozen; only carrier frequency changes via the existing compiler.
    """
    shots = _positive_integer(shots, "shots")
    npoints = _positive_integer(npoints, "npoints", minimum=7)
    if npoints % 2 == 0:
        raise ValueError("npoints must be odd")
    span_mhz = float(span_mhz)
    if not np.isfinite(span_mhz) or span_mhz <= 0:
        raise ValueError("span_mhz must be finite and positive")
    recipes = deepcopy(measurements.recipes)
    recipe = recipes["bswap"]
    center = float(recipe["frequency_ghz"])
    reference = float(measurements.references[measurements.qubits[0]])
    scale = float(recipe.get("design_delta_scale", 1.0))
    delta_design = scale * (reference - center)
    frequencies = center + np.linspace(-span_mhz, span_mhz, npoints) / 1e3
    physical_detunings = reference - frequencies
    if (
        not np.isfinite([center, reference, scale, delta_design]).all()
        or scale <= 0
        or delta_design == 0
        or not np.isfinite(physical_detunings).all()
        or np.any(np.sign(physical_detunings) != np.sign(delta_design))
    ):
        raise ValueError(
            "Carrier probe crosses zero detuning or cannot preserve signed SQUAD design"
        )
    probe_scales = delta_design / physical_detunings
    if not np.isfinite(probe_scales).all():
        raise ValueError("Carrier probe cannot preserve finite SQUAD design scale")
    baseline = _baseline_context(measurements, recipes)
    started = datetime.now().astimezone()
    anchor_dir = (
        Path(directory) / f"resonance_anchor_{started.strftime('%Y%m%d_%H%M%S_%f')}"
    )
    anchor_dir.mkdir(parents=True, exist_ok=False)
    data_file = anchor_dir / "resonance_anchor.npz"
    summary_file = anchor_dir / "resonance_anchor.json"
    order = np.random.default_rng(17).permutation(npoints).tolist()
    counts = np.full((2, npoints, 4), np.nan)
    timestamps = np.full((2, npoints), "", dtype="U80")
    rows = []
    summary = dict(
        schema_version=1,
        timestamp=started.isoformat(),
        started_at=started.isoformat(),
        completed_at=None,
        baseline_id=_fingerprint(baseline),
        baseline_context=baseline,
        recipe_fingerprint=baseline["recipe_fingerprint"],
        recipe=recipe,
        frozen_design_delta_rad_per_ns=float(2 * np.pi * delta_design),
        probe_design_delta_scales=probe_scales.tolist(),
        frequencies_ghz=frequencies.tolist(),
        frequency_order=order,
        span_mhz=span_mhz,
        shots_per_direction=shots,
        anchor_directory=str(anchor_dir),
        data_file=str(data_file),
        qualified=False,
        fit=None,
        status="acquiring",
        baseline_identity_scope="matching captured recipes, frequency settings, DRAG waveforms, classifier learned state and connection; not proof of physical readout/RF stability",
        claim="settings-matched transfer-peak time series; amplitude causality unmeasured",
    )

    def checkpoint() -> None:
        np.savez_compressed(
            data_file,
            frequencies_ghz=frequencies,
            counts=counts,
            timestamps=timestamps,
            probe_design_delta_scales=probe_scales,
            frequency_order=order,
            shots=shots,
        )
        _save_json(anchor_dir / "resonance_anchor_points.json", rows)
        _save_json(summary_file, summary)

    checkpoint()
    try:
        for position, fi in enumerate(order):
            for direction in (0, 1) if position % 2 == 0 else (1, 0):
                selected = deepcopy(recipes)
                selected["bswap"]["frequency_ghz"] = float(frequencies[fi])
                selected["bswap"]["design_delta_scale"] = float(probe_scales[fi])
                prepared = ("0", "0") if direction == 0 else ("1", "1")
                before = datetime.now().astimezone().isoformat()
                row = measurements.acquire(
                    ["BSWAP"],
                    anchor_dir,
                    f"anchor_f{fi}_direction{direction}",
                    prepared=prepared,
                    basis="ZZ",
                    shots=shots,
                    recipes=selected,
                )
                after = datetime.now().astimezone().isoformat()
                rows.append(
                    dict(
                        frequency_index=fi,
                        direction=direction,
                        frequency_ghz=float(frequencies[fi]),
                        requested_at=before,
                        returned_at=after,
                        measurement=row,
                    )
                )
                observed = _validated_anchor_counts(row["counts"], shots)
                counts[direction, fi] = observed
                timestamps[direction, fi] = row.get("timestamp", after)
                checkpoint()
        fit = fit_resonance_anchor(frequencies, counts)
        _save_json(anchor_dir / "resonance_anchor_fit.json", fit)
        summary.update(fit=fit, qualified=fit["qualified"], status="complete")
    except Exception as error:
        summary.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        completed = datetime.now().astimezone()
        summary["completed_at"] = completed.isoformat()
        summary["timestamp"] = (started + (completed - started) / 2).isoformat()
        checkpoint()
    return summary


def plot_resonance_history(
    records_or_directory: Sequence[dict[str, Any]] | str | Path,
    *,
    output_html: str | Path | None = None,
) -> go.Figure:
    """Plot qualified center offsets separately for each comparable baseline ID."""
    if isinstance(records_or_directory, (str, Path)):
        records = [
            json.loads(path.read_text())
            for path in Path(records_or_directory).rglob("resonance_anchor.json")
        ]
    else:
        records = list(records_or_directory)
    records.sort(key=lambda row: datetime.fromisoformat(row["timestamp"]))
    groups = {}
    for row in records:
        if row.get("qualified") and row.get("fit", {}).get("center_ghz") is not None:
            groups.setdefault(row["baseline_id"], []).append(row)
    figure = go.Figure()
    for baseline_id, group in groups.items():
        figure.add_scatter(
            x=[row["timestamp"] for row in group],
            y=[
                1e3 * (row["fit"]["center_ghz"] - row["recipe"]["frequency_ghz"])
                for row in group
            ],
            error_y=dict(
                type="data", array=[row["fit"]["center_sem_mhz"] for row in group]
            ),
            mode="lines+markers",
            name=f"baseline {baseline_id[:10]}",
            customdata=[row["anchor_directory"] for row in group],
            hovertemplate="%{x}<br>offset=%{y:.4f} MHz<br>%{customdata}<extra>%{fullData.name}</extra>",
        )
    rejected = len(records) - sum(len(group) for group in groups.values())
    figure.update_layout(
        title="Fixed-waveform transfer-peak anchors (amplitude cause unmeasured)",
        xaxis_title="Acquisition midpoint (recorded timezone)",
        yaxis_title="Peak offset from frozen recipe carrier (MHz; error bars: 1 SEM)",
        annotations=[
            dict(
                text=f"{rejected}/{len(records)} anchors unqualified; distinct baselines are not joined",
                xref="paper",
                yref="paper",
                x=0,
                y=1.12,
                showarrow=False,
            )
        ],
    )
    if output_html is not None:
        output = Path(output_html)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(output)
    return figure

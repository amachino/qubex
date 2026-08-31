"""Offline local quadratic fit of bidirectional bSWAP population maps."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fit_local_map(
    amplitudes: ArrayLike,
    frequencies_ghz: ArrayLike,
    transfer: ArrayLike,
    shots_per_direction: int,
    score_variance: ArrayLike | None = None,
) -> dict[str, Any]:
    """
    Rank bounded amplitude/frequency candidates with a local quadratic fit.

    Parameters
    ----------
    amplitudes : array_like
        Command-amplitude coordinates with a nonzero span.
    frequencies_ghz : array_like
        Carrier coordinates in GHz with a nonzero span.
    transfer : array_like
        Directional scores, shape (2, amplitude, frequency).
    shots_per_direction : int
        Positive shot count for raw binomial-score weights.
    score_variance : array_like, optional
        Independent score variances with the same shape as transfer; required
        when the score is not a raw binomial population.

    Returns
    -------
    dict
        Candidate command amplitude and frequency in GHz, fitted surfaces,
        residuals and a local pointwise ranking bound. The bound is neither a
        simultaneous confidence interval nor a gate-fidelity estimate.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    frequencies = np.asarray(frequencies_ghz, dtype=float)
    transfer = np.asarray(transfer, dtype=float)
    if transfer.shape != (2, len(amplitudes), len(frequencies)):
        raise ValueError("transfer must have shape (2, amplitude, frequency)")
    origins = np.array([amplitudes.mean(), frequencies.mean()])
    scales = np.array([np.ptp(amplitudes), np.ptp(frequencies)]) / 2
    if np.any(scales <= 0) or shots_per_direction <= 0:
        raise ValueError("Need nonzero amplitude/frequency spans and positive shots")

    def design(a: ArrayLike, f: ArrayLike) -> NDArray[np.float64]:
        x, y = (
            (np.asarray(a) - origins[0]) / scales[0],
            (np.asarray(f) - origins[1]) / scales[1],
        )
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)

    aa, ff = np.meshgrid(amplitudes, frequencies, indexing="ij")
    matrix = design(aa, ff).reshape(-1, 6)
    values = transfer.mean(axis=0).ravel()
    # Binomial shot variance, with a finite floor at probabilities 0 and 1.
    variance = np.maximum(
        np.sum(transfer * (1 - transfer), axis=0).ravel() / (4 * shots_per_direction),
        1 / (8 * shots_per_direction**2),
    )
    if score_variance is not None:
        score_variance = np.asarray(score_variance, dtype=float)
        if score_variance.shape != transfer.shape:
            raise ValueError("score_variance must match transfer shape")
        variance = np.maximum(np.sum(score_variance, axis=0).ravel() / 4, 1e-12)
    finite = np.isfinite(values) & np.isfinite(variance)
    if finite.sum() < 10 or np.linalg.matrix_rank(matrix[finite]) < 6:
        raise ValueError("Not enough measured points for a 2D quadratic fit")
    weighted = matrix[finite] / np.sqrt(variance[finite, None])
    coef = np.linalg.lstsq(
        weighted, values[finite] / np.sqrt(variance[finite]), rcond=None
    )[0]
    prediction = matrix @ coef
    residual = values - prediction
    reduced_chi2 = float(
        np.sum(residual[finite] ** 2 / variance[finite]) / (finite.sum() - 6)
    )
    covariance = np.linalg.inv(weighted.T @ weighted) * max(1, reduced_chi2)
    dense_a = np.linspace(amplitudes.min(), amplitudes.max(), 101)
    dense_f = np.linspace(frequencies.min(), frequencies.max(), 101)
    da, df = np.meshgrid(dense_a, dense_f, indexing="ij")
    dense_matrix = design(da, df)
    mean = dense_matrix @ coef
    se = np.sqrt(
        np.maximum(
            0, np.einsum("...i,ij,...j->...", dense_matrix, covariance, dense_matrix)
        )
    )
    lower = mean - 1.96 * se
    best = np.unravel_index(np.argmax(lower), lower.shape)
    return {
        "amplitude": float(da[best]),
        "frequency_ghz": float(df[best]),
        "predicted_transfer": float(mean[best]),
        "ranking_score": float(lower[best]),
        "boundary": bool(best[0] in (0, 100) or best[1] in (0, 100)),
        "coefficients": coef.tolist(),
        "origins": origins.tolist(),
        "scales": scales.tolist(),
        "rms_residual": float(np.sqrt(np.mean(residual[finite] ** 2))),
        "reduced_chi2": reduced_chi2,
        "prediction": prediction.reshape(aa.shape),
        "residual": residual.reshape(aa.shape),
        "score_scope": "local pointwise fit bound for ranking, not simultaneous confidence or gate fidelity",
    }

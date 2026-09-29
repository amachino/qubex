"""Internal probability helpers for a single canonical measurement result."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import reduce

import numpy as np
from numpy.typing import NDArray

from .classifiers.state_classifier import StateClassifier
from .models.capture_data import CaptureData
from .models.measurement_result import MeasurementResult


def _get_states(
    capture: CaptureData,
    classifier: StateClassifier | None,
) -> NDArray[np.intp]:
    """Use stored states or classify per-shot IQ with an explicit classifier."""
    states = capture.state_series
    if states is None:
        if capture.config.shot_averaging:
            raise ValueError(
                f"Per-shot IQ or stored states are required for {capture.target}."
            )
        if classifier is None:
            raise ValueError(f"Classifier for target {capture.target} is not set.")
        states = classifier.predict(capture.kerneled)
    states = np.asarray(states)
    if (
        states.shape != (capture.config.n_shots,)
        or states.dtype.kind not in "iuf"
        or not np.all(np.isfinite(states))
        or np.any(states < 0)
        or np.any(states != np.floor(states))
    ):
        raise ValueError(
            f"Expected one nonnegative integer state per shot for {capture.target}."
        )
    if classifier is not None and np.any(states >= classifier.n_states):
        raise ValueError(
            f"State labels exceed classifier dimensions for {capture.target}."
        )
    return states.astype(np.intp)


def get_probabilities(
    result: MeasurementResult,
    targets: Sequence[str] | None = None,
    *,
    classifiers: Mapping[str, StateClassifier] | None = None,
    capture_index: int = -1,
) -> dict[str, float]:
    """
    Return observed joint-state probabilities for selected captures.

    Parameters
    ----------
    result : MeasurementResult
        Single measurement result containing per-shot IQ or stored state labels.
    targets : Sequence[str] | None, optional
        Unique targets in state-label order. Defaults to the result's target order.
    classifiers : Mapping[str, StateClassifier] | None, optional
        Explicit classifiers for captures without `state_series`. Stored states
        take precedence, and classifier references are never loaded automatically.
    capture_index : int, default=-1
        Capture to select for every target. Negative indices are supported.

    Returns
    -------
    dict[str, float]
        Observed state labels such as `"01"` mapped to their relative shot counts.
        Unobserved states are omitted.
    """
    ordered_targets = list(result.data if targets is None else targets)
    if not ordered_targets or len(set(ordered_targets)) != len(ordered_targets):
        raise ValueError("Analysis requires nonempty, unique targets in basis order.")
    classifier_map = {} if classifiers is None else classifiers
    series = []
    for target in ordered_targets:
        captures = result.data.get(target)
        if not captures:
            raise ValueError(f"No measurement capture for target {target}.")
        if not -len(captures) <= capture_index < len(captures):
            raise IndexError(
                f"Capture index {capture_index} is out of range for {target}."
            )
        states = _get_states(captures[capture_index], classifier_map.get(target))
        if series and len(states) != len(series[0]):
            raise ValueError("Shot counts must match across targets.")
        series.append(states)
    counts = Counter(
        "".join(str(state) for state in shot) for shot in zip(*series, strict=True)
    )
    return {label: count / len(series[0]) for label, count in counts.items()}


def get_mitigated_probabilities(
    result: MeasurementResult,
    targets: Sequence[str] | None = None,
    *,
    classifiers: Mapping[str, StateClassifier],
    capture_index: int = -1,
) -> dict[str, float]:
    """
    Correct joint probabilities using explicit classifier confusion matrices.

    Parameters
    ----------
    result : MeasurementResult
        Single measurement result containing per-shot IQ or stored state labels.
    targets : Sequence[str] | None, optional
        Unique targets in state-label and Kronecker-product order.
        Defaults to the result's target order.
    classifiers : Mapping[str, StateClassifier]
        Classifiers for all selected targets, providing state dimensions and
        calibration matrices consistent with the stored or predicted labels.
    capture_index : int, default=-1
        Capture to select for every target. Negative indices are supported.

    Returns
    -------
    dict[str, float]
        All basis labels mapped to corrected probabilities. Values are not clipped
        or renormalized. A singular confusion matrix raises `numpy.linalg.LinAlgError`.

    Notes
    -----
    Confusion matrices are row-normalized as P(measured | prepared). Correction
    solves `corrected @ confusion = observed`. Classifier references are never loaded.
    """
    ordered_targets = list(result.data if targets is None else targets)
    dimensions = []
    matrices = []
    for target in ordered_targets:
        classifier = classifiers.get(target)
        if classifier is None:
            raise ValueError(f"Classifier for target {target} is not set.")
        dimension = classifier.n_states
        matrix = np.asarray(classifier.confusion_matrix, dtype=float)
        if dimension <= 0 or matrix.shape != (dimension, dimension):
            raise ValueError(
                f"Confusion matrix dimensions do not match states for {target}."
            )
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError(
                f"Confusion matrix must be finite and nonnegative for {target}."
            )
        row_totals = matrix.sum(axis=1, keepdims=True)
        if np.any(row_totals == 0):
            raise ValueError(
                f"Confusion matrix has a prepared state with no shots for {target}."
            )
        dimensions.append(dimension)
        matrices.append(matrix / row_totals)
    probabilities = get_probabilities(
        result, ordered_targets, classifiers=classifiers, capture_index=capture_index
    )
    labels = ["".join(map(str, basis)) for basis in np.ndindex(*dimensions)]
    observed = np.array([probabilities.get(label, 0.0) for label in labels])
    corrected = np.linalg.solve(reduce(np.kron, matrices).T, observed)
    return {label: float(value) for label, value in zip(labels, corrected, strict=True)}

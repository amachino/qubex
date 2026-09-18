"""Linear state classifier represented in Qubex I/Q coordinates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .state_classifier import StateClassifier

LinearClassificationLine = tuple[float, float, float]
DecisionPattern = tuple[bool, ...]


def _normalize_line(line: Sequence[float]) -> LinearClassificationLine:
    """Validate and normalize one ``a * I + b * Q + c = 0`` line."""
    if len(line) != 3:
        raise ValueError("Each classification line must contain exactly 3 values.")
    a, b, c = (float(value) for value in line)
    if not all(math.isfinite(value) for value in (a, b, c)):
        raise ValueError("Classification line parameters must be finite.")
    if math.isclose(a, 0.0) and math.isclose(b, 0.0):
        raise ValueError("Classification line coefficients must not both be zero.")
    return (a, b, c)


@dataclass(init=False)
class StateClassifierLinear(StateClassifier):
    """
    Classify normalized Qubex I/Q points with linear decision boundaries.

    Each line uses ``a * I + b * Q + c = 0`` and produces ``True`` on its
    negative side. ``state_map`` maps the resulting boolean tuple to a logical
    state; patterns not present in the map produce the negative rejection state.
    """

    lines: tuple[LinearClassificationLine, ...]
    state_map: dict[DecisionPattern, int]
    rejection_state: int

    def __init__(
        self,
        *,
        lines: Sequence[Sequence[float]],
        state_map: Mapping[DecisionPattern, int],
        rejection_state: int = -1,
        confusion_matrix: NDArray[Any] | None = None,
    ) -> None:
        """Create a classifier from lines and line-decision-to-state mappings."""
        normalized_lines = tuple(_normalize_line(line) for line in lines)
        if not normalized_lines:
            raise ValueError("At least one classification line is required.")

        normalized_state_map: dict[DecisionPattern, int] = {}
        for pattern, state in state_map.items():
            normalized_pattern = tuple(pattern)
            if len(normalized_pattern) != len(normalized_lines):
                raise ValueError(
                    "Each state-map pattern must have one decision per line."
                )
            if not all(
                isinstance(value, (bool, np.bool_)) for value in normalized_pattern
            ):
                raise TypeError("State-map decisions must be boolean values.")
            if isinstance(state, (bool, np.bool_)) or not isinstance(
                state,
                (int, np.integer),
            ):
                raise TypeError("Mapped states must be integer labels.")
            normalized_state_map[tuple(bool(value) for value in normalized_pattern)] = (
                int(state)
            )
        if not normalized_state_map:
            raise ValueError("state_map must contain at least one accepted pattern.")

        states = sorted(set(normalized_state_map.values()))
        if states != list(range(len(states))):
            raise ValueError("Mapped states must be contiguous integers starting at 0.")
        if isinstance(rejection_state, (bool, np.bool_)) or not isinstance(
            rejection_state,
            (int, np.integer),
        ):
            raise TypeError("rejection_state must be an integer label.")
        rejection_state = int(rejection_state)
        if rejection_state >= 0:
            raise ValueError("rejection_state must be negative.")

        if confusion_matrix is None:
            normalized_confusion_matrix = np.eye(len(states), dtype=np.float64)
        else:
            normalized_confusion_matrix = np.asarray(
                confusion_matrix,
                dtype=np.float64,
            )
            if normalized_confusion_matrix.shape != (len(states), len(states)):
                raise ValueError(
                    "confusion_matrix shape must match the number of mapped states."
                )

        self.dataset = {}
        self.model = None
        self.label_map = {
            sum(int(value) << index for index, value in enumerate(pattern)): state
            for pattern, state in normalized_state_map.items()
        }
        self.confusion_matrix = normalized_confusion_matrix
        self.scale = 1.0
        self.phase = 0.0
        self.created_at = datetime.now().isoformat()
        self.lines = normalized_lines
        self.state_map = normalized_state_map
        self.rejection_state = rejection_state

    @property
    def n_states(self) -> int:
        """Return the number of accepted logical states."""
        return len(set(self.state_map.values()))

    @property
    def means(self) -> NDArray:
        """Linear classifiers do not define distribution means."""
        raise NotImplementedError

    @property
    def covariances(self) -> NDArray:
        """Linear classifiers do not define distribution covariances."""
        raise NotImplementedError

    @property
    def centers(self) -> dict[int, complex]:
        """Linear classifiers do not define state centers."""
        raise NotImplementedError

    @property
    def stddevs(self) -> dict[int, float]:
        """Linear classifiers do not define state deviations."""
        raise NotImplementedError

    @property
    def weights(self) -> dict[int, float]:
        """Linear classifiers do not define mixture weights."""
        raise NotImplementedError

    @classmethod
    def fit(
        cls,
        data: dict[int, NDArray],
        phase: float = 0.0,
        n_init: int = 10,
        random_state: int = 42,
    ) -> StateClassifierLinear:
        """Linear boundaries must be supplied explicitly."""
        del data, phase, n_init, random_state
        raise NotImplementedError(
            "StateClassifierLinear cannot be fitted automatically; provide lines "
            "and state_map explicitly."
        )

    def classify_decisions(self, decisions: NDArray[Any]) -> NDArray[np.int64]:
        """Map per-line boolean decisions to logical or rejected states."""
        decision_array = np.asarray(decisions)
        if decision_array.ndim == 1:
            decision_array = decision_array.reshape(1, -1)
        if decision_array.ndim < 2 or decision_array.shape[-1] != len(self.lines):
            raise ValueError("Decision data must have one final-axis value per line.")
        flat_decisions = decision_array.astype(bool, copy=False).reshape(
            -1,
            len(self.lines),
        )
        labels = np.fromiter(
            (
                self.state_map.get(tuple(pattern), self.rejection_state)
                for pattern in flat_decisions
            ),
            dtype=np.int64,
            count=len(flat_decisions),
        )
        return labels.reshape(decision_array.shape[:-1])

    def predict(self, data: NDArray) -> NDArray[np.int64]:
        """Classify complex I/Q samples using the configured linear decisions."""
        samples = np.asarray(data).reshape(-1)
        decisions = np.column_stack(
            [a * samples.real + b * samples.imag + c < 0.0 for a, b, c in self.lines]
        )
        return self.classify_decisions(decisions)

    def classify(
        self,
        target: str,
        data: NDArray,
        plot: bool = True,
    ) -> dict[int, int]:
        """Return counts for accepted states; rejected samples are omitted."""
        del target
        if plot:
            raise NotImplementedError(
                "Plotting is not implemented for StateClassifierLinear."
            )
        labels = self.predict(data)
        accepted = labels[labels >= 0]
        counts = np.bincount(accepted, minlength=self.n_states)
        return {state: int(counts[state]) for state in range(self.n_states)}

    def plot(
        self,
        target: str,
        data: NDArray,
        labels: NDArray,
        n_samples: int = 1000,
    ) -> None:
        """Linear classifier plotting is not part of the core classifier API."""
        del target, data, labels, n_samples
        raise NotImplementedError

    def estimate_weights(
        self,
        data: NDArray,
        max_iter: int = 100,
    ) -> NDArray:
        """Linear classifiers do not estimate mixture weights."""
        del data, max_iter
        raise NotImplementedError

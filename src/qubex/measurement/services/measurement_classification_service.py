"""Classification services for measurement workflows."""

from __future__ import annotations

from collections.abc import Collection
from functools import reduce

import numpy as np
from numpy.typing import NDArray

from qubex.measurement.classifiers.state_classifier import StateClassifier
from qubex.typing import TargetMap


def _normalize_confusion_matrix_rows(matrix: NDArray) -> NDArray[np.float64]:
    """Normalize a confusion matrix as P(measured state | prepared state)."""
    confusion_matrix = np.asarray(matrix, dtype=float)
    row_totals = confusion_matrix.sum(axis=1, keepdims=True)
    if np.any(row_totals == 0):
        raise ValueError("Confusion matrix contains a prepared state with no shots.")
    return confusion_matrix / row_totals


class MeasurementClassificationService:
    """Manage classifiers and confusion-matrix helpers for measurement APIs."""

    def __init__(
        self,
        *,
        classifiers: TargetMap[StateClassifier],
    ) -> None:
        self._classifiers: dict[str, StateClassifier] = dict(classifiers)

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        """Return classifier mapping used for readout post-processing."""
        return self._classifiers

    def update_classifiers(self, classifiers: TargetMap[StateClassifier]) -> None:
        """Update the state classifiers."""
        self._classifiers.update(dict(classifiers))

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray[np.float64]:
        """
        Return the combined confusion matrix for targets.

        Parameters
        ----------
        targets : Collection[str]
            Target labels to include.

        Returns
        -------
        npt.NDArray
            Kronecker-product confusion matrix.
        """
        if len(targets) == 0:
            return np.array([[1.0]], dtype=np.float64)

        target_list = list(targets)
        confusion_matrices: list[NDArray[np.float64]] = []
        for target in target_list:
            cm = self.classifiers[target].confusion_matrix
            confusion_matrices.append(_normalize_confusion_matrix_rows(cm))
        return np.asarray(reduce(np.kron, confusion_matrices), np.float64)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray[np.float64]:
        """
        Return the inverse combined confusion matrix.

        Parameters
        ----------
        targets : Collection[str]
            Target labels to include.

        Returns
        -------
        npt.NDArray
            Inverse confusion matrix.
        """
        confusion_matrix = self.get_confusion_matrix(targets)
        return np.linalg.inv(confusion_matrix)

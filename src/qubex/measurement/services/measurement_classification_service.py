"""Classification services for measurement workflows."""

from __future__ import annotations

from collections.abc import Collection
from functools import reduce

import numpy as np
import numpy.typing as npt

from qubex.measurement.classifiers.state_classifier import StateClassifier
from qubex.typing import TargetMap


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
    ) -> npt.NDArray:
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
        target_list = list(targets)
        confusion_matrices = []
        for target in target_list:
            cm = self.classifiers[target].confusion_matrix
            n_shots = cm[0].sum()
            confusion_matrices.append(cm / n_shots)
        return reduce(np.kron, confusion_matrices)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> npt.NDArray:
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

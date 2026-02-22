"""Tests for measurement classification service."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from qubex.measurement.measurement_classification_service import (
    MeasurementClassificationService,
)


def _classifier_with_matrix(matrix: np.ndarray) -> Any:
    return type("_Classifier", (), {"confusion_matrix": matrix})()


def test_update_classifiers_merges_mapping() -> None:
    """Given classifier updates, when updating, then service stores merged mapping."""
    service = MeasurementClassificationService(
        classifiers={"Q00": cast(Any, _classifier_with_matrix(np.eye(2)))},
    )
    classifier_q01 = cast(Any, _classifier_with_matrix(np.eye(2)))

    service.update_classifiers({"Q01": classifier_q01})

    assert service.classifiers["Q01"] is classifier_q01


def test_get_confusion_matrix_returns_kron_product() -> None:
    """Given per-target matrices, when querying confusion matrix, then normalized Kronecker product is returned."""
    cm_q00 = np.array([[8.0, 2.0], [1.0, 9.0]])
    cm_q01 = np.array([[6.0, 4.0], [3.0, 7.0]])
    service = MeasurementClassificationService(
        classifiers={
            "Q00": cast(Any, _classifier_with_matrix(cm_q00)),
            "Q01": cast(Any, _classifier_with_matrix(cm_q01)),
        },
    )

    result = service.get_confusion_matrix(["Q00", "Q01"])

    expected = np.kron(cm_q00 / cm_q00[0].sum(), cm_q01 / cm_q01[0].sum())
    assert np.allclose(result, expected)


def test_get_inverse_confusion_matrix_returns_inverse() -> None:
    """Given target list, when querying inverse confusion matrix, then matrix inverse is returned."""
    cm_q00 = np.array([[9.0, 1.0], [1.0, 9.0]])
    service = MeasurementClassificationService(
        classifiers={"Q00": cast(Any, _classifier_with_matrix(cm_q00))},
    )

    result = service.get_inverse_confusion_matrix(["Q00"])

    expected = np.linalg.inv(cm_q00 / cm_q00[0].sum())
    assert np.allclose(result, expected)

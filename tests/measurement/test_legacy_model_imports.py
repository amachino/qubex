"""Compatibility tests for legacy measurement model imports."""

from __future__ import annotations

import pickle
import sys
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from qubex.compat.deprecated_imports import reset_deprecated_import_warning
from qubex.measurement.classifiers import (
    StateClassifier,
    StateClassifierGMM,
    StateClassifierKMeans,
)


def _reset_legacy_shim_state(module_name: str) -> None:
    """Reset cached shim import state for deterministic warning checks."""
    sys.modules.pop(module_name, None)
    reset_deprecated_import_warning(legacy_module=module_name)


def _import_symbol(module_name: str, symbol_name: str) -> object:
    """Resolve a symbol from the imported legacy module."""
    module = import_module(module_name)
    return getattr(module, symbol_name)


@pytest.mark.parametrize(
    ("legacy_module_name", "canonical_module_name", "exported_names"),
    [
        (
            "qubex.measurement.measurement_record",
            "qubex.measurement.models.measurement_record",
            ["MeasurementRecord"],
        ),
        (
            "qubex.measurement.measurement_result",
            "qubex.measurement.models.measure_result",
            [
                "MeasureData",
                "MeasureMode",
                "MeasureResult",
                "MultipleMeasureResult",
            ],
        ),
    ],
)
def test_legacy_measurement_model_module_exports_canonical_symbols(
    legacy_module_name: str,
    canonical_module_name: str,
    exported_names: list[str],
) -> None:
    """Legacy measurement model imports should resolve to canonical module symbols."""
    _reset_legacy_shim_state(legacy_module_name)
    canonical_module = import_module(canonical_module_name)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        import_module(legacy_module_name)

    assert not captured

    for exported_name in exported_names:
        _reset_legacy_shim_state(legacy_module_name)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", FutureWarning)
            imported_symbol = _import_symbol(legacy_module_name, exported_name)

        assert imported_symbol is getattr(canonical_module, exported_name)
        assert len(captured) == 1
        assert legacy_module_name in str(captured[0].message)
        assert canonical_module_name in str(captured[0].message)


def test_legacy_measurement_record_module_exports_default_data_dir_alias() -> None:
    """Legacy measurement record import should preserve DEFAULT_DATA_DIR."""
    _reset_legacy_shim_state("qubex.measurement.measurement_record")
    canonical_module = import_module("qubex.measurement.models.measurement_record")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        imported_symbol = _import_symbol(
            "qubex.measurement.measurement_record",
            "DEFAULT_DATA_DIR",
        )

    assert imported_symbol == canonical_module.DEFAULT_RAWDATA_DIR
    assert len(captured) == 1


@pytest.mark.parametrize(
    ("legacy_module_name", "canonical_module_name", "exported_name"),
    [
        (
            "qubex.measurement.state_classifier",
            "qubex.measurement.classifiers.state_classifier",
            "StateClassifier",
        ),
        (
            "qubex.measurement.state_classifier_gmm",
            "qubex.measurement.classifiers.state_classifier_gmm",
            "StateClassifierGMM",
        ),
        (
            "qubex.measurement.state_classifier_kmeans",
            "qubex.measurement.classifiers.state_classifier_kmeans",
            "StateClassifierKMeans",
        ),
    ],
)
def test_legacy_state_classifier_module_exports_canonical_symbol(
    legacy_module_name: str,
    canonical_module_name: str,
    exported_name: str,
) -> None:
    """Legacy state classifier imports should resolve to canonical symbols."""
    legacy_module = import_module(legacy_module_name)
    canonical_module = import_module(canonical_module_name)

    assert getattr(legacy_module, exported_name) is getattr(
        canonical_module,
        exported_name,
    )


@pytest.mark.parametrize(
    ("classifier", "legacy_module_name", "canonical_module_name"),
    [
        (
            StateClassifierGMM(
                dataset={0: np.array([[0.0, 0.0]])},
                model=cast(Any, None),
                label_map={0: 0},
                confusion_matrix=np.eye(1),
                scale=1.0,
                phase=0.0,
                created_at="2026-03-13T00:00:00",
            ),
            "qubex.measurement.state_classifier_gmm",
            "qubex.measurement.classifiers.state_classifier_gmm",
        ),
        (
            StateClassifierKMeans(
                dataset={0: np.array([[0.0, 0.0]])},
                model=cast(Any, None),
                label_map={0: 0},
                confusion_matrix=np.eye(1),
                scale=1.0,
                phase=0.0,
                created_at="2026-03-13T00:00:00",
            ),
            "qubex.measurement.state_classifier_kmeans",
            "qubex.measurement.classifiers.state_classifier_kmeans",
        ),
    ],
)
def test_state_classifier_load_supports_legacy_pickle_module_paths(
    tmp_path: Path,
    classifier: StateClassifier,
    legacy_module_name: str,
    canonical_module_name: str,
) -> None:
    """StateClassifier.load should restore classifiers pickled with legacy modules."""
    classifier_path = tmp_path / "classifier.pkl"
    payload = pickle.dumps(classifier, protocol=0)
    classifier_path.write_bytes(
        payload.replace(
            canonical_module_name.encode(),
            legacy_module_name.encode(),
        )
    )

    restored = StateClassifier.load(classifier_path)

    assert type(restored) is type(classifier)
    assert restored.label_map == classifier.label_map
    assert restored.scale == classifier.scale
    assert restored.phase == classifier.phase
    assert restored.created_at == classifier.created_at
    np.testing.assert_array_equal(
        restored.confusion_matrix,
        classifier.confusion_matrix,
    )
    np.testing.assert_array_equal(restored.dataset[0], classifier.dataset[0])

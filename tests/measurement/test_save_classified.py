from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from qubex.measurement.measurement_result import MeasureResult


def test_save_classified_json(measure_result: MeasureResult, tmp_path: Path):
    out = tmp_path / "classified.json.gz"
    path = measure_result.save_classified(out, format="json", compress=True)
    assert path.exists()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        obj = json.load(f)
    assert "counts" in obj and "probabilities" in obj
    assert obj["metadata"]["n_shots_kept"] <= obj["metadata"]["n_shots_raw"]


def test_save_classified_no_memory(measure_result: MeasureResult, tmp_path: Path):
    out = tmp_path / "classified.json"
    path = measure_result.save_classified(
        out, format="json", include_memory=False, compress=False
    )
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    assert "memory" not in obj


def test_save_classified_npz(measure_result: MeasureResult, tmp_path: Path):
    out = tmp_path / "classified"
    path = measure_result.save_classified(out, format="npz")
    data = np.load(path, allow_pickle=True)
    mem = data["memory"]
    assert mem.shape[1] == 1  # one qubit
    meta = json.loads(str(data["metadata"]))
    assert meta["n_qubits"] == 1


def test_save_classified_threshold(measure_result: MeasureResult, tmp_path: Path):
    out = tmp_path / "threshold.json.gz"
    path = measure_result.save_classified(out, format="json", threshold=0.9)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        obj = json.load(f)
    assert obj["metadata"]["threshold"] == 0.9
    # Shots kept could be fewer than raw
    assert obj["metadata"]["n_shots_kept"] <= obj["metadata"]["n_shots_raw"]


def test_save_classified_overwrite_false(measure_result: MeasureResult, tmp_path: Path):
    out = tmp_path / "dup.json.gz"
    measure_result.save_classified(out, format="json")
    with pytest.raises(ValueError):
        measure_result.save_classified(out, format="json", overwrite=False)

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from qubex.measurement.measurement_result import MeasureData, MeasureMode, MeasureResult
from qubex.measurement.state_classifier import StateClassifier


class DummyClassifier(StateClassifier):
    """
    Minimal two-state classifier for tests.

    Predicts label 1 when real(IQ) > 0.5 else 0.
    Probabilities via a logistic centered at 0.5.
    """

    @property
    def n_states(self) -> int:  # two states
        return 2

    @property
    def means(self) -> NDArray:
        return np.array([0 + 0j, 1 + 0j])

    @property
    def covariances(self) -> NDArray:
        return np.eye(2)

    @property
    def centers(self) -> dict[int, complex]:
        return {0: 0 + 0j, 1: 1 + 0j}

    @property
    def stddevs(self) -> dict[int, float]:
        return {0: 1.0, 1: 1.0}

    @property
    def weights(self) -> dict[int, float]:
        return {0: 0.5, 1: 0.5}

    @classmethod
    def fit(
        cls,
        data: dict[int, NDArray],
        n_init: int = 10,
        random_state: int = 42,
    ) -> "DummyClassifier":
        raise NotImplementedError

    def predict(self, data: NDArray) -> NDArray:
        return (np.real(data) > 0.5).astype(int)

    def predict_proba(self, data: NDArray) -> NDArray:  # soft logistic
        x = np.real(data)
        p1 = 1 / (1 + np.exp(-(x - 0.5)))
        return np.column_stack([1 - p1, p1])

    def classify(self, target: str, data: NDArray, plot: bool = True) -> dict[int, int]:
        labels = self.predict(data)
        return {0: int((labels == 0).sum()), 1: int((labels == 1).sum())}

    def plot(
        self,
        target: str,
        data: NDArray,
        labels: NDArray,
        n_samples: int = 1000,
    ) -> None:
        pass

    def estimate_weights(self, data: NDArray, max_iter: int = 100) -> NDArray:
        return np.array([0.5, 0.5])


@pytest.fixture
def dummy_classifier() -> DummyClassifier:
    # Deterministic small dataset around 0 and 1 for reproducibility
    rng = np.random.default_rng(1234)
    shots = 32
    iq0 = 0.2 + 0.05 * (rng.standard_normal(shots) + 1j * rng.standard_normal(shots))
    iq1 = 0.8 + 0.05 * (rng.standard_normal(shots) + 1j * rng.standard_normal(shots))
    return DummyClassifier(
        dataset={0: iq0, 1: iq1},
        model=None,
        label_map={0: 0, 1: 1},
        confusion_matrix=np.array([[15, 1], [1, 15]]),
        scale=1.0,
        phase=0.0,
        created_at="",
    )


@pytest.fixture
def measure_result(dummy_classifier: DummyClassifier) -> MeasureResult:
    rng = np.random.default_rng(5678)
    shots = 32
    iq0 = dummy_classifier.dataset[0]
    iq1 = dummy_classifier.dataset[1]
    mix = np.where(rng.random(shots) > 0.5, iq1, iq0)
    raw = mix[:, None]  # shape (shots, 1)
    data = {
        "Q00": MeasureData(
            target="Q00",
            mode=MeasureMode.SINGLE,
            raw=raw,
            classifier=dummy_classifier,
        )
    }
    return MeasureResult(mode=MeasureMode.SINGLE, data=data, config={"dummy": 1})

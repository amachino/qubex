from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class StateClassifier(ABC):
    """
    A protocol for state classifiers.
    """

    dataset: dict[int, NDArray[np.float32]]
    model: Any
    label_map: dict[int, int]
    confusion_matrix: NDArray
    scale: float

    @property
    @abstractmethod
    def n_states(self) -> int:
        """The number of states in the model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def centers(self) -> dict[int, complex]:
        """The center of each state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def stddevs(self) -> dict[int, float]:
        """The standard deviation of each state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def weights(self) -> dict[int, float]:
        """The weights of each state."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def fit(
        cls,
        data: dict[int, NDArray[np.complex64]],
        n_init: int = 10,
        random_state: int = 42,
    ) -> StateClassifier:
        """
        Fit a k-means model to the provided data.

        Parameters
        ----------
        data : dict[int, NDArray[np.complex64]]
            A dictionary of state labels and complex data.
        n_init : int, optional
            Number of time the k-means algorithm will be run with different center seeds, by default 10.
        random_state : int, optional
            The random state for the model, by default 42.

        Returns
        -------
        StateClassifier
            A state classifier model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        data: NDArray[np.complex128],
    ) -> NDArray:
        """
        Predict the state labels for the provided data.

        Parameters
        ----------
        data : NDArray[np.complex128]
            An array of complex numbers representing the data to classify.

        Returns
        -------
        NDArray
            An array of predicted state labels based on the fitted model.
        """
        raise NotImplementedError

    @abstractmethod
    def classify(
        self,
        target: str,
        data: NDArray[np.complex128],
        plot: bool = True,
    ) -> dict[int, int]:
        """
        Classify the provided data and return the state counts.

        Parameters
        ----------
        data : NDArray[np.complex128]
            An array of complex numbers representing the data to classify.
        plot : bool, optional
            A flag to plot the data and predicted labels, by default True.

        Returns
        -------
        dict[int, int]
            A dictionary of state labels and their counts.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(
        self,
        target: str,
        data: NDArray[np.complex128],
        labels: NDArray,
    ):
        """
        Plot the data and the predicted labels.

        Parameters
        ----------
        data : NDArray[np.complex128]
            An array of complex numbers representing the data.
        labels : NDArray
            An array of predicted state labels.
        """
        raise NotImplementedError

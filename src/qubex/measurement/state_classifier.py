from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
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
    created_at: str

    def save(self, path: Path | str):
        """
        Save the state classifier model to a file.

        Parameters
        ----------
        path : Path | str
            The path to save the model.
        """
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path | str) -> StateClassifier:
        """
        Load a state classifier model from a file.

        Parameters
        ----------
        path : Path | str
            The path to load the model.

        Returns
        -------
        StateClassifier
            A state classifier model.
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    @property
    @abstractmethod
    def n_states(self) -> int:
        """The number of states in the model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def means(self) -> NDArray:
        """The means of the model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def covariances(self) -> NDArray:
        """The covariances of the model."""
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
        n_samples: int = 1000,
    ):
        """
        Plot the data and the predicted labels.

        Parameters
        ----------
        data : NDArray[np.complex128]
            An array of complex numbers representing the data.
        labels : NDArray
            An array of predicted state labels.
        n_samples : int, optional
            The number of samples to plot, by default 1000.
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_weights(
        self,
        data: NDArray,
        max_iter: int = 100,
    ) -> NDArray:
        """
        Parameters
        ----------
        data : NDArray
            The mixed gaussian data to estimate the weights.
        max_iter : int, optional
            The maximum number of iterations, by default 100.

        Returns
        -------
        NDArray
            The estimated weights of the mixed gaussian data.
        """
        raise NotImplementedError

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture


@dataclass
class StateClassifier:
    """
    A state classifier model that uses a Gaussian Mixture Model to classify data.

    Attributes
    ----------
    dataset : dict[int, NDArray[np.float32]]
        A dictionary of state labels and preprocessed data.
    model : GaussianMixture
        The fitted Gaussian Mixture Model.
    label_map : dict[int, int]
        A mapping from state labels to GMM component labels.
    mean : dict[int, complex]
        The mean of each state.
    stddev : dict[int, float]
        The standard deviation of each state.
    weight : dict[int, float]
        The weight of each state.
    """

    dataset: dict[int, NDArray[np.float32]]
    model: GaussianMixture
    label_map: dict[int, int]
    conf_matrix: NDArray
    means: dict[int, complex]
    covariances: dict[int, float]
    weights: dict[int, float]

    @classmethod
    def fit(
        cls,
        data: dict[int, NDArray[np.complex64]],
        covariance_type: Literal["full", "spherical", "tied", "diag"] = "full",
        random_state: int = 42,
    ) -> StateClassifier:
        """
        Fit a Gaussian Mixture Model to the provided data.

        Parameters
        ----------
        data : dict[int, NDArray[np.complex128]]
            A dictionary of state labels and complex data.
        covariance_type : Literal["full", "spherical", "tied", "diag"], optional
            The type of covariance matrix to use, by default "full".
        random_state : int, optional
            The random state for the model, by default 42.

        Returns
        -------
        StateClassifier or None
            A state classifier model, or None if the fitting fails.
        """
        # Validate input data
        if not isinstance(data, dict) or not all(
            isinstance(k, int) and isinstance(v, np.ndarray) for k, v in data.items()
        ):
            raise ValueError(
                "Input data must be a dictionary with integer keys and numpy array values."
            )

        # Convert complex data to real-valued features
        dataset = {
            state: np.column_stack([np.real(data[state]), np.imag(data[state])])
            for state in data
        }

        # Fit Gaussian Mixture Model
        model = GaussianMixture(
            n_components=len(dataset),
            covariance_type=covariance_type,
            random_state=random_state,
        )
        concat_data = np.concatenate(list(dataset.values()))
        model.fit(concat_data)

        # Create label map
        label_map = cls._create_label_map(model, dataset)

        # Create confusion matrix
        conf_matrix = cls._create_confusion_matrix(model, dataset, label_map)

        # Extract model parameters
        means, covariances, weights = cls._extract_model_parameters(
            model=model,
            dataset=dataset,
            label_map=label_map,
        )

        # Return state classifier model
        return cls(
            dataset=dataset,
            model=model,
            label_map=label_map,
            conf_matrix=conf_matrix,
            means=means,
            covariances=covariances,
            weights=weights,
        )

    @staticmethod
    def _create_label_map(
        model: GaussianMixture,
        dataset: dict[int, NDArray[np.float32]],
    ) -> dict[int, int]:
        """
        Create a mapping from state labels to GMM component labels.

        Parameters
        ----------
        model : GaussianMixture
            The fitted Gaussian Mixture Model.
        dataset : dict[int, NDArray[np.float32]]
            The preprocessed dataset.

        Returns
        -------
        dict[int, int]
            A mapping from state labels to GMM component labels.
        """
        label_map = {}
        for state, data in dataset.items():
            result = model.predict(data)
            count = np.bincount(result)
            label_map[state] = np.argmax(count).astype(int)
        return label_map

    @staticmethod
    def _create_confusion_matrix(
        model: GaussianMixture,
        dataset: dict[int, NDArray[np.float32]],
        label_map: dict[int, int],
    ) -> NDArray:
        """
        Create a confusion matrix based on the true and predicted labels.

        Parameters
        ----------
        model : GaussianMixture
            The fitted Gaussian Mixture Model.
        dataset : dict[int, NDArray[np.float32]]
            The preprocessed dataset.
        label_map : dict[int, int]
            A mapping from state labels to GMM component labels.

        Returns
        -------
        NDArray
            A confusion matrix of true and predicted labels.
        """
        true_labels = np.concatenate(
            [np.full(len(dataset[state]), state) for state in dataset]
        )
        concat_data = np.concatenate(list(dataset.values()))
        predicted_labels_ = model.predict(concat_data)
        predicted_labels = np.array([label_map[label] for label in predicted_labels_])
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        return conf_matrix

    @staticmethod
    def _extract_model_parameters(
        model: GaussianMixture,
        dataset: dict[int, NDArray[np.float32]],
        label_map: dict[int, int],
    ) -> tuple[dict[int, complex], dict[int, float], dict[int, float]]:
        """
        Extract the mean, standard deviation, and weight of each state.

        Parameters
        ----------
        model : GaussianMixture
            The fitted Gaussian Mixture Model.
        dataset : dict[int, NDArray[np.float32]]
            The preprocessed dataset.
        label_map : dict[int, int]
            A mapping from state labels to GMM component labels.

        Returns
        -------
        tuple[dict[int, complex], dict[int, float], dict[int, float]]
            The means, covariances, and weights of each state.
        """
        means, covariances, weights = {}, {}, {}
        for state in dataset:
            means_arr = np.array(model.means_)[label_map[state]]
            means[state] = means_arr[0] + 1j * means_arr[1]
            covariances[state] = np.array(model.covariances_)[label_map[state]]
            weights[state] = np.array(model.weights_)[label_map[state]]

        return means, covariances, weights

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
        # Convert complex data to real-valued features
        real_imag_data = np.column_stack([np.real(data), np.imag(data)])
        # Predict GMM component labels
        gmm_labels = self.model.predict(real_imag_data)
        # Convert GMM component labels to state labels
        state_labels = np.array([self.label_map[label] for label in gmm_labels])

        return state_labels

    def predict_proba(
        self,
        data: NDArray[np.complex128],
    ) -> NDArray:
        """
        Predict the probability of each sample belonging to each state.

        Parameters
        ----------
        data : NDArray[np.complex128]
            An array of complex numbers representing the data to classify.

        Returns
        -------
        NDArray
            An array of probabilities for each sample, where each row represents
            the probabilities of that sample belonging to each state.
        """
        # Convert complex data to real-valued features
        real_imag_data = np.column_stack([np.real(data), np.imag(data)])
        # Predict probabilities of GMM component labels
        gmm_proba = self.model.predict_proba(real_imag_data)

        # Map GMM component probabilities to state probabilities
        state_proba = np.zeros((gmm_proba.shape[0], len(self.label_map)))
        for component, state in self.label_map.items():
            state_proba[:, state] += gmm_proba[:, component]

        return state_proba

    def classify(
        self,
        value: complex,
    ) -> tuple[int, float]:
        """
        Classify a single value based on the fitted model.

        Parameters
        ----------
        value : complex
            A complex number to classify.

        Returns
        -------
        tuple[int, float]
            The predicted state label and the probability of the prediction.
        """
        # Predict the state label
        label = self.predict(np.array([value]))[0]
        # Predict the probability of the state label
        proba = self.predict_proba(np.array([value]))[0, label]

        return label, proba

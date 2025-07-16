from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

from ..style import get_colors, get_config
from .state_classifier import StateClassifier


@dataclass
class StateClassifierGMM(StateClassifier):
    """
    A state classifier model that uses Gaussian Mixture Model (GMM) to classify data.

    Attributes
    ----------
    dataset : dict[int, NDArray[np.float32]]
        A dictionary of state labels and preprocessed data.
    model : GaussianMixture
        The fitted GMM model.
    label_map : dict[int, int]
        A mapping from GMM component labels to state labels.
    confusion_matrix : NDArray
        The confusion matrix of the classifier.
    centers : dict[int, complex]
        The center of each state.
    """

    dataset: dict[int, NDArray]
    model: GaussianMixture
    label_map: dict[int, int]
    confusion_matrix: NDArray
    scale: float
    phase: float
    created_at: str

    @property
    def n_states(self) -> int:
        """The number of states in the model."""
        return len(self.dataset)

    @property
    def means(self) -> NDArray:
        """The means of the model."""
        means_ = np.asarray(self.model.means_)
        means = np.zeros_like(means_)
        for idx, mean in enumerate(means_):
            state = self.label_map[idx]
            means[state] = mean
        return means

    @property
    def covariances(self) -> NDArray:
        """The covariances of the model."""
        covariances_ = np.asarray(self.model.covariances_)
        covariances = np.zeros_like(covariances_)
        for idx, covariance in enumerate(covariances_):
            state = self.label_map[idx]
            covariances[state] = covariance
        return covariances

    @property
    def centers(self) -> dict[int, complex]:
        """The center of each state."""
        return {
            state: complex(mean[0], mean[1]) * np.exp(1j * self.phase) / self.scale
            for state, mean in enumerate(self.means)
        }

    @property
    def stddevs(self) -> dict[int, float]:
        """The standard deviation of each state."""
        return {
            state: np.sqrt(covariance) / self.scale
            for state, covariance in enumerate(self.covariances)
        }

    @property
    def weights(self) -> dict[int, float]:
        """The weights of each state."""
        weights_arr = np.asarray(self.model.weights_)
        return {label: weight for label, weight in enumerate(weights_arr)}

    @classmethod
    def fit(
        cls,
        data: dict[int, NDArray],
        phase: float = 0.0,
        n_init: int = 10,
        random_state: int = 42,
    ) -> StateClassifierGMM:
        """
        Fit a Gaussian Mixture Model (GMM) to the provided data.

        Parameters
        ----------
        data : dict[int, NDArray]
            A dictionary of state labels and complex data.
        phase : float, optional
            The phase offset to apply to the data, by default 0.0.
        n_init : int, optional
            Number of initializations to perform, by default 10.
        random_state : int, optional
            The random state for the model, by default 42.

        Returns
        -------
        StateClassifierGMM
            A state classifier model.
        """
        # Validate input data
        if not isinstance(data, dict) or not all(
            isinstance(k, int) and isinstance(v, np.ndarray) for k, v in data.items()
        ):
            raise ValueError(
                "Input data must be a dictionary with integer keys and numpy array values."
            )

        # Adjust data phase
        data = {state: np.exp(-1j * phase) * data[state] for state in data}

        # Convert complex data to real-valued features
        dataset = {
            state: np.column_stack((np.real(data[state]), np.imag(data[state])))
            for state in data
        }

        # Number of components
        n_components = len(dataset)

        # Concatenate data
        concat_data = np.concatenate(list(dataset.values()))

        # Scale data
        range_x = np.max(concat_data[:, 0]) - np.min(concat_data[:, 0])
        range_y = np.max(concat_data[:, 1]) - np.min(concat_data[:, 1])
        scale = 1 / max(range_x, range_y).astype(float)

        # Scale dataset
        scaled_dataset = {state: scale * data for state, data in dataset.items()}

        # Scale concatenated data
        scaled_concat_data = np.concatenate(list(scaled_dataset.values()))

        # Fit GMM model
        model = GaussianMixture(
            n_components=n_components,
            n_init=n_init,
            random_state=random_state,
            covariance_type="spherical",
        )
        model.fit(scaled_concat_data)

        # Create label map
        label_map = cls._create_label_map(model, scaled_dataset)

        # Create confusion matrix
        confusion_matrix = cls._create_confusion_matrix(
            model,
            scaled_dataset,
            label_map,
        )

        # Return state classifier model
        return cls(
            dataset=dataset,
            model=model,
            label_map=label_map,
            confusion_matrix=confusion_matrix,
            scale=scale,
            phase=phase,
            created_at=datetime.now().isoformat(),
        )

    @staticmethod
    def _create_label_map(
        model: GaussianMixture,
        dataset: dict[int, NDArray],
    ) -> dict[int, int]:
        """
        Create a mapping from GMM component labels to state labels.

        Parameters
        ----------
        model : GaussianMixture
            The fitted GMM model.
        dataset : dict[int, NDArray]
            The preprocessed dataset.

        Returns
        -------
        dict[int, int]
            A mapping from GMM component labels to state labels.
        """
        n_components = len(dataset)
        label_map = {label: -1 for label in range(n_components)}
        for state, data in dataset.items():
            result = model.predict(data)
            count = np.bincount(result, minlength=n_components)
            for label in np.argsort(count)[::-1]:
                if label_map[label] == -1:
                    label_map[label] = state
                    break
        return label_map

    @staticmethod
    def _create_confusion_matrix(
        model: GaussianMixture,
        dataset: dict[int, NDArray],
        label_map: dict[int, int],
    ) -> NDArray:
        """
        Create a confusion matrix based on the true and predicted labels.

        Parameters
        ----------
        model : GaussianMixture
            The fitted GMM model.
        dataset : dict[int, NDArray]
            The preprocessed dataset.
        label_map : dict[int, int]
            A mapping from GMM component labels to state labels.

        Returns
        -------
        NDArray
            A confusion matrix of true and predicted labels.
        """
        true_labels = np.concatenate(
            [np.full(len(data), state) for state, data in dataset.items()]
        )
        concat_data = np.concatenate(list(dataset.values()))
        predicted_labels_ = model.predict(concat_data)
        predicted_labels = np.array([label_map[label] for label in predicted_labels_])
        return confusion_matrix(true_labels, predicted_labels)

    def predict(
        self,
        data: NDArray,
    ) -> NDArray:
        """
        Predict the state labels for the provided data.

        Parameters
        ----------
        data : NDArray
            An array of complex numbers representing the data to classify.

        Returns
        -------
        NDArray
            An array of predicted state labels based on the fitted model.
        """
        # Normalize data
        norm_data = data * self.scale * np.exp(-1j * self.phase)
        # Convert complex data to real-valued features
        real_imag_data = np.column_stack([np.real(norm_data), np.imag(norm_data)])
        # Predict GMM component labels
        component_labels = self.model.predict(real_imag_data)
        # Convert GMM component labels to state labels
        state_labels = np.array([self.label_map[label] for label in component_labels])

        return state_labels

    def predict_proba(
        self,
        data: NDArray,
    ) -> NDArray:
        """
        Predict the state probabilities for the provided data.

        Parameters
        ----------
        data : NDArray
            An array of complex numbers representing the data to classify.

        Returns
        -------
        NDArray
            An array of predicted state probabilities based on the fitted model.
        """
        # Normalize data
        norm_data = data * self.scale * np.exp(-1j * self.phase)
        # Convert complex data to real-valued features
        real_imag_data = np.column_stack([np.real(norm_data), np.imag(norm_data)])
        # Predict GMM component probabilities
        component_label_proba = self.model.predict_proba(real_imag_data)
        # Convert GMM component probabilities to state probabilities
        state_proba = np.zeros((len(data), self.n_states))
        for label in range(self.n_states):
            state_proba[:, label] = component_label_proba[:, self.label_map[label]]

        return state_proba

    def classify(
        self,
        target: str,
        data: NDArray,
        plot: bool = True,
    ) -> dict[int, int]:
        """
        Classify the provided data and return the state counts.

        Parameters
        ----------
        data : NDArray
            An array of complex numbers representing the data to classify.
        plot : bool, optional
            A flag to plot the data and predicted labels, by default True.

        Returns
        -------
        dict[int, int]
            A dictionary of state labels and their counts.
        """
        predicted_labels = self.predict(data)
        if plot:
            self.plot(target, data, predicted_labels)
        count = np.bincount(predicted_labels, minlength=self.n_states)
        state = {label: count[label] for label in range(len(count))}
        return state

    def plot(
        self,
        target: str,
        data: NDArray,
        labels: NDArray,
        n_samples: int = 1000,
    ):
        """
        Plot the data and the predicted labels.

        Parameters
        ----------
        data : NDArray[np.complexfloating[Any, Any]],
            An array of complex numbers representing the data.
        labels : NDArray
            An array of predicted state labels.
        n_samples : int, optional
            The number of samples to plot, by default 1000.
        """
        if len(data) > n_samples:
            data = data[:n_samples]
            labels = labels[:n_samples]
        x = data.real
        y = data.imag
        unique_labels = np.unique(labels)
        colors = get_colors(alpha=0.8)

        max_val = np.max(np.abs(data))
        axis_range = [-max_val * 1.1, max_val * 1.1]
        dtick = max_val / 2

        fig = go.Figure()
        for idx, label in enumerate(unique_labels):
            color = colors[idx % len(colors)]
            mask = labels == label
            fig.add_trace(
                go.Scatter(
                    x=x[mask],
                    y=y[mask],
                    mode="markers",
                    name=f"|{label}⟩",
                    marker=dict(
                        size=4,
                        color=f"rgba{color}",
                    ),
                )
            )
        for label, center in self.centers.items():
            fig.add_trace(
                go.Scatter(
                    x=[center.real],
                    y=[center.imag],
                    mode="markers",
                    name=f"|{label}⟩",
                    showlegend=True,
                    marker=dict(
                        size=10,
                        color="black",
                        symbol="x",
                    ),
                )
            )
        for label, center in self.centers.items():
            sigma = self.stddevs[label]
            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = center.real + 2 * sigma * np.cos(theta)
            y_circle = center.imag + 2 * sigma * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    mode="lines",
                    name=f"|{label}⟩ ± 2σ",
                    showlegend=True,
                    line=dict(
                        color="black",
                        width=2,
                        dash="dot",
                    ),
                )
            )

        fig.update_layout(
            title=f"State classification : {target}",
            xaxis_title="In-Phase (arb. units)",
            yaxis_title="Quadrature (arb. units)",
            showlegend=True,
            width=500,
            height=400,
            margin=dict(l=120, r=120),
            xaxis=dict(
                range=axis_range,
                dtick=dtick,
                tickformat=".2g",
                showticklabels=True,
                zeroline=True,
                zerolinecolor="black",
                showgrid=True,
            ),
            yaxis=dict(
                range=axis_range,
                scaleanchor="x",
                scaleratio=1,
                dtick=dtick,
                tickformat=".2g",
                showticklabels=True,
                zeroline=True,
                zerolinecolor="black",
                showgrid=True,
            ),
        )
        fig.show(config=get_config())

    def estimate_weights(
        self,
        data: NDArray,
        max_iter: int = 100,
        tol: float = 1e-4,
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
        N = self.n_states
        norm_data = np.column_stack([np.real(data), np.imag(data)]) * self.scale
        weights = np.ones(N) / N

        # Expectation-Maximization (EM) algorithm
        for _ in range(max_iter):
            responsibilities = np.zeros((norm_data.shape[0], N))
            for k in range(N):
                responsibilities[:, k] = weights[k] * multivariate_normal.pdf(
                    norm_data,
                    mean=self.means[k],
                    cov=self.covariances[k],
                )
            resp_sum = responsibilities.sum(axis=1, keepdims=True)
            resp_sum[resp_sum == 0] = 1e-12  # Avoid division by zero
            responsibilities /= resp_sum

            new_weights = responsibilities.mean(axis=0)
            if np.allclose(weights, new_weights, atol=tol):
                break
            weights = new_weights

        return weights

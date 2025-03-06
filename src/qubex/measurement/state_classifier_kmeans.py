from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from ..style import get_colors, get_config
from .state_classifier import StateClassifier


@dataclass
class StateClassifierKMeans(StateClassifier):
    """
    A state classifier model that uses k-means to classify data.

    Attributes
    ----------
    dataset : dict[int, NDArray[np.float32]]
        A dictionary of state labels and preprocessed data.
    model : KMeans
        The fitted k-means model.
    label_map : dict[int, int]
        A mapping from k-means cluster labels to state labels.
    confusion_matrix : NDArray
        The confusion matrix of the classifier.
    centers : dict[int, complex]
        The center of each state.
    """

    dataset: dict[int, NDArray[np.float32]]
    model: KMeans
    label_map: dict[int, int]
    confusion_matrix: NDArray

    @property
    def n_states(self) -> int:
        """The number of states in the model."""
        return len(self.dataset)

    @property
    def centers(self) -> dict[int, complex]:
        """The center of each state."""
        centers = {}
        centers_arr = np.asarray(self.model.cluster_centers_)
        for idx, center in enumerate(centers_arr):
            state = self.label_map[idx]
            centers[state] = complex(center[0], center[1])
        centers = dict(sorted(centers.items()))
        return centers

    @property
    def means(self) -> NDArray:
        """The means of the model."""
        raise NotImplementedError

    @property
    def covariances(self) -> NDArray:
        """The covariances of the model."""
        raise NotImplementedError

    @property
    def stddevs(self) -> dict[int, float]:
        """The standard deviation of each state."""
        raise NotImplementedError

    @property
    def weights(self) -> dict[int, float]:
        """The weights of each state."""
        raise NotImplementedError

    @classmethod
    def fit(
        cls,
        data: dict[int, NDArray[np.complex64]],
        n_init: int = 10,
        random_state: int = 42,
    ) -> StateClassifierKMeans:
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
        # Validate input data
        if not isinstance(data, dict) or not all(
            isinstance(k, int) and isinstance(v, np.ndarray) for k, v in data.items()
        ):
            raise ValueError(
                "Input data must be a dictionary with integer keys and numpy array values."
            )

        # Convert complex data to real-valued features
        dataset = {
            state: np.column_stack((np.real(data[state]), np.imag(data[state])))
            for state in data
        }

        # Fit k-means model
        concat_data = np.concatenate(list(dataset.values()))
        n_clusters = len(dataset)
        model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        )
        model.fit(concat_data)

        # Create label map
        label_map = cls._create_label_map(model, dataset)

        # Create confusion matrix
        confusion_matrix = cls._create_confusion_matrix(model, dataset, label_map)

        # Return state classifier model
        return cls(
            dataset=dataset,
            model=model,
            label_map=label_map,
            confusion_matrix=confusion_matrix,
            scale=1.0,
        )

    @staticmethod
    def _create_label_map(
        model: KMeans,
        dataset: dict[int, NDArray[np.float32]],
    ) -> dict[int, int]:
        """
        Create a mapping from k-means cluster labels to state labels.

        Parameters
        ----------
        model : KMeans
            The fitted k-means model.
        dataset : dict[int, NDArray[np.float32]]
            The preprocessed dataset.

        Returns
        -------
        dict[int, int]
            A mapping from k-means cluster labels to state labels.
        """
        n_clusters = len(dataset)
        label_map = {label: -1 for label in range(n_clusters)}
        for state, data in dataset.items():
            result = model.predict(data)
            count = np.bincount(result, minlength=n_clusters)
            for label in np.argsort(count)[::-1]:
                if label_map[label] == -1:
                    label_map[label] = state
                    break
        return label_map

    @staticmethod
    def _create_confusion_matrix(
        model: KMeans,
        dataset: dict[int, NDArray[np.float32]],
        label_map: dict[int, int],
    ) -> NDArray:
        """
        Create a confusion matrix based on the true and predicted labels.

        Parameters
        ----------
        model : KMeans
            The fitted k-means model.
        dataset : dict[int, NDArray[np.float32]]
            The preprocessed dataset.
        label_map : dict[int, int]
            A mapping from k-means cluster labels to state labels.

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
        # Predict k-means cluster labels
        cluster_labels = self.model.predict(real_imag_data)
        # Convert k-means cluster labels to state labels
        state_labels = np.array([self.label_map[label] for label in cluster_labels])

        return state_labels

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
        predicted_labels = self.predict(data)
        if plot:
            self.plot(target, data, predicted_labels)
        count = np.bincount(predicted_labels, minlength=self.n_states)
        state = {label: count[label] for label in range(len(count))}
        return state

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
        fig.update_layout(
            title=f"State classification : {target}",
            xlabel="In-Phase (arb. unit)",
            ylabel="Quadrature (arb. unit)",
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

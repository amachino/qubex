from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

from ..style import get_colors, get_config


@dataclass
class StateClassifierGMM:
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

    dataset: dict[int, NDArray[np.float32]]
    model: GaussianMixture
    label_map: dict[int, int]
    confusion_matrix: NDArray
    centers: dict[int, complex]

    @property
    def n_clusters(self) -> int:
        """The number of clusters in the model."""
        return len(self.dataset)

    @classmethod
    def fit(
        cls,
        data: dict[int, NDArray[np.complex64]],
        n_init: int = 10,
        random_state: int = 42,
    ) -> StateClassifierGMM:
        """
        Fit a Gaussian Mixture Model (GMM) to the provided data.

        Parameters
        ----------
        data : dict[int, NDArray[np.complex64]]
            A dictionary of state labels and complex data.
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

        # Convert complex data to real-valued features
        dataset = {
            state: np.column_stack((np.real(data[state]), np.imag(data[state])))
            for state in data
        }

        # Fit GMM model
        concat_data = np.concatenate(list(dataset.values()))
        n_clusters = len(dataset)
        model = GaussianMixture(
            n_components=n_clusters,
            n_init=n_init,
            random_state=random_state,
        )
        model.fit(concat_data)

        # Create label map
        label_map = cls._create_label_map(model, dataset)

        # Create confusion matrix
        confusion_matrix = cls._create_confusion_matrix(model, dataset, label_map)

        # Extract model parameters (means of each component)
        centers = cls._extract_model_parameters(model)

        # Return state classifier model
        return cls(
            dataset=dataset,
            model=model,
            label_map=label_map,
            confusion_matrix=confusion_matrix,
            centers=centers,
        )

    @staticmethod
    def _create_label_map(
        model: GaussianMixture,
        dataset: dict[int, NDArray[np.float32]],
    ) -> dict[int, int]:
        """
        Create a mapping from GMM component labels to state labels.

        Parameters
        ----------
        model : GaussianMixture
            The fitted GMM model.
        dataset : dict[int, NDArray[np.float32]]
            The preprocessed dataset.

        Returns
        -------
        dict[int, int]
            A mapping from GMM component labels to state labels.
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
        model: GaussianMixture,
        dataset: dict[int, NDArray[np.float32]],
        label_map: dict[int, int],
    ) -> NDArray:
        """
        Create a confusion matrix based on the true and predicted labels.

        Parameters
        ----------
        model : GaussianMixture
            The fitted GMM model.
        dataset : dict[int, NDArray[np.float32]]
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

    @staticmethod
    def _extract_model_parameters(
        model: GaussianMixture,
    ) -> dict[int, complex]:
        """
        Extract the center (mean) of each component.

        Parameters
        ----------
        model : GaussianMixture
            The fitted GMM model.

        Returns
        -------
        dict[int, complex]
            The centers (means) of each component as complex numbers.
        """
        centers = {}
        centers_arr = model.means_
        for label in range(len(centers_arr)):
            centers[label] = complex(centers_arr[label][0], centers_arr[label][1])
        return centers

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
        component_labels = self.model.predict(real_imag_data)
        # Convert GMM component labels to state labels
        state_labels = np.array([self.label_map[label] for label in component_labels])

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
        count = np.bincount(predicted_labels, minlength=self.n_clusters)
        state = {label: count[label] for label in range(len(count))}
        return state

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
                    showlegend=False,
                    marker=dict(
                        size=10,
                        color="black",
                        symbol="x",
                    ),
                )
            )
        fig.update_layout(
            title=f"State classification of {target}",
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

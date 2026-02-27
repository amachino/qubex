"""Per-capture measurement data model."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

import qubex.visualization as viz
from qubex.backend.quel1 import SAMPLING_PERIOD
from qubex.core import DataModel
from qubex.measurement.classifiers.state_classifier import StateClassifier

from .classifier_ref import ClassifierRef
from .measurement_config import MeasurementConfig

logger = logging.getLogger(__name__)


class CaptureData(DataModel):
    """Serializable capture payload with per-target sampling metadata."""

    target: str
    raw: NDArray
    config: MeasurementConfig
    classifier_ref: ClassifierRef | None = None
    sampling_period: float = SAMPLING_PERIOD

    @model_validator(mode="after")
    def _validate_invariants(self) -> CaptureData:
        """Validate sampling period and raw-data shape constraints."""
        if self.sampling_period <= 0:
            raise ValueError("sampling_period must be positive.")

        raw = np.asarray(self.raw)
        config = self.config
        if config.shot_averaging:
            return self

        if raw.ndim == 0:
            raise ValueError(
                "raw must have at least one dimension when shot_averaging is disabled."
            )
        if config.time_integration and raw.shape[0] != config.n_shots:
            raise ValueError(
                "raw first-axis length must match config.n_shots "
                "when time_integration is enabled and shot_averaging is disabled."
            )
        if (
            not config.time_integration
            and raw.ndim >= 2
            and raw.shape[0] != config.n_shots
        ):
            raise ValueError(
                "raw shot-axis length must match config.n_shots "
                "when waveform shots are retained."
            )
        return self

    def __array__(self, dtype: Any = None) -> NDArray:
        """Return raw capture array for NumPy interoperability."""
        return np.asarray(self.raw, dtype=dtype)

    def _require_classifier(
        self,
        *,
        classifier_model: StateClassifier | None,
    ) -> StateClassifier:
        """Resolve classifier model from runtime input."""
        if classifier_model is not None:
            return classifier_model
        classifier = self.classifier
        if classifier is None:
            raise ValueError("Classifier is not set")
        return classifier

    @property
    def classifier(self) -> StateClassifier | None:
        """Return classifier resolved from `classifier_ref`."""
        if self.classifier_ref is None:
            return None
        return self.classifier_ref.load()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return raw-array shape."""
        return np.asarray(self.raw).shape

    @cached_property
    def length(self) -> int:
        """Return the number of raw samples along the primary axis."""
        raw = np.asarray(self.raw)
        if raw.ndim == 0:
            return 1
        return int(raw.shape[0])

    @cached_property
    def times(self) -> NDArray[np.float64]:
        """Return capture times in ns for the primary axis."""
        times = np.arange(self.length, dtype=np.float64) * self.sampling_period
        times.setflags(write=False)
        return times

    @cached_property
    def kerneled(self) -> NDArray:
        """Return kernel-integrated IQ samples."""
        if not self.config.shot_averaging:
            shots = np.asarray(self.raw)
            if shots.ndim <= 1:
                kerneled = np.atleast_1d(shots)
            else:
                kerneled = np.sum(shots, axis=1)
        else:
            kerneled = np.asarray(np.sum(self.raw))
        kerneled.setflags(write=False)
        return kerneled

    def get_n_states(
        self,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> int:
        """Return number of classifier states from runtime classifier model."""
        model = self._require_classifier(classifier_model=classifier_model)
        return model.n_states

    def get_soft_classified_data(
        self,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> NDArray:
        """Return soft classification probabilities for each shot."""
        if self.config.shot_averaging:
            raise ValueError("Invalid mode for classification: shot_averaging=True")
        model = self._require_classifier(classifier_model=classifier_model)
        return model.predict_proba(self.kerneled)

    def get_classified_data(
        self,
        threshold: float | None = None,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> NDArray:
        """Return classified labels with optional confidence threshold."""
        if self.config.shot_averaging:
            raise ValueError("Invalid mode for classification: shot_averaging=True")
        model = self._require_classifier(classifier_model=classifier_model)
        labels = model.predict(self.kerneled)
        if threshold is None:
            return labels
        data = self.get_soft_classified_data(classifier_model=model)
        if len(data) == 0:
            raise ValueError("No classification data available")
        max_probs = np.max(data, axis=1)
        hard_labels = np.argmax(data, axis=1)
        return np.where(max_probs > threshold, hard_labels, -1)

    def get_counts(
        self,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> dict[str, int]:
        """Return per-state counts for classified data."""
        classified = self.get_classified_data(classifier_model=classifier_model)
        if len(classified) == 0:
            raise ValueError("No classification data available")
        n_states = self.get_n_states(classifier_model=classifier_model)
        count = np.bincount(classified, minlength=n_states)
        return {str(label): int(count[label]) for label in range(len(count))}

    def get_probabilities(
        self,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> NDArray[np.float64]:
        """Return per-state probabilities for classified data."""
        counts = self.get_counts(classifier_model=classifier_model)
        total = sum(counts.values())
        if total == 0:
            raise ValueError("No classification data available")
        return np.array([count / total for count in counts.values()])

    def get_standard_deviations(
        self,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> NDArray[np.float64]:
        """Return binomial standard deviations for probabilities."""
        counts = self.get_counts(classifier_model=classifier_model)
        total = sum(counts.values())
        if total == 0:
            raise ValueError("No classification data available")
        probs = self.get_probabilities(classifier_model=classifier_model)
        return np.sqrt(probs * (1 - probs) / total)

    def get_confusion_matrix(
        self,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> NDArray:
        """Return normalized confusion matrix from runtime classifier model."""
        if self.config.shot_averaging:
            raise ValueError("Invalid mode for classification: shot_averaging=True")
        model = self._require_classifier(classifier_model=classifier_model)
        cm = model.confusion_matrix
        n_shots = cm[0].sum()
        return cm / n_shots

    def get_inverse_confusion_matrix(
        self,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> NDArray:
        """Return inverse confusion matrix from runtime classifier model."""
        confusion_matrix = self.get_confusion_matrix(classifier_model=classifier_model)
        return np.linalg.inv(confusion_matrix)

    def get_mitigated_counts(
        self,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> dict[str, int]:
        """Return error-mitigated counts using inverse confusion matrix."""
        raw = np.array(
            list(self.get_counts(classifier_model=classifier_model).values())
        )
        cm_inv = self.get_inverse_confusion_matrix(classifier_model=classifier_model)
        mitigated = raw @ cm_inv
        return {str(i): int(count) for i, count in enumerate(mitigated)}

    def get_mitigated_probabilities(
        self,
        *,
        classifier_model: StateClassifier | None = None,
    ) -> NDArray:
        """Return error-mitigated probabilities."""
        raw = np.array(
            list(self.get_counts(classifier_model=classifier_model).values())
        )
        cm_inv = self.get_inverse_confusion_matrix(classifier_model=classifier_model)
        mitigated = raw @ cm_inv
        total = float(np.sum(mitigated))
        if total == 0:
            raise ValueError("No classification data available")
        return mitigated / total

    def plot(
        self,
        title: str | None = None,
        return_figure: bool = False,
        save_image: bool = False,
    ) -> Any:
        """Plot capture data according to measurement configuration."""
        plot_title = title or f"Readout data : {self.target}"
        if self.config.time_integration:
            data = {self.target: np.atleast_1d(self.kerneled)}
            if return_figure:
                fig = viz.make_iq_scatter_figure(data=data, title=plot_title)
                if save_image:
                    viz.save_figure(fig, name="plot_state_distribution")
                return fig
            viz.scatter_iq_data(data=data, title=plot_title, save_image=save_image)
            return None

        waveform = np.asarray(self.raw)
        if not self.config.shot_averaging and waveform.ndim >= 2:
            waveform = np.mean(waveform, axis=0)
        waveform = np.squeeze(waveform)
        if return_figure:
            fig = viz.make_waveform_figure(
                data=waveform,
                sampling_period=self.sampling_period,
                title=plot_title,
                xlabel="Capture time (ns)",
                ylabel="Signal (arb. units)",
            )
            if save_image:
                viz.save_figure(fig, name="plot_waveform")
            return fig
        viz.plot_waveform(
            data=waveform,
            sampling_period=self.sampling_period,
            title=plot_title,
            xlabel="Capture time (ns)",
            ylabel="Signal (arb. units)",
            save_image=save_image,
        )
        return None

    def plot_fft(
        self,
        title: str | None = None,
        return_figure: bool = False,
        save_image: bool = False,
    ) -> Any:
        """Plot FFT of capture waveform data."""
        plot_title = title or f"Fourier transform : {self.target}"
        if self.config.time_integration:
            logger.info(
                "Skipping FFT plot for %s: data is not waveform data.",
                self.target,
            )
            return None
        waveform = np.asarray(self.raw)
        if not self.config.shot_averaging and waveform.ndim >= 2:
            waveform = np.mean(waveform, axis=0)
        waveform = np.squeeze(waveform)
        if np.asarray(waveform).ndim == 0:
            waveform = np.atleast_1d(waveform)
        times = np.arange(len(waveform)) * self.sampling_period
        if return_figure:
            fig = viz.make_fft_figure(
                x=times * 1e-3,
                y=waveform,
                title=plot_title,
                xlabel="Frequency (MHz)",
                ylabel="Signal (arb. units)",
            )
            if save_image:
                viz.save_figure(fig, name="plot_fft")
            return fig
        viz.plot_fft(
            x=times * 1e-3,
            y=waveform,
            title=plot_title,
            xlabel="Frequency (MHz)",
            ylabel="Signal (arb. units)",
            save_image=save_image,
        )
        return None

    def save(
        self,
        path: str | Path,
    ) -> Path:
        """Alias of `save_netcdf`."""
        return self.save_netcdf(path)

    @classmethod
    def load(cls, path: str | Path) -> CaptureData:
        """Alias of `load_netcdf`."""
        return cls.load_netcdf(path)

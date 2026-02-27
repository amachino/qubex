"""Measurement result model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np

import qubex.visualization as viz
from qubex.core import DataModel
from qubex.measurement.classifiers.state_classifier import StateClassifier

from .capture_data import CaptureData
from .measure_result import MeasureMode
from .measurement_config import MeasurementConfig


class MeasurementResult(DataModel):
    """Canonical serializable result of a measurement run."""

    measurement_config: MeasurementConfig
    data: dict[str, list[CaptureData]]
    device_config: dict[str, Any] | None = None

    @property
    def mode(self) -> MeasureMode:
        """Return legacy-like mode derived from measurement configuration."""
        return (
            MeasureMode.AVG
            if self.measurement_config.shot_averaging
            else MeasureMode.SINGLE
        )

    def __repr__(self) -> str:
        """Return a concise representation of the result."""
        targets = ", ".join(self.data.keys())
        return f"<MeasurementResult mode={self.mode.value}, targets=[{targets}]>"

    def get_basis_indices(
        self,
        targets: Collection[str] | None = None,
        *,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> list[tuple[int, ...]]:
        """Return basis indices for selected targets."""
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        if targets is None:
            targets = self.data.keys()
        dimensions: list[int] = []
        for target in targets:
            capture = self._get_capture(target=target, index=0)
            classifier = self._resolve_classifier(
                target=target,
                capture=capture,
                classifiers=classifiers,
            )
            dimensions.append(capture.get_n_states(classifier_model=classifier))
        return list(np.ndindex(*list(dimensions)))

    def get_basis_labels(
        self,
        targets: Collection[str] | None = None,
        *,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> list[str]:
        """Return basis labels for selected targets."""
        basis_indices = self.get_basis_indices(targets, classifiers=classifiers)
        return ["".join(str(i) for i in basis) for basis in basis_indices]

    def get_classified_data(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        threshold: float | None = None,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> np.ndarray:
        """Return classified labels for selected targets."""
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        target_tuples = self._normalize_target_tuples(targets=targets)
        return np.column_stack(
            [
                self._get_capture(target=target, index=index).get_classified_data(
                    threshold=threshold,
                    classifier_model=self._resolve_classifier(
                        target=target,
                        capture=self._get_capture(target=target, index=index),
                        classifiers=classifiers,
                    ),
                )
                for (target, index) in target_tuples
            ]
        )

    def get_counts(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        threshold: float | None = None,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> Counter:
        """Return counts of classified bitstrings."""
        classified_data = self.get_classified_data(
            targets,
            threshold=threshold,
            classifiers=classifiers,
        )
        labels = np.array(["".join(map(str, row)) for row in classified_data])
        return Counter(labels)

    def get_probabilities(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        threshold: float | None = None,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> dict[str, float]:
        """Return probabilities of classified bitstrings."""
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        counts = self.get_counts(
            targets=targets,
            threshold=threshold,
            classifiers=classifiers,
        )
        total = sum(counts.values())
        if total == 0:
            return {}
        return {key: count / total for key, count in counts.items()}

    def get_standard_deviations(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        threshold: float | None = None,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> dict[str, float]:
        """Return binomial standard deviations for probabilities."""
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        counts = self.get_counts(
            targets=targets,
            threshold=threshold,
            classifiers=classifiers,
        )
        probs = self.get_probabilities(
            targets=targets,
            threshold=threshold,
            classifiers=classifiers,
        )
        return {
            key: float(np.sqrt(prob * (1 - prob) / total))
            for key, prob, total in zip(
                counts.keys(),
                probs.values(),
                counts.values(),
                strict=True,
            )
        }

    def get_classifier(
        self,
        target: str,
        *,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> StateClassifier:
        """Return classifier for the target label."""
        capture = self._get_capture(target=target, index=0)
        classifier = self._resolve_classifier(
            target=target,
            capture=capture,
            classifiers=classifiers,
        )
        if classifier is None:
            raise ValueError(f"Classifier for target {target} is not set")
        return classifier

    def get_confusion_matrix(
        self,
        targets: Collection[str] | None = None,
        *,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> np.ndarray:
        """Return Kronecker-product confusion matrix."""
        if targets is None:
            targets = self.data.keys()
        confusion_matrices = []
        for target in targets:
            capture = self._get_capture(target=target, index=0)
            confusion_matrices.append(
                capture.get_confusion_matrix(
                    classifier_model=self._resolve_classifier(
                        target=target,
                        capture=capture,
                        classifiers=classifiers,
                    )
                )
            )
        return reduce(np.kron, confusion_matrices)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str] | None = None,
        *,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> np.ndarray:
        """Return inverse confusion matrix."""
        confusion_matrix = self.get_confusion_matrix(
            targets=targets,
            classifiers=classifiers,
        )
        return np.linalg.inv(confusion_matrix)

    def get_mitigated_counts(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> dict[str, int]:
        """Return error-mitigated counts using inverse confusion matrix."""
        labels = self._extract_labels_from_targets(targets=targets)
        basis_labels = self.get_basis_labels(labels, classifiers=classifiers)
        raw_counts = self.get_counts(targets=targets, classifiers=classifiers)
        raw = np.array([raw_counts.get(label, 0) for label in basis_labels])
        cm_inv = self.get_inverse_confusion_matrix(labels, classifiers=classifiers)
        mitigated = raw @ cm_inv
        return {
            basis_label: int(mitigated[i]) for i, basis_label in enumerate(basis_labels)
        }

    def get_mitigated_probabilities(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        classifiers: Mapping[str, StateClassifier] | None = None,
    ) -> dict[str, float]:
        """Return error-mitigated probabilities."""
        labels = self._extract_labels_from_targets(targets=targets)
        basis_labels = self.get_basis_labels(labels, classifiers=classifiers)
        raw_probs = self.get_probabilities(targets=targets, classifiers=classifiers)
        raw = np.array([raw_probs.get(label, 0.0) for label in basis_labels])
        cm_inv = self.get_inverse_confusion_matrix(labels, classifiers=classifiers)
        mitigated = raw @ cm_inv
        return {
            basis_label: float(mitigated[i])
            for i, basis_label in enumerate(basis_labels)
        }

    def plot(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ) -> Any:
        """Plot measurement data for each capture."""
        figures: list[Any] = []
        for target, captures in self.data.items():
            for capture_index, capture in enumerate(captures):
                title = f"{target} : data[{capture_index}]"
                config = capture.config
                if config.time_integration:
                    shots = np.asarray(capture.raw)
                    kerneled = np.atleast_1d(
                        shots if shots.ndim <= 1 else np.sum(shots, axis=1)
                    )
                    data = {target: kerneled}
                    if return_figure:
                        figure = viz.make_iq_scatter_figure(
                            data=data,
                            title=title,
                        )
                        if save_image:
                            viz.save_figure(figure, name="plot_state_distribution")
                        figures.append(figure)
                    else:
                        viz.scatter_iq_data(
                            data=data,
                            title=title,
                            save_image=save_image,
                        )
                    continue

                waveform = np.asarray(capture.raw)
                if not config.shot_averaging and waveform.ndim >= 2:
                    waveform = np.mean(waveform, axis=0)
                waveform = np.squeeze(waveform)
                if return_figure:
                    figure = viz.make_waveform_figure(
                        data=waveform,
                        sampling_period=capture.sampling_period,
                        title=title,
                        xlabel="Capture time (ns)",
                        ylabel="Signal (arb. units)",
                    )
                    if save_image:
                        viz.save_figure(figure, name="plot_waveform")
                    figures.append(figure)
                else:
                    viz.plot_waveform(
                        data=waveform,
                        sampling_period=capture.sampling_period,
                        title=title,
                        xlabel="Capture time (ns)",
                        ylabel="Signal (arb. units)",
                        save_image=save_image,
                    )
        if return_figure:
            return figures
        return None

    def save(
        self,
        path: str | Path,
    ) -> Path:
        """Alias of `save_netcdf`."""
        return self.save_netcdf(path)

    @classmethod
    def load(cls, path: str | Path) -> MeasurementResult:
        """Alias of `load_netcdf`."""
        return cls.load_netcdf(path)

    def _normalize_target_tuples(
        self,
        *,
        targets: Collection[str | tuple[str, int]] | None,
    ) -> list[tuple[str, int]]:
        """Normalize target specification into `(target, index)` tuples."""
        if targets is None:
            return [(target, -1) for target in self.data]
        target_tuples: list[tuple[str, int]] = []
        for target in targets:
            if isinstance(target, str):
                target_tuples.append((target, -1))
            elif isinstance(target, tuple) and len(target) == 2:
                target_tuples.append((str(target[0]), int(target[1])))
            else:
                raise ValueError(f"Invalid target format: {target}")
        return target_tuples

    def _extract_labels_from_targets(
        self,
        *,
        targets: Collection[str | tuple[str, int]] | None,
    ) -> list[str]:
        """Extract target labels from target selector values."""
        if targets is None:
            return list(self.data.keys())
        labels: list[str] = []
        for target in targets:
            if isinstance(target, str):
                labels.append(target)
            elif isinstance(target, tuple) and len(target) == 2:
                labels.append(str(target[0]))
            else:
                raise ValueError(f"Invalid target format: {target}")
        return labels

    def _get_capture(
        self,
        *,
        target: str,
        index: int,
    ) -> CaptureData:
        """Return one capture entry by target/index."""
        if target not in self.data:
            raise ValueError(f"Target {target} not found in data")
        captures = self.data[target]
        if len(captures) == 0:
            raise ValueError(f"Target {target} has no capture data")
        if not (-len(captures) <= index < len(captures)):
            raise IndexError(
                f"Capture index {index} is out of range for target {target}."
            )
        return captures[index]

    @staticmethod
    def _resolve_classifier(
        *,
        target: str,
        capture: CaptureData,
        classifiers: Mapping[str, StateClassifier] | None,
    ) -> StateClassifier | None:
        """Resolve classifier model from target/capture references."""
        if classifiers is not None:
            if target in classifiers:
                return classifiers[target]
            if capture.classifier_ref is not None:
                classifier_from_path = classifiers.get(capture.classifier_ref.path)
                if classifier_from_path is not None:
                    return classifier_from_path
        return capture.classifier

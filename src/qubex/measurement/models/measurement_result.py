"""Measurement result model."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Collection, Mapping
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import model_validator

import qubex.visualization as viz
from qubex.core import DataModel
from qubex.measurement.classifiers.state_classifier import StateClassifier

from .capture_data import CaptureData
from .classifier_ref import ClassifierRef
from .measurement_config import MeasurementConfig


class MeasurementResult(DataModel):
    """Canonical serializable result of a measurement run."""

    data: dict[str, list[CaptureData]]
    measurement_config: MeasurementConfig
    device_config: dict[str, Any] | None = None
    classifier_refs: dict[str, ClassifierRef] | None = None

    @model_validator(mode="after")
    def _validate_classifier_refs(self) -> MeasurementResult:
        """Validate classifier-ref keys and infer mapping from capture metadata."""
        if self.classifier_refs is not None:
            unknown_targets = sorted(set(self.classifier_refs) - set(self.data))
            if unknown_targets:
                joined = ", ".join(unknown_targets)
                raise ValueError(
                    "classifier_refs contains unknown targets not present in data: "
                    f"{joined}."
                )
            return self

        inferred: dict[str, ClassifierRef] = {}
        for target, captures in self.data.items():
            target_ref: ClassifierRef | None = None
            for capture in captures:
                capture_ref = capture.classifier_ref
                if capture_ref is None:
                    continue
                if target_ref is None:
                    target_ref = capture_ref
                    continue
                if capture_ref != target_ref:
                    raise ValueError(
                        "Multiple classifier_ref values found in captures for "
                        f"target {target}."
                    )
            if target_ref is not None:
                inferred[target] = target_ref
        if len(inferred) > 0:
            object.__setattr__(self, "classifier_refs", inferred)
        return self

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
            if classifier is None:
                raise ValueError(f"Classifier for target {target} is not set")
            dimensions.append(classifier.n_states)
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
                self._classify_capture(
                    capture=self._get_capture(target=target, index=index),
                    threshold=threshold,
                    classifier=self._resolve_classifier(
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
            classifier = self._resolve_classifier(
                target=target,
                capture=capture,
                classifiers=classifiers,
            )
            if classifier is None:
                raise ValueError(f"Classifier for target {target} is not set")
            confusion_matrices.append(
                self._compute_confusion_matrix(
                    capture=capture,
                    classifier=classifier,
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
        if return_figure:
            warnings.warn(
                "`return_figure` is deprecated; use `figure()` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            figures = self.figure()
            if save_image:
                figure_index = 0
                for captures in self.data.values():
                    for capture in captures:
                        waveform = np.asarray(capture.data)
                        use_scatter = capture.config.time_integration or (
                            not capture.config.shot_averaging and waveform.ndim >= 2
                        )
                        figure_name = (
                            "plot_state_distribution"
                            if use_scatter
                            else "plot_waveform"
                        )
                        viz.save_figure(figures[figure_index], name=figure_name)
                        figure_index += 1
            return figures

        for target, captures in self.data.items():
            for capture_index, capture in enumerate(captures):
                title = f"{target} : data[{capture_index}]"
                config = capture.config
                if config.time_integration:
                    shots = np.asarray(capture.data)
                    kerneled = np.atleast_1d(
                        shots if shots.ndim <= 1 else np.sum(shots, axis=1)
                    )
                    viz.scatter_iq_data(
                        data={target: kerneled},
                        title=title,
                        save_image=save_image,
                    )
                    continue

                waveform = np.asarray(capture.data)
                if not config.shot_averaging and waveform.ndim >= 2:
                    shot_iq = np.mean(waveform, axis=1)
                    viz.scatter_iq_data(
                        data={target: np.atleast_1d(shot_iq)},
                        title=f"Readout IQ data : {target}",
                        save_image=save_image,
                    )
                    continue
                waveform = np.squeeze(waveform)
                waveform_title = f"Readout waveform : {target}"
                viz.plot_waveform(
                    data=waveform,
                    sampling_period=capture.sampling_period,
                    title=waveform_title,
                    xlabel="Capture time (ns)",
                    ylabel="Signal (arb. units)",
                    save_image=save_image,
                )
        return None

    def figure(self) -> list[Any]:
        """Return figure objects for all capture entries without rendering."""
        figures: list[Any] = []
        for target, captures in self.data.items():
            for capture_index, capture in enumerate(captures):
                title = f"{target} : data[{capture_index}]"
                config = capture.config
                if config.time_integration:
                    shots = np.asarray(capture.data)
                    kerneled = np.atleast_1d(
                        shots if shots.ndim <= 1 else np.sum(shots, axis=1)
                    )
                    figures.append(
                        viz.make_iq_scatter_figure(
                            data={target: kerneled},
                            title=title,
                        )
                    )
                    continue

                waveform = np.asarray(capture.data)
                if not config.shot_averaging and waveform.ndim >= 2:
                    shot_iq = np.mean(waveform, axis=1)
                    figures.append(
                        viz.make_iq_scatter_figure(
                            data={target: np.atleast_1d(shot_iq)},
                            title=f"Readout IQ data : {target}",
                        )
                    )
                    continue
                waveform = np.squeeze(waveform)
                waveform_title = f"Readout waveform : {target}"
                figures.append(
                    viz.make_waveform_figure(
                        data=waveform,
                        sampling_period=capture.sampling_period,
                        title=waveform_title,
                        xlabel="Capture time (ns)",
                        ylabel="Signal (arb. units)",
                    )
                )
        return figures

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

    def _resolve_classifier(
        self,
        *,
        target: str,
        capture: CaptureData,
        classifiers: Mapping[str, StateClassifier] | None,
    ) -> StateClassifier | None:
        """Resolve classifier model from target/capture references."""
        classifier_ref = None
        if self.classifier_refs is not None:
            classifier_ref = self.classifier_refs.get(target)
        if classifier_ref is None:
            classifier_ref = capture.classifier_ref

        if classifiers is not None:
            if target in classifiers:
                return classifiers[target]
            if classifier_ref is not None:
                classifier_from_path = classifiers.get(classifier_ref.path)
                if classifier_from_path is not None:
                    return classifier_from_path
        if classifier_ref is None:
            return None
        return classifier_ref.load()

    @staticmethod
    def _compute_kerneled_data(
        *,
        capture: CaptureData,
    ) -> np.ndarray:
        """Compute kernel-integrated IQ samples from capture primary data."""
        data = np.asarray(capture.data)
        if not capture.config.shot_averaging:
            if data.ndim <= 1:
                kerneled = np.atleast_1d(data)
            else:
                kerneled = np.sum(data, axis=1)
        else:
            kerneled = np.asarray(np.sum(data))
        kerneled.setflags(write=False)
        return kerneled

    @staticmethod
    def _classify_capture(
        *,
        capture: CaptureData,
        threshold: float | None,
        classifier: StateClassifier | None,
    ) -> np.ndarray:
        """Classify capture shots into state labels."""
        if capture.config.shot_averaging:
            raise ValueError("Invalid mode for classification: shot_averaging=True")
        if threshold is None and capture.payload.state_series is not None:
            return capture.payload.state_series
        if classifier is None:
            raise ValueError("Classifier is not set")
        kerneled = MeasurementResult._compute_kerneled_data(capture=capture)
        labels = classifier.predict(kerneled)
        if threshold is None:
            return labels
        data = classifier.predict_proba(kerneled)
        if len(data) == 0:
            raise ValueError("No classification data available")
        max_probs = np.max(data, axis=1)
        hard_labels = np.argmax(data, axis=1)
        return np.where(max_probs > threshold, hard_labels, -1)

    @staticmethod
    def _compute_confusion_matrix(
        *,
        capture: CaptureData,
        classifier: StateClassifier,
    ) -> np.ndarray:
        """Compute normalized confusion matrix for one capture target."""
        if capture.config.shot_averaging:
            raise ValueError("Invalid mode for classification: shot_averaging=True")
        confusion_matrix = classifier.confusion_matrix
        n_shots = confusion_matrix[0].sum()
        return confusion_matrix / n_shots

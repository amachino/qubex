"""Measurement result model."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Collection
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Any, TypeAlias, TypeGuard

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

import qubex.visualization as viz
from qubex.core import DataModel

from .capture_data import CaptureData
from .classifier_ref import ClassifierRef
from .measurement_config import MeasurementConfig

TargetSpecifier: TypeAlias = str | tuple[str, int]
TargetSelection: TypeAlias = TargetSpecifier | Collection[TargetSpecifier] | None
_DEFAULT_CAPTURE_INDEX = -1


def _is_target_capture_spec(value: object) -> TypeGuard[tuple[str, int]]:
    """Return whether value is a `(target, capture_index)` selector."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


class MeasurementResult(DataModel):
    """Canonical serializable result of a measurement run."""

    data: dict[str, list[CaptureData]]
    measurement_config: MeasurementConfig
    device_config: dict[str, Any] | None = None
    classifier_refs: dict[str, ClassifierRef] | None = None

    def __repr__(self) -> str:
        """Return a concise summary without embedding full capture payloads."""
        targets = ", ".join(self.data.keys())
        captures = sum(len(captures) for captures in self.data.values())
        config = self.measurement_config
        return (
            "MeasurementResult("
            f"targets=[{targets}], "
            f"captures={captures}, "
            f"shot_averaging={config.shot_averaging}, "
            f"time_integration={config.time_integration}, "
            f"state_classification={config.state_classification})"
        )

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

    def _resolve_target_captures(
        self,
        targets: TargetSelection,
    ) -> list[tuple[str, int]]:
        """Normalize target/capture selection while preserving caller order."""
        if len(self.data) == 0:
            raise ValueError("No measurement data available.")
        if targets is None:
            selections: list[object] = list(self.data)
        elif isinstance(targets, str) or _is_target_capture_spec(targets):
            selections = [targets]
        else:
            selections = list(targets)
        if len(selections) == 0:
            raise ValueError("No targets were selected.")

        target_captures: list[tuple[str, int]] = []
        for selection in selections:
            if isinstance(selection, str):
                target_captures.append((selection, _DEFAULT_CAPTURE_INDEX))
            elif _is_target_capture_spec(selection):
                target_captures.append(selection)
            else:
                raise ValueError(
                    "Target selection must be a target label or "
                    "`(target, capture_index)` tuple."
                )

        missing_targets = [
            target for target, _ in target_captures if target not in self.data
        ]
        if missing_targets:
            joined = ", ".join(missing_targets)
            raise ValueError(f"Targets not found in data: {joined}.")
        return target_captures

    def _resolve_targets(
        self,
        targets: TargetSelection,
    ) -> list[str]:
        """Normalize target selection and ignore capture indices."""
        target_captures = self._resolve_target_captures(
            targets,
        )
        target_list = [target for target, _ in target_captures]
        return target_list

    def _get_capture(self, target: str, capture_index: int) -> CaptureData:
        """Return one capture for a target with a clear error on bad index."""
        captures = self.data[target]
        if len(captures) == 0:
            raise ValueError(f"Target {target} has no capture data.")
        try:
            return captures[capture_index]
        except IndexError as exc:
            raise IndexError(
                f"capture_index {capture_index} is out of range for target {target}."
            ) from exc

    def _get_state_series(
        self,
        target: str,
        capture_index: int,
    ) -> NDArray[np.int64]:
        """Return classified state labels stored in a capture."""
        capture = self._get_capture(target, capture_index)
        state_series = capture.state_series
        if state_series is None:
            raise ValueError(
                f"Capture for target {target} at index {capture_index} does not "
                "contain state_series. Run measurement with state_classification=True."
            )
        return np.asarray(state_series, dtype=np.int64)

    def get_classified_data(
        self,
        targets: TargetSelection = None,
    ) -> NDArray[np.int64]:
        """
        Return classified state labels stacked by target.

        For one target this returns shape ``(n_shots, 1)``. For multiple targets,
        each row is one simultaneous shot and each column follows the requested
        target order.
        """
        target_captures = self._resolve_target_captures(
            targets,
        )
        state_series = [
            self._get_state_series(target, selected_capture_index)
            for target, selected_capture_index in target_captures
        ]
        shot_counts = {series.shape[0] for series in state_series}
        if len(shot_counts) != 1:
            raise ValueError(
                "Selected targets have different classified shot counts: "
                f"{sorted(shot_counts)}."
            )
        return np.asarray(np.column_stack(state_series), dtype=np.int64)

    def get_memory(
        self,
        targets: TargetSelection = None,
    ) -> list[str]:
        """Return per-shot classified memory as bitstring labels."""
        classified_data = self.get_classified_data(targets)
        return ["".join(map(str, row)) for row in classified_data]

    def get_counts(
        self,
        targets: TargetSelection = None,
    ) -> Counter[str]:
        """Return counts of classified bitstrings."""
        return Counter(self.get_memory(targets))

    def get_probabilities(
        self,
        targets: TargetSelection = None,
    ) -> dict[str, float]:
        """Return probabilities of classified bitstrings."""
        counts = self.get_counts(targets)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {key: count / total for key, count in counts.items()}

    def get_standard_deviations(
        self,
        targets: TargetSelection = None,
    ) -> dict[str, float]:
        """Return binomial standard deviations for classified probabilities."""
        counts = self.get_counts(targets)
        total = sum(counts.values())
        if total == 0:
            return {}
        probabilities = {key: count / total for key, count in counts.items()}
        return {
            key: float(np.sqrt(prob * (1.0 - prob) / total))
            for key, prob in probabilities.items()
        }

    def _get_classifier_ref(self, target: str) -> ClassifierRef:
        """Return classifier reference for a target."""
        if self.classifier_refs is None or target not in self.classifier_refs:
            raise ValueError(
                f"Classifier reference for target {target} is not set. "
                "Run measurement with classifier persistence or attach classifier_refs."
            )
        return self.classifier_refs[target]

    def _get_classifier_state_count(self, target: str) -> int:
        """Return number of classifier states for a target."""
        return int(self._get_classifier_ref(target).load().n_states)

    def get_basis_indices(
        self,
        targets: TargetSelection = None,
    ) -> list[tuple[int, ...]]:
        """Return basis index tuples for selected targets."""
        target_list = self._resolve_targets(targets)
        dimensions = [
            self._get_classifier_state_count(target) for target in target_list
        ]
        return list(product(*[range(dimension) for dimension in dimensions]))

    def get_basis_labels(
        self,
        targets: TargetSelection = None,
    ) -> list[str]:
        """Return basis labels for selected targets."""
        return [
            "".join(str(index) for index in basis)
            for basis in self.get_basis_indices(targets)
        ]

    def get_confusion_matrix(
        self,
        targets: TargetSelection = None,
    ) -> NDArray[np.float64]:
        """Return row-normalized Kronecker-product confusion matrix."""
        target_list = self._resolve_targets(targets)
        confusion_matrices: list[NDArray[np.float64]] = []
        for target in target_list:
            matrix = np.asarray(
                self._get_classifier_ref(target).load().confusion_matrix,
                dtype=np.float64,
            )
            row_sums = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix, dtype=np.float64),
                where=row_sums != 0,
            )
            confusion_matrices.append(normalized)
        return np.asarray(reduce(np.kron, confusion_matrices), dtype=np.float64)

    def get_inverse_confusion_matrix(
        self,
        targets: TargetSelection = None,
    ) -> NDArray[np.float64]:
        """Return inverse confusion matrix for selected targets."""
        return np.asarray(
            np.linalg.inv(self.get_confusion_matrix(targets)),
            dtype=np.float64,
        )

    def get_mitigated_probabilities(
        self,
        targets: TargetSelection = None,
    ) -> dict[str, float]:
        """Return readout-error-mitigated probabilities for classified bitstrings."""
        basis_labels = self.get_basis_labels(targets)
        raw_probabilities = self.get_probabilities(targets)
        raw = np.array(
            [raw_probabilities.get(label, 0.0) for label in basis_labels],
            dtype=float,
        )
        mitigated = raw @ self.get_inverse_confusion_matrix(targets)
        return {
            basis_label: float(mitigated[index])
            for index, basis_label in enumerate(basis_labels)
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

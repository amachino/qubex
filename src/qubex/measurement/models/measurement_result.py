"""Measurement result model."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import model_validator

import qubex.visualization as viz
from qubex.core import DataModel

from .capture_data import CaptureData
from .classifier_ref import ClassifierRef
from .measurement_config import MeasurementConfig


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

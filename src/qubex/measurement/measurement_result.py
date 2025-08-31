from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import cached_property, reduce
from pathlib import Path
from typing import Collection, Literal

import numpy as np
from numpy.typing import NDArray

from ..analysis import visualization as viz
from ..backend import SAMPLING_PERIOD
from .measurement_record import MeasurementRecord
from .state_classifier import StateClassifier

SAMPLING_PERIOD_SINGLE = SAMPLING_PERIOD
SAMPLING_PERIOD_AVG = SAMPLING_PERIOD * 4


class MeasureMode(Enum):
    SINGLE = "single"
    AVG = "avg"

    @cached_property
    def integral_mode(self) -> str:
        if self == MeasureMode.SINGLE:
            return "single"
        elif self == MeasureMode.AVG:
            return "integral"
        else:
            raise ValueError(f"Invalid mode: {self}")


@dataclass(frozen=True)
class MeasureData:
    target: str
    mode: MeasureMode
    raw: NDArray
    classifier: StateClassifier | None = None

    @cached_property
    def n_states(self) -> int:
        if self.classifier is None:
            raise ValueError("Classifier is not set")
        else:
            return self.classifier.n_states

    @cached_property
    def kerneled(
        self,
    ) -> NDArray:
        if self.mode == MeasureMode.SINGLE:
            return np.sum(self.raw, axis=1)
        elif self.mode == MeasureMode.AVG:
            return np.asarray(np.sum(self.raw))
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @cached_property
    def classified(self) -> NDArray:
        if self.mode == MeasureMode.SINGLE:
            if self.classifier is not None:
                return self.classifier.predict(self.kerneled)
            else:
                raise ValueError("Classifier is not set")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @cached_property
    def length(self) -> int:
        return len(self.raw)

    @cached_property
    def times(self) -> NDArray[np.float64]:
        if self.mode == MeasureMode.SINGLE:
            return np.arange(self.length) * SAMPLING_PERIOD_SINGLE
        elif self.mode == MeasureMode.AVG:
            return np.arange(self.length) * SAMPLING_PERIOD_AVG
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @cached_property
    def counts(self) -> dict[str, int]:
        if len(self.classified) == 0:
            raise ValueError("No classification data available")
        classified_labels = self.classified
        count = np.bincount(classified_labels, minlength=self.n_states)
        state = {str(label): count[label] for label in range(len(count))}
        return state

    @cached_property
    def probabilities(self) -> NDArray[np.float64]:
        if len(self.classified) == 0:
            raise ValueError("No classification data available")
        total = sum(self.counts.values())
        return np.array([count / total for count in self.counts.values()])

    @cached_property
    def standard_deviations(self) -> NDArray[np.float64]:
        if len(self.classified) == 0:
            raise ValueError("No classification data available")
        return np.sqrt(
            self.probabilities * (1 - self.probabilities) / sum(self.counts.values())
        )

    @cached_property
    def confusion_matrix(self) -> NDArray:
        if self.mode == MeasureMode.SINGLE:
            if self.classifier is not None:
                cm = self.classifier.confusion_matrix
                n_shots = cm[0].sum()
                return cm / n_shots
            else:
                raise ValueError("Classifier is not set")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @cached_property
    def inverse_confusion_matrix(self) -> NDArray:
        if self.mode == MeasureMode.SINGLE:
            if self.classifier is not None:
                cm = self.confusion_matrix
                # if np.linalg.det(cm) == 0:
                #     raise ValueError("Confusion matrix is singular")
                return np.linalg.inv(cm)
            else:
                raise ValueError("Classifier is not set")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @cached_property
    def mitigated_counts(self) -> dict[str, int]:
        if self.mode == MeasureMode.SINGLE:
            if self.classifier is not None:
                cm_inv = self.inverse_confusion_matrix
                raw = np.array(list(self.counts.values()))
                mitigated_counts = raw @ cm_inv
                return {str(i): int(count) for i, count in enumerate(mitigated_counts)}
            else:
                raise ValueError("Classifier is not set")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @cached_property
    def mitigated_probabilities(self) -> NDArray:
        if self.mode == MeasureMode.SINGLE:
            if self.classifier is not None:
                cm_inv = self.inverse_confusion_matrix
                raw = np.array(list(self.counts.values()))
                mitigated = raw @ cm_inv
                total = sum(mitigated)
                return mitigated / total
            else:
                raise ValueError("Classifier is not set")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def get_soft_classified_data(
        self,
    ) -> NDArray:
        if self.mode == MeasureMode.SINGLE:
            if self.classifier is not None:
                return self.classifier.predict_proba(self.kerneled)
            else:
                raise ValueError("Classifier is not set")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def get_classified_data(
        self,
        threshold: float | None = None,
    ) -> NDArray:
        if threshold is None:
            return self.classified
        else:
            data = self.get_soft_classified_data()
            if len(data) == 0:
                raise ValueError("No classification data available")
            max_probs = np.max(data, axis=1)
            labels = np.argmax(data, axis=1)
            result = np.where(max_probs > threshold, labels, -1)
            return result

    def plot(
        self,
        title: str | None = None,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        if self.mode == MeasureMode.SINGLE:
            return viz.scatter_iq_data(
                data={self.target: np.asarray(self.kerneled)},
                title=title or f"Readout IQ data : {self.target}",
                return_figure=return_figure,
                save_image=save_image,
            )
        elif self.mode == MeasureMode.AVG:
            return viz.plot_waveform(
                data=self.raw,
                sampling_period=SAMPLING_PERIOD_AVG,
                title=title or f"Readout waveform : {self.target}",
                xlabel="Capture time (ns)",
                ylabel="Signal (arb. units)",
                return_figure=return_figure,
                save_image=save_image,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def plot_fft(
        self,
        title: str | None = None,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        return viz.plot_fft(
            x=self.times * 1e-3,
            y=self.raw,
            title=title or f"Fourier transform : {self.target}",
            xlabel="Frequency (MHz)",
            ylabel="Signal (arb. units)",
            return_figure=return_figure,
            save_image=save_image,
        )


@dataclass(frozen=True)
class MeasureResult:
    mode: MeasureMode
    data: dict[str, MeasureData]
    config: dict

    @cached_property
    def counts(self) -> dict[str, int]:
        return self.get_counts()

    @cached_property
    def probabilities(self) -> dict[str, float]:
        return self.get_probabilities()

    @cached_property
    def standard_deviations(self) -> dict[str, float]:
        return self.get_standard_deviations()

    @cached_property
    def mitigated_counts(self) -> dict[str, int]:
        return self.get_mitigated_counts()

    @cached_property
    def mitigated_probabilities(self) -> dict[str, float]:
        return self.get_mitigated_probabilities()

    def get_basis_indices(
        self,
        targets: Collection[str] | None = None,
    ) -> list[tuple[int, ...]]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        if targets is None:
            targets = self.data.keys()
        dimensions = [self.data[target].n_states for target in targets]
        return list(np.ndindex(*[dim for dim in dimensions]))

    def get_basis_labels(
        self,
        targets: Collection[str] | None = None,
    ) -> list[str]:
        basis_indices = self.get_basis_indices(targets)
        return ["".join(str(i) for i in basis) for basis in basis_indices]

    def get_classified_data(
        self,
        targets: Collection[str] | None = None,
        *,
        threshold: float | None = None,
    ) -> NDArray:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        if targets is None:
            targets = self.data.keys()
        return np.column_stack(
            [
                self.data[target].get_classified_data(threshold=threshold)
                for target in targets
            ]
        )

    def get_memory(
        self,
        targets: Collection[str] | None = None,
        *,
        threshold: float | None = None,
    ) -> list[str]:
        """
        Returns memory: list of bitstrings (e.g., ['0110', '1010', ...])
        representing each shot's classified result.
        """
        classified_data = self.get_classified_data(targets, threshold=threshold)
        return ["".join(map(str, row)) for row in classified_data if all(row >= 0)]

    def get_counts(
        self,
        targets: Collection[str] | None = None,
        *,
        threshold: float | None = None,
    ) -> Counter:
        classified_labels = self.get_memory(targets, threshold=threshold)
        return Counter(classified_labels)

    def get_probabilities(
        self,
        targets: Collection[str] | None = None,
        *,
        threshold: float | None = None,
    ) -> dict[str, float]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        counts = self.get_counts(targets, threshold=threshold)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {key: count / total for key, count in counts.items()}

    def get_standard_deviations(
        self,
        targets: Collection[str] | None = None,
        *,
        threshold: float | None = None,
    ) -> dict[str, float]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        counts = self.get_counts(targets, threshold=threshold)
        probs = self.get_probabilities(targets, threshold=threshold)
        return {
            key: np.sqrt(prob * (1 - prob) / total)
            for key, prob, total in zip(
                counts.keys(),
                probs.values(),
                counts.values(),
            )
        }

    def get_classifier(self, target: str) -> StateClassifier:
        if target not in self.data:
            raise ValueError(f"Target {target} not found in data")
        classifier = self.data[target].classifier
        if classifier is None:
            raise ValueError(f"Classifier for target {target} is not set")
        return classifier

    def get_confusion_matrix(
        self,
        targets: Collection[str] | None = None,
    ) -> NDArray:
        if targets is None:
            targets = self.data.keys()
        confusion_matrices = [self.data[target].confusion_matrix for target in targets]
        return reduce(np.kron, confusion_matrices)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str] | None = None,
    ) -> NDArray:
        if targets is None:
            targets = self.data.keys()
        confusion_matrix = self.get_confusion_matrix(targets)
        return np.linalg.inv(confusion_matrix)

    def get_mitigated_counts(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, int]:
        if targets is None:
            targets = self.data.keys()
        raw_counter = self.get_counts(targets)
        basis_labels = self.get_basis_labels(targets)
        # Ensure all basis labels are present in the raw counts
        raw = np.array([raw_counter.get(label, 0) for label in basis_labels])
        cm_inv = self.get_inverse_confusion_matrix(targets)
        mitigated = raw @ cm_inv
        mitigated_counts = {
            basis_label: int(mitigated[i]) for i, basis_label in enumerate(basis_labels)
        }
        return mitigated_counts

    def get_mitigated_probabilities(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, float]:
        if targets is None:
            targets = self.data.keys()
        basis_labels = self.get_basis_labels(targets)
        raw_probs = self.get_probabilities(targets)
        # Ensure the order of raw matches basis_labels
        raw = np.array([raw_probs.get(label, 0.0) for label in basis_labels])
        cm_inv = self.get_inverse_confusion_matrix(targets)
        mitigated = raw @ cm_inv
        mitigated_probabilities = {
            basis_label: mitigated[i] for i, basis_label in enumerate(basis_labels)
        }
        return mitigated_probabilities

    def plot(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        if self.mode == MeasureMode.SINGLE:
            data = {
                qubit: np.asarray(data.kerneled) for qubit, data in self.data.items()
            }
            return viz.scatter_iq_data(
                data=data,
                return_figure=return_figure,
                save_image=save_image,
            )
        elif self.mode == MeasureMode.AVG:
            figures = []
            for data in self.data.values():
                fig = data.plot(
                    return_figure=return_figure,
                    save_image=save_image,
                )
                figures.append(fig)
            if return_figure:
                return figures
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def plot_fft(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        figures = []
        for data in self.data.values():
            fig = data.plot_fft(
                return_figure=return_figure,
                save_image=save_image,
            )
            figures.append(fig)
        if return_figure:
            return figures

    def save(
        self,
        data_dir: Path | str | None = None,
    ) -> MeasurementRecord[MeasureResult]:
        return MeasurementRecord.create(data=self, data_dir=data_dir)

    # ------------------------------------------------------------------
    # Export (classified 0/1 data) API
    # ------------------------------------------------------------------
    def save_classified(
        self,
        path: str | Path,
        *,
        targets: Collection[str] | None = None,
        threshold: float | None = None,
        format: Literal["json", "csv", "npz"] = "json",
        include_memory: bool = True,
        compress: bool = True,
        overwrite: bool = False,
        metadata: dict | None = None,
    ) -> Path:
        """
        Save classified (0/1, bitstring) measurement data in a lightweight format.

        Parameters
        ----------
        path : str | Path
            Output file path (extension may be adjusted based on format/compress).
        targets : Collection[str], optional
            Subset of targets to export. If None, all targets used.
        threshold : float, optional
            Probability threshold for accepting a classified label. Shots whose
            max probability < threshold are dropped.
        format : {"json", "csv", "npz"}, default "json"
            Export format.
        include_memory : bool, default True
            Whether to include per-shot memory (bitstrings). When False only
            aggregated statistics (counts, probabilities, stdev) are stored.
        compress : bool, default True
            For json/csv -> gzip (.gz). For npz always compressed (numpy). If
            False, no gzip wrapper for json/csv.
        overwrite : bool, default False
            Overwrite existing file. If False and file exists -> ValueError.
        metadata : dict, optional
            Extra metadata to merge into auto-generated metadata.

        Returns
        -------
        Path
            Path to the created file.

        Notes
        -----
        - Only ``MeasureMode.SINGLE`` is supported (requires shot-wise data).
        - ``threshold`` filters ambiguous shots (based on soft probabilities).
        - Large shot counts: prefer ``format='npz'`` for efficiency.
        """
        if self.mode != MeasureMode.SINGLE:
            raise ValueError(
                "save_classified: mode='AVG' is not supported (requires SINGLE)."
            )
        if format not in {"json", "csv", "npz"}:
            raise ValueError(f"Unsupported format: {format}")

        if targets is None:
            targets = self.data.keys()
        else:
            # Validate targets
            for t in targets:
                if t not in self.data:
                    raise ValueError(f"Target '{t}' not found in MeasureResult.data")

        targets = list(targets)

        classified_matrix = self.get_classified_data(targets, threshold=threshold)
        n_shots_raw = self.data[targets[0]].length if len(targets) > 0 else 0
        memory_list = [
            "".join(map(str, row)) for row in classified_matrix if all(row >= 0)
        ]
        n_shots_kept = len(memory_list)

        counts_counter = Counter(memory_list)
        total = sum(counts_counter.values())
        probabilities = (
            {k: v / total for k, v in counts_counter.items()} if total else {}
        )
        # Wilson/normal approx same as earlier API (binomial per bitstring) -> use existing per-bitstring stdev formula
        standard_deviations = {
            k: float(np.sqrt(p * (1 - p) / total)) if total else 0.0
            for k, p in probabilities.items()
        }

        # Auto metadata
        created_at_iso = datetime.now(timezone.utc).isoformat()
        auto_meta: dict[str, object] = {
            "created_at": created_at_iso,
            "mode": self.mode.value,
            "targets": targets,
            "n_qubits": len(targets),
            "n_shots_raw": n_shots_raw,
            "n_shots_kept": n_shots_kept,
            "threshold": threshold,
        }
        # Attempt lightweight hash of config for reproducibility
        try:
            cfg_json = json.dumps(self.config, sort_keys=True, default=str)
            auto_meta["config_hash"] = hashlib.sha256(cfg_json.encode()).hexdigest()[
                :12
            ]
        except Exception:
            auto_meta["config_hash"] = None
        if metadata:
            auto_meta.update(metadata)

        path = Path(path)
        # If path is directory -> generate filename
        if path.exists() and path.is_dir():
            timestamp = (
                created_at_iso.replace(":", "").replace("-", "").replace(".", "")
            )
            stem = f"classified_{timestamp}"
            if format == "npz":
                path = path / f"{stem}.npz"
            elif format == "json":
                path = path / f"{stem}.json"
                if compress:
                    path = Path(str(path) + ".gz")
            else:  # csv
                path = path / f"{stem}.csv"
                if compress:
                    path = Path(str(path) + ".gz")
        else:
            # Adjust suffixes if user specified file base
            if format == "npz":
                if path.suffix != ".npz":
                    path = path.with_suffix(".npz")
            elif format == "json":
                if not path.name.endswith(".json") and not path.name.endswith(
                    ".json.gz"
                ):
                    path = path.with_suffix(".json")
                if compress and not str(path).endswith(".gz"):
                    path = Path(str(path) + ".gz")
            else:  # csv
                if not path.name.endswith(".csv") and not path.name.endswith(".csv.gz"):
                    path = path.with_suffix(".csv")
                if compress and not str(path).endswith(".gz"):
                    path = Path(str(path) + ".gz")

        if path.exists() and not overwrite:
            raise ValueError(f"File already exists: {path}")
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        if format == "npz":
            # Build dense uint8 matrix for memory (optional)
            if include_memory and n_shots_kept > 0:
                mem_array = np.array(
                    [[int(c) for c in bitstr] for bitstr in memory_list], dtype=np.uint8
                )
            else:
                mem_array = np.empty((0, len(targets)), dtype=np.uint8)
            meta_str = json.dumps(auto_meta, ensure_ascii=False)
            counts_str = json.dumps(counts_counter, ensure_ascii=False)
            probs_str = json.dumps(probabilities, ensure_ascii=False)
            stds_str = json.dumps(standard_deviations, ensure_ascii=False)
            # Store JSON strings as zero-d arrays of dtype=object to avoid typing issues
            np.savez_compressed(
                path,
                memory=mem_array,
                counts=np.array(counts_str, dtype=object),
                probabilities=np.array(probs_str, dtype=object),
                standard_deviations=np.array(stds_str, dtype=object),
                metadata=np.array(meta_str, dtype=object),
            )
        elif format == "json":
            obj = {
                "metadata": auto_meta,
                "counts": dict(counts_counter),
                "probabilities": probabilities,
                "standard_deviations": standard_deviations,
            }
            if include_memory:
                obj["memory"] = memory_list
            text = json.dumps(obj, ensure_ascii=False, indent=2)
            if compress:
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    f.write(text)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
        else:  # csv
            import csv as _csv

            if include_memory:
                # Each row: shot_index, bitstring
                if compress:
                    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
                        writer = _csv.writer(f)
                        writer.writerow(["shot", "bitstring"])
                        for idx, bitstr in enumerate(memory_list):
                            writer.writerow([idx, bitstr])
                else:
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        writer = _csv.writer(f)
                        writer.writerow(["shot", "bitstring"])
                        for idx, bitstr in enumerate(memory_list):
                            writer.writerow([idx, bitstr])
            else:
                if compress:
                    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
                        writer = _csv.writer(f)
                        writer.writerow(["bitstring", "count", "probability", "stddev"])
                        for k in counts_counter:
                            writer.writerow(
                                [
                                    k,
                                    counts_counter[k],
                                    probabilities.get(k, 0.0),
                                    standard_deviations.get(k, 0.0),
                                ]
                            )
                else:
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        writer = _csv.writer(f)
                        writer.writerow(["bitstring", "count", "probability", "stddev"])
                        for k in counts_counter:
                            writer.writerow(
                                [
                                    k,
                                    counts_counter[k],
                                    probabilities.get(k, 0.0),
                                    standard_deviations.get(k, 0.0),
                                ]
                            )
            # Write metadata sidecar
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(auto_meta, mf, ensure_ascii=False, indent=2)
        return path


@dataclass(frozen=True)
class MultipleMeasureResult:
    mode: MeasureMode
    data: dict[str, list[MeasureData]]
    config: dict

    def get_basis_indices(
        self,
        targets: Collection[str] | None = None,
    ) -> list[tuple[int, ...]]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        if targets is None:
            targets = self.data.keys()
        dimensions = [self.data[target][0].n_states for target in targets]
        return list(np.ndindex(*[dim for dim in dimensions]))

    def get_basis_labels(
        self,
        targets: Collection[str] | None = None,
    ) -> list[str]:
        basis_indices = self.get_basis_indices(targets)
        return ["".join(str(i) for i in basis) for basis in basis_indices]

    def get_classified_data(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        threshold: float | None = None,
    ) -> NDArray:
        if len(self.data) == 0:
            raise ValueError("No classification data available")

        if targets is None:
            target_tuples = [(target, -1) for target in self.data.keys()]
        else:
            target_tuples: list[tuple[str, int]] = []
            for target in targets:
                if isinstance(target, str):
                    target_tuples.append((target, -1))
                elif isinstance(target, tuple) and len(target) == 2:
                    target_tuples.append(target)
                else:
                    raise ValueError(f"Invalid target format: {target}")

        return np.column_stack(
            [
                self.data[target][idx].get_classified_data(threshold=threshold)
                for (target, idx) in target_tuples
            ]
        )

    def get_counts(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        threshold: float | None = None,
    ) -> Counter:
        classified_data = self.get_classified_data(targets, threshold=threshold)
        classified_labels = np.array(
            ["".join(map(str, row)) for row in classified_data]
        )
        return Counter(classified_labels)

    def get_probabilities(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        threshold: float | None = None,
    ) -> dict[str, float]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        counts = self.get_counts(targets, threshold=threshold)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {key: count / total for key, count in counts.items()}

    def get_standard_deviations(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
        *,
        threshold: float | None = None,
    ) -> dict[str, float]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        counts = self.get_counts(targets, threshold=threshold)
        probs = self.get_probabilities(targets, threshold=threshold)
        return {
            key: np.sqrt(prob * (1 - prob) / total)
            for key, prob, total in zip(
                counts.keys(),
                probs.values(),
                counts.values(),
            )
        }

    def get_classifier(self, target: str) -> StateClassifier:
        if target not in self.data:
            raise ValueError(f"Target {target} not found in data")
        classifier = self.data[target][0].classifier
        if classifier is None:
            raise ValueError(f"Classifier for target {target} is not set")
        return classifier

    def get_confusion_matrix(
        self,
        targets: Collection[str] | None = None,
    ) -> NDArray:
        if targets is None:
            targets = self.data.keys()
        confusion_matrices = [
            self.data[target][0].confusion_matrix for target in targets
        ]
        return reduce(np.kron, confusion_matrices)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str] | None = None,
    ) -> NDArray:
        if targets is None:
            targets = self.data.keys()
        confusion_matrix = self.get_confusion_matrix(targets)
        return np.linalg.inv(confusion_matrix)

    def get_mitigated_counts(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
    ) -> dict[str, int]:
        if targets is None:
            labels = list(self.data.keys())
        else:
            labels = []
            for target in targets:
                if isinstance(target, str):
                    labels.append(target)
                elif isinstance(target, tuple) and len(target) == 2:
                    labels.append(target[0])
                else:
                    raise ValueError(f"Invalid target format: {target}")

        basis_labels = self.get_basis_labels(labels)
        raw_counts = self.get_counts(targets)
        # Ensure the order of raw matches basis_labels
        raw = np.array([raw_counts.get(label, 0) for label in basis_labels])
        cm_inv = self.get_inverse_confusion_matrix(labels)
        mitigated = raw @ cm_inv
        basis_labels = self.get_basis_labels(labels)
        mitigated_counts = {
            basis_label: int(mitigated[i]) for i, basis_label in enumerate(basis_labels)
        }
        return mitigated_counts

    def get_mitigated_probabilities(
        self,
        targets: Collection[str | tuple[str, int]] | None = None,
    ) -> dict[str, float]:
        if targets is None:
            labels = list(self.data.keys())
        else:
            labels = []
            for target in targets:
                if isinstance(target, str):
                    labels.append(target)
                elif isinstance(target, tuple) and len(target) == 2:
                    labels.append(target[0])
                else:
                    raise ValueError(f"Invalid target format: {target}")

        basis_labels = self.get_basis_labels(labels)
        raw_probs = self.get_probabilities(targets)
        # Ensure the order of raw matches basis_labels
        raw = np.array([raw_probs.get(label, 0.0) for label in basis_labels])
        cm_inv = self.get_inverse_confusion_matrix(labels)
        mitigated = raw @ cm_inv
        mitigated_probabilities = {
            basis_label: mitigated[i] for i, basis_label in enumerate(basis_labels)
        }
        return mitigated_probabilities

    def plot(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        for qubit, data_list in self.data.items():
            figures = []
            for capture_index, data in enumerate(data_list):
                fig = data.plot(
                    title=f"{qubit} : data[{capture_index}]",
                    return_figure=return_figure,
                    save_image=save_image,
                )
                figures.append(fig)
            if return_figure:
                return figures
        return None

    def plot_fft(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        for qubit, data_list in self.data.items():
            figures = []
            for capture_index, data in enumerate(data_list):
                fig = data.plot_fft(
                    title=f"{qubit} : data[{capture_index}]",
                    return_figure=return_figure,
                    save_image=save_image,
                )
                figures.append(fig)
            if return_figure:
                return figures
        return None

    def save(
        self,
        data_dir: Path | str | None = None,
    ) -> MeasurementRecord[MultipleMeasureResult]:
        return MeasurementRecord.create(data=self, data_dir=data_dir)

    # ------------------------------------------------------------------
    # Export (classified 0/1 data) API for multiple captures
    # ------------------------------------------------------------------
    def save_classified(
        self,
        path: str | Path,
        *,
        capture_indices: dict[str, int] | None = None,
        targets: Collection[str] | None = None,
        threshold: float | None = None,
        format: Literal["json", "csv", "npz"] = "json",
        include_memory: bool = True,
        compress: bool = True,
        overwrite: bool = False,
        metadata: dict | None = None,
    ) -> Path:
        """
        Save classified data selecting one capture per target.

        This is analogous to :meth:`MeasureResult.save_classified` but
        allows specifying which capture index to export for each target.
        If ``capture_indices`` is None, index 0 is used for all targets.
        """
        if format not in {"json", "csv", "npz"}:
            raise ValueError(f"Unsupported format: {format}")
        if targets is None:
            targets = self.data.keys()
        else:
            for t in targets:
                if t not in self.data:
                    raise ValueError(f"Target '{t}' not in MultipleMeasureResult.data")
        targets = list(targets)
        if capture_indices is None:
            capture_indices = {t: 0 for t in targets}
        # Validate capture indices
        for t, idx in capture_indices.items():
            if t not in self.data:
                raise ValueError(f"Target '{t}' not found")
            if idx < 0 or idx >= len(self.data[t]):
                raise ValueError(f"capture index {idx} out of range for target {t}")

        # Build a synthetic MeasureResult for reuse
        selected = {t: self.data[t][capture_indices[t]] for t in targets}
        synthetic = MeasureResult(mode=self.mode, data=selected, config=self.config)
        return synthetic.save_classified(
            path=path,
            targets=targets,
            threshold=threshold,
            format=format,
            include_memory=include_memory,
            compress=compress,
            overwrite=overwrite,
            metadata=metadata,
        )

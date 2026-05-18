"""Contributed repeated coherence measurement helper function."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, NamedTuple

import numpy as np

from qubex.experiment import Experiment
from qubex.experiment.models.result import Result


class _ModeSpec(NamedTuple):
    method_name: str
    value_attribute: str
    metric_name: str


_MODE_SPECS: dict[str, _ModeSpec] = {
    "t1": _ModeSpec(
        method_name="t1_experiment",
        value_attribute="t1",
        metric_name="t1",
    ),
    "t2_echo": _ModeSpec(
        method_name="t2_experiment",
        value_attribute="t2",
        metric_name="t2_echo",
    ),
    "ramsey": _ModeSpec(
        method_name="ramsey_experiment",
        value_attribute="t2",
        metric_name="t2_star",
    ),
}

_MODE_ALIASES: dict[str, str] = {
    "t1": "t1",
    "t_1": "t1",
    "t2": "t2_echo",
    "t_2": "t2_echo",
    "t2_echo": "t2_echo",
    "t2echo": "t2_echo",
    "echo": "t2_echo",
    "ramsey": "ramsey",
    "t2_star": "ramsey",
    "t2star": "ramsey",
    "t2*": "ramsey",
}


def _normalize_modes(modes: Collection[str] | None) -> list[str]:
    if modes is None:
        return list(_MODE_SPECS)

    normalized_modes: list[str] = []
    for mode in modes:
        normalized = _MODE_ALIASES.get(mode.strip().lower())
        if normalized is None:
            raise ValueError(
                f"Unknown coherence measurement mode `{mode}`. "
                f"Expected one of {sorted(_MODE_SPECS)}."
            )
        if normalized in normalized_modes:
            raise ValueError(f"Duplicate coherence measurement mode `{mode}`.")
        normalized_modes.append(normalized)

    if len(normalized_modes) == 0:
        raise ValueError("At least one coherence measurement mode must be specified.")

    return normalized_modes


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        return list(exp.ctx.qubit_labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _validate_options(
    *,
    t1_options: Mapping[str, Any] | None,
    t2_options: Mapping[str, Any] | None,
    ramsey_options: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    return {
        "t1": t1_options or {},
        "t2_echo": t2_options or {},
        "ramsey": ramsey_options or {},
    }


def _build_measurement_kwargs(
    *,
    targets: Collection[str] | str | None,
    n_shots: int | None,
    shot_interval: float | None,
    plot: bool | None,
    save_image: bool | None,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "targets": targets,
        "n_shots": n_shots,
        "shot_interval": shot_interval,
        "plot": plot,
        "save_image": save_image,
    }
    kwargs.update(options)
    return kwargs


def _ensure_target_storage(
    *,
    values: dict[str, dict[str, list[float]]],
    failed_runs: dict[str, dict[str, list[int]]],
    mode: str,
    target: str,
    run_index: int,
) -> None:
    if target in values[mode]:
        return

    values[mode][target] = [np.nan] * run_index
    failed_runs[mode][target] = []


def _extract_float_value(data: object, attribute: str) -> float:
    value = getattr(data, attribute, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _build_statistics(
    values: dict[str, dict[str, list[float]]],
    metric_names: Mapping[str, str],
    n_runs: int,
) -> dict[str, dict[str, dict[str, float | int | str]]]:
    statistics: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for mode, target_values in values.items():
        statistics[mode] = {}
        for target, series in target_values.items():
            value_array = np.asarray(series, dtype=float)
            finite_values = value_array[np.isfinite(value_array)]
            count = int(finite_values.size)
            mean = float(np.mean(finite_values)) if count > 0 else float("nan")
            std = float(np.std(finite_values, ddof=1)) if count > 1 else float("nan")
            statistics[mode][target] = {
                "metric": metric_names[mode],
                "mean": mean,
                "std": std,
                "count": count,
                "n_runs": n_runs,
            }
    return statistics


def repeated_coherence_measurement(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    n_runs: int,
    modes: Collection[str] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = False,
    save_image: bool | None = False,
    t1_options: Mapping[str, Any] | None = None,
    t2_options: Mapping[str, Any] | None = None,
    ramsey_options: Mapping[str, Any] | None = None,
) -> Result:
    """
    Run coherence measurements repeatedly and summarize their variability.

    Parameters
    ----------
    exp
        Experiment instance used to run the underlying coherence measurements.
    targets
        Target qubits to characterize. When omitted, all experiment qubits are
        used.
    n_runs
        Number of repeated measurements to run for each selected mode.
    modes
        Measurement modes to execute. Accepted canonical values are ``"t1"``,
        ``"t2_echo"``, and ``"ramsey"``. ``"t2"`` aliases ``"t2_echo"`` and
        ``"t2_star"`` aliases ``"ramsey"``.
    n_shots
        Common number of shots passed to each underlying measurement unless a
        mode-specific options mapping overrides it.
    shot_interval
        Common shot interval passed to each underlying measurement unless a
        mode-specific options mapping overrides it.
    plot
        Common plotting flag for underlying measurements.
    save_image
        Common image saving flag for underlying measurements.
    t1_options
        Additional keyword arguments passed to ``exp.t1_experiment``.
    t2_options
        Additional keyword arguments passed to ``exp.t2_experiment``.
    ramsey_options
        Additional keyword arguments passed to ``exp.ramsey_experiment``.

    Returns
    -------
    Result
        Mapping-style result containing raw per-run results, extracted values,
        summary statistics, and failed run indexes.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")

    normalized_modes = _normalize_modes(modes)
    options_by_mode = _validate_options(
        t1_options=t1_options,
        t2_options=t2_options,
        ramsey_options=ramsey_options,
    )

    raw_results: dict[str, list[object]] = {mode: [] for mode in normalized_modes}
    values: dict[str, dict[str, list[float]]] = {mode: {} for mode in normalized_modes}
    failed_runs: dict[str, dict[str, list[int]]] = {
        mode: {} for mode in normalized_modes
    }
    target_sets: dict[str, list[str]] = {}
    metric_names = {mode: _MODE_SPECS[mode].metric_name for mode in normalized_modes}

    measurement_kwargs: dict[str, dict[str, Any]] = {}
    for mode in normalized_modes:
        kwargs = _build_measurement_kwargs(
            targets=targets,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
            save_image=save_image,
            options=options_by_mode[mode],
        )
        measurement_kwargs[mode] = kwargs
        target_sets[mode] = _normalize_targets(exp, kwargs.get("targets"))
        for target in target_sets[mode]:
            _ensure_target_storage(
                values=values,
                failed_runs=failed_runs,
                mode=mode,
                target=target,
                run_index=0,
            )

    for run_index in range(n_runs):
        for mode in normalized_modes:
            spec = _MODE_SPECS[mode]
            method = getattr(exp, spec.method_name)
            result = method(**measurement_kwargs[mode])
            raw_results[mode].append(result)

            result_data = getattr(result, "data", {})
            measured_targets = set(result_data)
            expected_targets = set(target_sets[mode])
            targets_to_record = [*target_sets[mode]]
            for target in measured_targets - expected_targets:
                target_sets[mode].append(target)
                targets_to_record.append(target)
                _ensure_target_storage(
                    values=values,
                    failed_runs=failed_runs,
                    mode=mode,
                    target=target,
                    run_index=run_index,
                )

            for target in targets_to_record:
                target_data = result_data.get(target)
                value = _extract_float_value(target_data, spec.value_attribute)
                values[mode][target].append(value)
                if not np.isfinite(value):
                    failed_runs[mode][target].append(run_index)

    return Result(
        data={
            "targets": target_sets,
            "n_runs": n_runs,
            "modes": normalized_modes,
            "metrics": metric_names,
            "raw_results": raw_results,
            "values": values,
            "statistics": _build_statistics(values, metric_names, n_runs),
            "failed_runs": failed_runs,
        },
    )

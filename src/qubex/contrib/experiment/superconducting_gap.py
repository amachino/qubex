"""Contributed superconducting-gap estimation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import yaml

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.system.lattice_graph import NODE_SIZE, TEXT_SIZE, LatticeGraph
from qubex.visualization import save_figure

_ELECTRON_CHARGE_C = 1.602176634e-19
_DEFAULT_DESCRIPTION = (
    "The minimum energy required to break an electron pair in a superconductor"
)
_DEFAULT_UNIT = "ueV"
_DEFAULT_RESISTANCE_DESCRIPTION = "Resistance charge after annealing"
_DEFAULT_RESISTANCE_UNIT = "ohms"


def _infer_all_qubit_labels(exp: Experiment) -> list[str]:
    chip_qubit_count_text = exp.chip_id.split("Q", maxsplit=1)[0]
    if not chip_qubit_count_text.isdigit():
        return list(exp.ctx.qubit_labels)

    chip_qubit_count = int(chip_qubit_count_text)
    label_width = max((len(label) - 1 for label in exp.ctx.qubit_labels), default=0)
    if label_width <= 0:
        label_width = len(str(chip_qubit_count - 1))
    return [f"Q{index:0{label_width}d}" for index in range(chip_qubit_count)]


def _load_resistance_map_from_file(path: Path) -> dict[str, float | None]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("resistance yaml must contain a mapping payload.")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise TypeError("resistance yaml must contain a `data` mapping.")

    resistance_map: dict[str, float | None] = {}
    for key, value in data.items():
        key_text = str(key)
        if value is None:
            resistance_map[key_text] = None
            continue
        resistance_map[key_text] = float(value)
    return resistance_map


def _load_resistance_payload(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("resistance yaml must contain a mapping payload.")

    data_obj = payload.get("data")
    if not isinstance(data_obj, dict):
        raise TypeError("resistance yaml must contain a `data` mapping.")

    meta_obj = payload.get("meta")
    if meta_obj is None:
        meta_obj = {}
    if not isinstance(meta_obj, dict):
        raise TypeError("resistance yaml `meta` must be a mapping.")

    parsed_data: dict[str, float | None] = {}
    for key, value in data_obj.items():
        if not isinstance(key, str):
            raise TypeError("resistance yaml keys must be qubit-label strings.")
        if value is None:
            parsed_data[key] = None
            continue
        parsed_data[key] = float(value)

    description = meta_obj.get("description", _DEFAULT_RESISTANCE_DESCRIPTION)
    unit = meta_obj.get("unit", _DEFAULT_RESISTANCE_UNIT)
    if not isinstance(description, str):
        raise TypeError("resistance yaml `meta.description` must be a string.")
    if not isinstance(unit, str):
        raise TypeError("resistance yaml `meta.unit` must be a string.")

    return {
        "meta": {
            "description": description,
            "unit": unit,
        },
        "data": parsed_data,
    }


def _load_superconducting_gap_payload(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("superconducting gap yaml must contain a mapping payload.")

    data_obj = payload.get("data")
    if not isinstance(data_obj, dict):
        raise TypeError("superconducting gap yaml must contain a `data` mapping.")

    meta_obj = payload.get("meta")
    if meta_obj is None:
        meta_obj = {}
    if not isinstance(meta_obj, dict):
        raise TypeError("superconducting gap yaml `meta` must be a mapping.")

    parsed_data: dict[str, float | None] = {}
    for key, value in data_obj.items():
        key_text = str(key)
        if value is None:
            parsed_data[key_text] = None
            continue
        parsed_data[key_text] = float(value)

    description = meta_obj.get("description", _DEFAULT_DESCRIPTION)
    unit = meta_obj.get("unit", _DEFAULT_UNIT)
    if not isinstance(description, str):
        raise TypeError("superconducting gap yaml `meta.description` must be a string.")
    if not isinstance(unit, str):
        raise TypeError("superconducting gap yaml `meta.unit` must be a string.")

    return {
        "meta": {
            "description": description,
            "unit": unit,
        },
        "data": parsed_data,
    }


def _resolve_params_path(exp: Experiment) -> Path | None:
    config_loader = getattr(exp, "config_loader", None)
    if config_loader is None:
        return None
    params_path = getattr(config_loader, "params_path", None)
    if params_path is None:
        return None
    return Path(params_path)


def _resolve_qubit_params_by_label(exp: Experiment) -> dict[str, object]:
    experiment_system = getattr(exp, "experiment_system", None)
    system_qubits = (
        getattr(experiment_system, "qubits", None)
        if experiment_system is not None
        else None
    )
    if system_qubits is not None:
        resolved: dict[str, object] = {}
        for qubit in system_qubits:
            label = getattr(qubit, "label", None)
            if isinstance(label, str):
                resolved[label] = qubit
        if len(resolved) > 0:
            return resolved

    ctx = getattr(exp, "ctx", None)
    ctx_qubits = getattr(ctx, "qubits", None) if ctx is not None else None
    if isinstance(ctx_qubits, dict):
        return {str(label): qubit for label, qubit in ctx_qubits.items()}
    return {}


def _infer_available_qubit_labels(exp: Experiment, *, all_labels: list[str]) -> set[str]:
    experiment_system = getattr(exp, "experiment_system", None)
    ge_targets = (
        getattr(experiment_system, "ge_targets", None)
        if experiment_system is not None
        else None
    )
    if ge_targets is not None:
        available = {
            str(qubit_label)
            for target in ge_targets
            for qubit_label in [getattr(target, "qubit", None)]
            if qubit_label is not None
        }
        if len(available) > 0:
            return available

    ctx = getattr(exp, "ctx", None)
    ctx_qubit_labels = getattr(ctx, "qubit_labels", None) if ctx is not None else None
    if ctx_qubit_labels is None:
        return set(all_labels)
    return {str(label) for label in ctx_qubit_labels}


def _normalize_qubit_keyed_values(
    raw_values: Mapping[str, float | None],
    *,
    all_labels: list[str],
) -> dict[str, float | None]:
    normalized: dict[str, float | None] = {}
    label_set = set(all_labels)
    index_to_label = dict(enumerate(all_labels))

    for key, value in raw_values.items():
        if key in label_set:
            normalized[key] = value
            continue

        index: int | None = None
        if key.isdigit():
            index = int(key)
        elif key.startswith("Q") and key[1:].isdigit():
            index = int(key[1:])

        if index is not None and index in index_to_label:
            normalized[index_to_label[index]] = value

    return normalized


def dump_superconducting_gap_yaml(
    superconducting_gap: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """
    Serialize a superconducting-gap payload to YAML.

    Parameters
    ----------
    superconducting_gap
        Superconducting-gap payload returned by `get_superconducting_gap`.
    output_path
        Destination path for the YAML file.
    """
    path = Path(output_path)
    serialized = yaml.safe_dump(
        dict(superconducting_gap),
        sort_keys=False,
        allow_unicode=False,
    )
    path.write_text(serialized, encoding="utf-8")


def _build_superconducting_gap_figure(
    *,
    all_labels: list[str],
    values_by_label: Mapping[str, float | None],
    title: str,
    unit_label: str,
) -> go.Figure:
    graph = LatticeGraph(len(all_labels))
    ordered_labels = all_labels
    plot_values = [
        np.nan if values_by_label.get(label) is None else values_by_label[label]
        for label in ordered_labels
    ]
    plot_texts: list[str] = []
    plot_hovertexts: list[str] = []
    for label in ordered_labels:
        value = values_by_label.get(label)
        if value is None:
            plot_texts.append("N/A")
            plot_hovertexts.append(f"{label}: N/A")
            continue
        plot_texts.append(f"{label}<br>{value:.1f}<br>{unit_label}")
        plot_hovertexts.append(f"{label}: {value:.3f} {unit_label}")

    fig = viz.make_figure()
    fig.add_trace(
        go.Heatmap(
            z=graph.create_data_matrix(plot_values),
            text=graph.create_data_matrix(plot_texts),
            colorscale="Viridis",
            hoverinfo="text",
            hovertext=graph.create_data_matrix(plot_hovertexts),
            texttemplate="%{text}",
            showscale=False,
            textfont=dict(
                family="monospace",
                size=TEXT_SIZE,
                weight="bold",
            ),
        )
    )

    width = 3 * NODE_SIZE * graph.n_qubit_cols
    height = 3 * NODE_SIZE * graph.n_qubit_rows
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(b=30, l=30, r=30, t=60),
        xaxis=dict(
            ticks="",
            linewidth=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            ticks="",
            autorange="reversed",
            linewidth=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        width=width,
        height=height,
    )
    return fig


def _resolve_default_resistance_path(params_path: Path | None) -> Path:
    if params_path is None:
        raise FileNotFoundError(
            "No `resistance_charge` source was provided and params path "
            "is unavailable from `exp.config_loader.params_path`."
        )
    default_path = params_path / "resistance_charge.yaml"
    if not default_path.exists():
        raise FileNotFoundError(
            "No `resistance_charge` source was provided, and default file "
            f"`{default_path}` was not found."
        )
    return default_path


def _load_resistance_map_from_source(
    *,
    params_path: Path | None,
    resistance_charge: Mapping[str | int, float | None] | str | Path | None,
) -> dict[str, float | None]:
    if resistance_charge is None:
        return _load_resistance_map_from_file(
            _resolve_default_resistance_path(params_path)
        )
    if isinstance(resistance_charge, (str, Path)):
        resistance_path = Path(resistance_charge)
        if not resistance_path.exists():
            raise FileNotFoundError(
                f"`resistance_charge` file was not found: {resistance_path}"
            )
        return _load_resistance_map_from_file(resistance_path)
    return {
        str(key): (None if value is None else float(value))
        for key, value in resistance_charge.items()
    }


def _load_resistance_payload_from_source(
    *,
    params_path: Path | None,
    resistance_charge: Mapping[str | int, float | None] | str | Path | None,
) -> dict[str, object]:
    if resistance_charge is None:
        return _load_resistance_payload(_resolve_default_resistance_path(params_path))
    if isinstance(resistance_charge, (str, Path)):
        resistance_path = Path(resistance_charge)
        if not resistance_path.exists():
            raise FileNotFoundError(
                f"`resistance_charge` file was not found: {resistance_path}"
            )
        return _load_resistance_payload(resistance_path)
    return {
        "meta": {
            "description": _DEFAULT_RESISTANCE_DESCRIPTION,
            "unit": _DEFAULT_RESISTANCE_UNIT,
        },
        "data": {
            str(key): (None if value is None else float(value))
            for key, value in resistance_charge.items()
        },
    }


def _build_superconducting_gap_payload(
    data: Mapping[str, float | None],
) -> dict[str, object]:
    return {
        "meta": {
            "description": _DEFAULT_DESCRIPTION,
            "unit": _DEFAULT_UNIT,
        },
        "data": dict(data),
    }


def _compute_superconducting_gap_data(
    *,
    exp: Experiment,
    all_labels: list[str],
    resistance_map: Mapping[str, float | None],
) -> dict[str, float | None]:
    available_labels = _infer_available_qubit_labels(exp, all_labels=all_labels)
    qubit_params_by_label = _resolve_qubit_params_by_label(exp)

    data: dict[str, float | None] = {}
    for qubit_label in all_labels:
        qubit_param = qubit_params_by_label.get(qubit_label)
        if qubit_label not in available_labels or qubit_param is None:
            data[qubit_label] = None
            continue

        if qubit_label not in resistance_map:
            raise ValueError(f"`resistance_charge` is missing target `{qubit_label}`.")

        resistance_ohm = resistance_map[qubit_label]
        if resistance_ohm is None:
            data[qubit_label] = None
            continue
        if resistance_ohm <= 0:
            raise ValueError(
                f"`resistance_charge[{qubit_label}]` must be positive: {resistance_ohm}."
            )

        anharmonicity_ghz = abs(float(qubit_param.anharmonicity))
        if anharmonicity_ghz == 0:
            raise ValueError(
                f"Anharmonicity for `{qubit_label}` must not be zero to estimate gap."
            )

        frequency_ghz = float(qubit_param.frequency)
        data[qubit_label] = (
            1e15
            * _ELECTRON_CHARGE_C
            * resistance_ohm
            * (frequency_ghz + anharmonicity_ghz) ** 2
            / anharmonicity_ghz
        )

    return data


def _expand_resistance_data(
    *,
    exp: Experiment,
    all_labels: list[str],
    raw_data: Mapping[str, float | None],
) -> dict[str, float | None]:
    available_labels = _infer_available_qubit_labels(exp, all_labels=all_labels)
    normalized_data = _normalize_qubit_keyed_values(raw_data, all_labels=all_labels)

    full_data: dict[str, float | None] = {}
    for qubit_label in all_labels:
        if qubit_label not in available_labels:
            full_data[qubit_label] = None
            continue
        value = normalized_data.get(qubit_label)
        full_data[qubit_label] = None if value is None else float(value)
    return full_data


def _save_figure_if_requested(
    *,
    figure: go.Figure,
    save_image: bool,
    image_name: str,
) -> None:
    if not save_image:
        return
    figure_width = int(figure.layout.width) if figure.layout.width is not None else None
    figure_height = (
        int(figure.layout.height) if figure.layout.height is not None else None
    )
    save_figure(
        figure,
        name=image_name,
        format="png",
        width=figure_width,
        height=figure_height,
        scale=3,
    )


def _build_optional_heatmap(
    *,
    plot: bool,
    save_image: bool,
    image_name: str,
    all_labels: list[str],
    values_by_label: Mapping[str, float | None],
    title: str,
    unit_label: str,
) -> go.Figure | None:
    if not plot:
        return None
    figure = _build_superconducting_gap_figure(
        all_labels=all_labels,
        values_by_label=values_by_label,
        title=title,
        unit_label=unit_label,
    )
    figure.show()
    _save_figure_if_requested(
        figure=figure,
        save_image=save_image,
        image_name=image_name,
    )
    return figure


def get_superconducting_gap(
    exp: Experiment,
    resistance_charge: Mapping[str | int, float | None] | str | Path | None = None,
    *,
    plot: bool | None = None,
    save_image: bool | None = None,
    image_name: str | None = None,
    output_path: str | Path | None = None,
) -> Result:
    """
    Estimate superconducting-gap parameters from qubit and resistance data.

    Parameters
    ----------
    exp
        Experiment instance that provides per-qubit frequency and anharmonicity.
    resistance_charge
        Resistance values in ohms keyed by qubit label, a yaml path with
        `{"meta": ..., "data": {...}}` payload, or `None`.
        When `None`, this function loads
        `<exp.config_loader.params_path>/resistance_charge.yaml`.
    plot
        Whether to render a chip-layout heatmap similarly to
        `qubex.experiment.experiment_tool.print_chip_info`.
    save_image
        Whether to save the rendered heatmap image.
    image_name
        Image name used when `save_image=True`.
    output_path
        Optional yaml path. When provided, the result is serialized to this path.

    Returns
    -------
    Result
        Result container with payload keys `meta` and `data`, and optional
        `figure` when plotting is requested.

    Raises
    ------
    FileNotFoundError
        If `resistance_charge` is not provided and the default params file is
        missing, or when a provided path does not exist.
    ValueError
        If required resistance data is missing, resistance is non-positive,
        or anharmonicity is zero for an available qubit.

    Notes
    -----
    This function is cache-first.
    If `<params_path>/superconducting_gap.yaml` exists, it is loaded and used.
    Otherwise the value is computed from resistance and then saved to that path.

    This helper uses the same empirical formula as the superconducting-gap
    notebook prototype:
    `gap = 1e15 * e * R_n * (f + |alpha|)^2 / |alpha|`.
    """
    if plot is None:
        plot = False
    if save_image is None:
        save_image = False
    if image_name is None:
        image_name = "superconducting_gap"

    params_path = _resolve_params_path(exp)
    inferred_all_labels = _infer_all_qubit_labels(exp)

    default_gap_path = (
        params_path / "superconducting_gap.yaml" if params_path is not None else None
    )
    if default_gap_path is not None and default_gap_path.exists():
        superconducting_gap = _load_superconducting_gap_payload(default_gap_path)
        loaded_data = superconducting_gap["data"]
        if not isinstance(loaded_data, dict):
            raise TypeError("superconducting gap payload `data` must be a mapping.")
        data = loaded_data
    else:
        raw_resistance_map = _load_resistance_map_from_source(
            params_path=params_path,
            resistance_charge=resistance_charge,
        )
        resistance_map = _normalize_qubit_keyed_values(
            raw_resistance_map,
            all_labels=inferred_all_labels,
        )
        data = _compute_superconducting_gap_data(
            exp=exp,
            all_labels=inferred_all_labels,
            resistance_map=resistance_map,
        )
        superconducting_gap = _build_superconducting_gap_payload(data)

        if default_gap_path is not None:
            default_gap_path.parent.mkdir(parents=True, exist_ok=True)
            dump_superconducting_gap_yaml(
                superconducting_gap=superconducting_gap,
                output_path=default_gap_path,
            )

    if output_path is not None:
        dump_superconducting_gap_yaml(
            superconducting_gap=superconducting_gap,
            output_path=output_path,
        )

    figure = _build_optional_heatmap(
        plot=plot,
        save_image=save_image,
        image_name=image_name,
        all_labels=inferred_all_labels,
        values_by_label=data,
        title="Superconducting gap (ueV)",
        unit_label="ueV",
    )

    return Result(data=superconducting_gap, figure=figure)


def get_resistance_charge(
    exp: Experiment,
    resistance_charge: Mapping[str | int, float | None] | str | Path | None = None,
    *,
    plot: bool | None = None,
    save_image: bool | None = None,
    image_name: str | None = None,
    output_path: str | Path | None = None,
) -> Result:
    """
    Load resistance-charge data and optionally plot it on chip layout.

    Parameters
    ----------
    exp
        Experiment instance used to resolve chip labels and params path.
    resistance_charge
        Resistance values in ohms keyed by qubit label, a yaml path with
        `{"meta": ..., "data": {...}}` payload, or `None`.
        When `None`, this function loads
        `<exp.config_loader.params_path>/resistance_charge.yaml`.
    plot
        Whether to render a chip-layout heatmap.
    save_image
        Whether to save the rendered heatmap image.
    image_name
        Image name used when `save_image=True`.
    output_path
        Optional yaml path. When provided, the payload is serialized to this path.

    Returns
    -------
    Result
        Result container with payload keys `meta` and `data`, and optional
        `figure` when plotting is requested.

    Raises
    ------
    FileNotFoundError
        If no resistance source is provided and the default params file is
        missing, or when a provided path does not exist.
    """
    if plot is None:
        plot = False
    if save_image is None:
        save_image = False
    if image_name is None:
        image_name = "resistance_charge"

    params_path = _resolve_params_path(exp)
    inferred_all_labels = _infer_all_qubit_labels(exp)

    resistance_payload = _load_resistance_payload_from_source(
        params_path=params_path,
        resistance_charge=resistance_charge,
    )

    payload_data_obj = resistance_payload["data"]
    if not isinstance(payload_data_obj, dict):
        raise TypeError("resistance payload `data` must be a mapping.")

    full_data = _expand_resistance_data(
        exp=exp,
        all_labels=inferred_all_labels,
        raw_data=payload_data_obj,  # type: ignore[arg-type]
    )

    result_payload: dict[str, object] = {
        "meta": resistance_payload["meta"],
        "data": full_data,
    }

    if output_path is not None:
        dump_superconducting_gap_yaml(
            superconducting_gap=result_payload,
            output_path=output_path,
        )

    figure = _build_optional_heatmap(
        plot=plot,
        save_image=save_image,
        image_name=image_name,
        all_labels=inferred_all_labels,
        values_by_label=full_data,
        title="Resistance charge (ohms)",
        unit_label="ohms",
    )

    return Result(data=result_payload, figure=figure)


__all__ = [
    "dump_superconducting_gap_yaml",
    "get_resistance_charge",
    "get_superconducting_gap",
]

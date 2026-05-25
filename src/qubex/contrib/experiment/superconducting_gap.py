"""Contributed superconducting-gap estimation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

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
_KeyT = TypeVar("_KeyT")


def _infer_all_qubit_labels(exp: Experiment) -> list[str]:
    chip_qubit_count_text = exp.chip_id.split("Q", maxsplit=1)[0]
    if not chip_qubit_count_text.isdigit():
        return list(exp.ctx.qubit_labels)

    chip_qubit_count = int(chip_qubit_count_text)
    label_width = max(2, len(str(chip_qubit_count - 1)))
    return [f"Q{index:0{label_width}d}" for index in range(chip_qubit_count)]


def _resolve_params_path(exp: Experiment) -> Path | None:
    config_loader = getattr(exp, "config_loader", None)
    if config_loader is None:
        return None
    params_path = getattr(config_loader, "params_path", None)
    if params_path is None:
        return None
    return Path(params_path)


def _resolve_resistance_source(
    *,
    params_path: Path | None,
    resistance_charge: Mapping[str | int, float | None] | str | Path | None,
) -> tuple[Path | None, Mapping[str | int, float | None] | None]:
    if resistance_charge is None:
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
        return default_path, None

    if isinstance(resistance_charge, (str, Path)):
        resistance_path = Path(resistance_charge)
        if not resistance_path.exists():
            raise FileNotFoundError(
                f"`resistance_charge` file was not found: {resistance_path}"
            )
        return resistance_path, None

    return None, resistance_charge


def _normalize_qubit_values(
    raw_values: Mapping[_KeyT, float | None],
    *,
    all_labels: list[str],
) -> dict[str, float | None]:
    label_set = set(all_labels)

    normalized: dict[str, float | None] = {}
    for raw_key, value in raw_values.items():
        key = str(raw_key)
        if key not in label_set:
            continue
        normalized[key] = None if value is None else float(value)

    return normalized


def _load_resistance_map_from_file(path: Path) -> dict[str, float | None]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("resistance yaml must contain a mapping payload.")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise TypeError("resistance yaml must contain a `data` mapping.")

    return {
        str(key): (None if value is None else float(value))
        for key, value in data.items()
    }


def _build_resistance_payload_from_values(
    values: Mapping[str | int, float | None],
) -> dict[str, object]:
    return {
        "meta": {
            "description": _DEFAULT_RESISTANCE_DESCRIPTION,
            "unit": _DEFAULT_RESISTANCE_UNIT,
        },
        "data": {
            str(key): (None if value is None else float(value))
            for key, value in values.items()
        },
    }


def _load_resistance_payload(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("resistance yaml must contain a mapping payload.")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise TypeError("resistance yaml must contain a `data` mapping.")

    meta = payload.get("meta")
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise TypeError("resistance yaml `meta` must be a mapping.")

    description = meta.get("description", _DEFAULT_RESISTANCE_DESCRIPTION)
    unit = meta.get("unit", _DEFAULT_RESISTANCE_UNIT)
    if not isinstance(description, str):
        raise TypeError("resistance yaml `meta.description` must be a string.")
    if not isinstance(unit, str):
        raise TypeError("resistance yaml `meta.unit` must be a string.")

    return {
        "meta": {"description": description, "unit": unit},
        "data": {
            str(key): (None if value is None else float(value))
            for key, value in data.items()
        },
    }


def _load_superconducting_gap_payload(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("superconducting gap yaml must contain a mapping payload.")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise TypeError("superconducting gap yaml must contain a `data` mapping.")

    meta = payload.get("meta")
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise TypeError("superconducting gap yaml `meta` must be a mapping.")

    description = meta.get("description", _DEFAULT_DESCRIPTION)
    unit = meta.get("unit", _DEFAULT_UNIT)
    if not isinstance(description, str):
        raise TypeError("superconducting gap yaml `meta.description` must be a string.")
    if not isinstance(unit, str):
        raise TypeError("superconducting gap yaml `meta.unit` must be a string.")

    return {
        "meta": {"description": description, "unit": unit},
        "data": {
            str(key): (None if value is None else float(value))
            for key, value in data.items()
        },
    }


def dump_superconducting_gap_yaml(
    superconducting_gap: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """Serialize a superconducting-gap payload to YAML."""
    path = Path(output_path)
    path.write_text(
        yaml.safe_dump(dict(superconducting_gap), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _build_superconducting_gap_figure(
    *,
    all_labels: list[str],
    values_by_label: Mapping[str, float | None],
    title: str,
    unit_label: str,
) -> go.Figure:
    graph = LatticeGraph(len(all_labels))
    plot_values = [
        np.nan if values_by_label.get(label) is None else values_by_label[label]
        for label in all_labels
    ]

    plot_texts: list[str] = []
    plot_hovertexts: list[str] = []
    for label in all_labels:
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
            textfont=dict(family="monospace", size=TEXT_SIZE, weight="bold"),
        )
    )

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
        width=3 * NODE_SIZE * graph.n_qubit_cols,
        height=3 * NODE_SIZE * graph.n_qubit_rows,
    )
    return fig


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

    if save_image:
        figure_width = (
            int(figure.layout.width) if figure.layout.width is not None else None
        )
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
    """Estimate superconducting-gap parameters from qubit and resistance data."""
    if plot is None:
        plot = False
    if save_image is None:
        save_image = False
    if image_name is None:
        image_name = "superconducting_gap"

    params_path = _resolve_params_path(exp)
    all_labels = _infer_all_qubit_labels(exp)

    default_gap_path = params_path / "superconducting_gap.yaml" if params_path else None
    if default_gap_path is not None and default_gap_path.exists():
        superconducting_gap = _load_superconducting_gap_payload(default_gap_path)
        loaded_data = superconducting_gap["data"]
        if not isinstance(loaded_data, dict):
            raise TypeError("superconducting gap payload `data` must be a mapping.")
        data = loaded_data
    else:
        resistance_path, resistance_mapping = _resolve_resistance_source(
            params_path=params_path,
            resistance_charge=resistance_charge,
        )
        raw_resistance = (
            _load_resistance_map_from_file(resistance_path)
            if resistance_path is not None
            else _normalize_qubit_values(
                resistance_mapping or {}, all_labels=all_labels
            )
        )
        resistance_map = _normalize_qubit_values(raw_resistance, all_labels=all_labels)

        experiment_system = getattr(exp, "experiment_system", None)
        system_qubits = (
            getattr(experiment_system, "qubits", None)
            if experiment_system is not None
            else None
        )
        if system_qubits is not None:
            qubit_params = {
                str(label): qubit
                for qubit in system_qubits
                for label in [getattr(qubit, "label", None)]
                if label is not None
            }
        else:
            ctx = getattr(exp, "ctx", None)
            ctx_qubits = getattr(ctx, "qubits", None) if ctx is not None else None
            qubit_params = (
                {str(label): qubit for label, qubit in ctx_qubits.items()}
                if isinstance(ctx_qubits, dict)
                else {}
            )

        data: dict[str, float | None] = {}
        for qubit_label in all_labels:
            resistance_ohm = resistance_map.get(qubit_label)
            qubit_param = qubit_params.get(qubit_label)
            if resistance_ohm is None or qubit_param is None:
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

        superconducting_gap = {
            "meta": {"description": _DEFAULT_DESCRIPTION, "unit": _DEFAULT_UNIT},
            "data": data,
        }

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
        all_labels=all_labels,
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
    """Load resistance-charge data and optionally plot it on chip layout."""
    if plot is None:
        plot = False
    if save_image is None:
        save_image = False
    if image_name is None:
        image_name = "resistance_charge"

    params_path = _resolve_params_path(exp)
    all_labels = _infer_all_qubit_labels(exp)

    resistance_path, resistance_mapping = _resolve_resistance_source(
        params_path=params_path,
        resistance_charge=resistance_charge,
    )

    resistance_payload = (
        _load_resistance_payload(resistance_path)
        if resistance_path is not None
        else _build_resistance_payload_from_values(resistance_mapping or {})
    )

    payload_data_obj = resistance_payload["data"]
    if not isinstance(payload_data_obj, dict):
        raise TypeError("resistance payload `data` must be a mapping.")

    normalized_data = _normalize_qubit_values(payload_data_obj, all_labels=all_labels)
    full_data = {label: normalized_data.get(label) for label in all_labels}

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
        all_labels=all_labels,
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

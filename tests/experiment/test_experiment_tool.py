"""Tests for experiment_tool helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import plotly.graph_objects as go
import pytest

from qubex.backend.quel1.quel1_backend_constants import (
    DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT,
)
from qubex.experiment import experiment_tool
from qubex.experiment.models.result import Result
from qubex.system.control_system import PortType
from qubex.visualization.style import FONT_FAMILY


@dataclass(frozen=True)
class FakePort:
    """Port stub for box info tests."""

    number: int
    type: PortType


@dataclass(frozen=True)
class FakeBox:
    """Box stub for box info tests."""

    id: str
    ports: tuple[FakePort, ...]


class FakeExperimentSystem:
    """Experiment-system stub with box lookup."""

    def __init__(self, boxes: list[FakeBox]) -> None:
        self.boxes = boxes
        self._boxes = {box.id: box for box in boxes}

    def get_box(self, box_id: str) -> FakeBox:
        """Return a fake box."""
        return self._boxes[box_id]


class FakeBackendController:
    """Backend-controller stub with dump recording."""

    def __init__(self) -> None:
        self.dumped_box_ids: list[str] = []

    def dump_box(self, box_id: str) -> dict:
        """Return a fake dump for the requested box."""
        self.dumped_box_ids.append(box_id)
        return {
            "ports": {
                1: {
                    "direction": "out",
                    "sideband": "L",
                    "lo_freq": 10_000_000_000,
                    "cnco_freq": 1_500,
                    "vatt": 2_048,
                    "fullscale_current": 40_527,
                    "channels": {0: {"fnco_freq": 100}},
                }
            }
        }


class FakeQuel1Box:
    """Quel1Box stub recording reconnect calls."""

    def __init__(self) -> None:
        self.background_noise_thresholds: list[float] = []

    def reconnect(self, *, background_noise_threshold: float) -> None:
        """Record reconnect threshold."""
        self.background_noise_thresholds.append(background_noise_threshold)


class FakeBackendControllerWithGetBox(FakeBackendController):
    """Backend-controller stub with box lookup."""

    def __init__(self, boxes: dict[str, FakeQuel1Box]) -> None:
        super().__init__()
        self._boxes = boxes

    def get_box(self, box_id: str) -> FakeQuel1Box:
        """Return a fake Quel1 box."""
        return self._boxes[box_id]


class FakeBackendControllerWithSkew(FakeBackendController):
    """Backend-controller stub with skew measurement support."""

    def __init__(self, figure: go.FigureWidget) -> None:
        super().__init__()
        self.figure = figure
        self.run_skew_measurement_calls: list[dict[str, object]] = []
        self.update_skew_calls: list[dict[str, object]] = []

    def run_skew_measurement(
        self,
        *,
        skew_yaml_path: Path,
        box_yaml_path: Path,
        clockmaster_ip: str,
        box_names: list[str],
        estimate: bool,
    ) -> tuple[dict[str, str], go.FigureWidget]:
        """Return fake skew results and record render parameters."""
        self.run_skew_measurement_calls.append(
            {
                "skew_yaml_path": skew_yaml_path,
                "box_yaml_path": box_yaml_path,
                "clockmaster_ip": clockmaster_ip,
                "box_names": box_names,
                "estimate": estimate,
            }
        )
        return {"status": "ok"}, self.figure

    def update_skew(
        self,
        *,
        file_path: Path,
        wait: int,
        box_names: list[str] | None,
        backup: bool,
    ) -> dict[str, object]:
        """Record skew-update requests and return a fake result."""
        self.update_skew_calls.append(
            {
                "file_path": file_path,
                "wait": wait,
                "box_names": box_names,
                "backup": backup,
            }
        )
        return {
            "file_path": file_path,
            "backup_path": file_path.with_suffix(".yaml.bak") if backup else None,
            "box_names": box_names if box_names is not None else [],
            "wait": wait,
        }


@dataclass
class FakeSystemManager:
    """System-manager stub for experiment_tool tests."""

    experiment_system: object
    backend_controller: FakeBackendController
    config_loader: object | None = None


@dataclass(frozen=True)
class FakeConfigLoader:
    """Config-loader stub for experiment_tool tests."""

    system_id: str
    config_path: Path
    params_path: Path


def test_print_box_info_fetch_uses_dump_box(monkeypatch) -> None:
    """Given fetch mode, when printing box info, then dump_box is used."""
    fake_manager = FakeSystemManager(
        experiment_system=FakeExperimentSystem(
            [FakeBox(id="A", ports=(FakePort(number=1, type=PortType.CTRL),))]
        ),
        backend_controller=FakeBackendController(),
    )
    monkeypatch.setattr(experiment_tool, "system_manager", fake_manager)
    monkeypatch.setattr(experiment_tool.console, "print", lambda *_: None)

    experiment_tool.print_box_info("A", fetch=True)

    assert fake_manager.backend_controller.dumped_box_ids == ["A"]


def test_get_quel1_box_reconnects_box_with_default_threshold(monkeypatch) -> None:
    """Given a backend box, when requesting Quel1 box, then reconnect is called once."""
    box = FakeQuel1Box()
    fake_manager = FakeSystemManager(
        experiment_system=FakeExperimentSystem([]),
        backend_controller=FakeBackendControllerWithGetBox({"U15A": box}),
    )
    monkeypatch.setattr(experiment_tool, "system_manager", fake_manager)

    returned_box = experiment_tool.get_quel1_box("U15A")

    assert returned_box is box
    assert box.background_noise_thresholds == [
        DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT
    ]


def test_print_chip_info_uses_active_system_id_for_chip_summary(
    monkeypatch, tmp_path
) -> None:
    """Given a loaded shared-chip system, chip summary should use active system_id."""

    class FakeChipInspector:
        """Inspector stub recording initialization arguments."""

        init_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs: object) -> None:
            self.__class__.init_kwargs = kwargs

        def execute(self, params: dict | None = None) -> object:
            """Return a fake inspection summary."""
            del params
            return self

        def draw(
            self,
            *,
            draw_individual_results: bool = True,
            save_image: bool = False,
        ) -> None:
            """Accept draw requests."""
            del draw_individual_results, save_image

    fake_chip = type("FakeChip", (), {"id": "144Q-LF", "n_qubits": 144})()
    fake_loader = FakeConfigLoader(
        system_id="144Q-LF-Q3",
        config_path=tmp_path / "config",
        params_path=tmp_path / "params",
    )
    fake_manager = FakeSystemManager(
        experiment_system=type(
            "FakeExperimentSystemWithChip", (), {"chip": fake_chip}
        )(),
        backend_controller=FakeBackendController(),
        config_loader=fake_loader,
    )
    monkeypatch.setattr(experiment_tool, "system_manager", fake_manager)
    monkeypatch.setattr(experiment_tool, "ChipInspector", FakeChipInspector)

    experiment_tool.print_chip_info("chip_summary")

    assert FakeChipInspector.init_kwargs == {
        "chip_id": "144Q-LF",
        "system_id": "144Q-LF-Q3",
        "config_dir": tmp_path / "config",
        "params_dir": tmp_path / "params",
    }


def test_print_chip_info_maps_t2_star_ef_ratio(monkeypatch) -> None:
    """Given T2* and T2* EF data, when printing T2* EF info, then ratio is mapped."""

    class FakeParamLoader:
        """Config-loader stub returning coherence maps."""

        def __init__(self) -> None:
            self.requests: list[str] = []

        def load_param_data(self, name: str) -> dict[str, float | None]:
            """Return fake parameter data by name."""
            self.requests.append(name)
            if name == "t2_star":
                return {
                    "Q0": 10_000.0,
                    "Q1": 0.0,
                    "Q2": None,
                    "Q3": float("nan"),
                }
            if name == "t2_star_ef":
                return {
                    "Q0": 5_000.0,
                    "Q1": 1_000.0,
                    "Q2": 4_000.0,
                    "Q3": 2_000.0,
                }
            return {}

    plot_calls: list[dict[str, object]] = []

    class FakeLatticeGraph:
        """LatticeGraph stub recording plot arguments."""

        def __init__(self, n_qubits: int) -> None:
            self.n_qubits = n_qubits

        def plot_lattice_data(self, **kwargs: object) -> None:
            """Record one lattice plot call."""
            plot_calls.append(kwargs)

    fake_loader = FakeParamLoader()
    fake_chip = type("FakeChip", (), {"id": "TESTCHIP", "n_qubits": 4})()
    fake_manager = FakeSystemManager(
        experiment_system=type(
            "FakeExperimentSystemWithChip", (), {"chip": fake_chip}
        )(),
        backend_controller=FakeBackendController(),
        config_loader=fake_loader,
    )
    monkeypatch.setattr(experiment_tool, "system_manager", fake_manager)
    monkeypatch.setattr(experiment_tool, "LatticeGraph", FakeLatticeGraph)

    experiment_tool.print_chip_info("t2_star_ef", save_image=True)

    assert fake_loader.requests == ["t2_star", "t2_star_ef"]
    assert len(plot_calls) == 1
    call = plot_calls[0]
    values = cast(list[float], call["values"])
    texts = cast(list[str], call["texts"])
    hovertexts = cast(list[str], call["hovertexts"])
    assert call["title"] == "T2* EF / T2* (%)"
    assert values[0] == 50.0
    assert math.isnan(values[1])
    assert math.isnan(values[2])
    assert math.isnan(values[3])
    assert texts[0] == "Q0<br>50.00<br>%"
    assert texts[1:] == ["N/A", "N/A", "N/A"]
    assert hovertexts[0] == "Q0: 50.000%"
    assert call["save_image"] is True
    assert call["image_name"] == "t2_star_ef"


def test_check_skew_renders_figure_widget_via_plotly_figure(
    monkeypatch, tmp_path
) -> None:
    """Given a FigureWidget skew plot, check_skew should return Result and preserve legacy fig access."""
    figure_widget = go.FigureWidget()
    figure_widget.add_scatter(y=[1, 2, 3])
    backend = FakeBackendControllerWithSkew(figure_widget)
    fake_manager = FakeSystemManager(
        experiment_system=SimpleNamespace(
            control_system=SimpleNamespace(clock_master_address="192.0.2.10")
        ),
        backend_controller=backend,
        config_loader=SimpleNamespace(config_path=tmp_path),
    )
    (tmp_path / "skew.yaml").write_text("reference_port: REF-1\n", encoding="utf-8")
    (tmp_path / "box.yaml").write_text("boxes: {}\n", encoding="utf-8")

    shown: dict[str, object] = {}

    def _fail_widget_show(self, *args, **kwargs) -> None:
        raise AssertionError("FigureWidget.show should not be used by check_skew")

    def _record_figure_show(self, *args, **kwargs) -> None:
        shown["figure"] = self
        shown["kwargs"] = kwargs

    monkeypatch.setattr(experiment_tool, "system_manager", fake_manager)
    monkeypatch.setattr(experiment_tool.Confirm, "ask", lambda *args, **kwargs: True)
    monkeypatch.setattr(go.FigureWidget, "show", _fail_widget_show, raising=False)
    monkeypatch.setattr(go.Figure, "show", _record_figure_show, raising=False)

    result = experiment_tool.check_skew(["BOX1"], config_dir=str(tmp_path))

    rendered_figure = shown["figure"]
    assert isinstance(rendered_figure, go.Figure)
    assert isinstance(result, Result)
    assert result == {"skew": {"status": "ok"}, "fig": figure_widget}
    assert result.figure is figure_widget
    with pytest.warns(DeprecationWarning, match="figure` attribute"):
        assert result["fig"] is figure_widget
    rendered_layout = rendered_figure.to_dict()["layout"]
    assert rendered_layout["title"]["text"] == "Skew : BOX1 (Ref. REF)"
    assert rendered_layout["width"] == 800
    assert rendered_layout["template"]["layout"]["font"]["family"] == FONT_FAMILY
    assert len(backend.run_skew_measurement_calls) == 1
    call = backend.run_skew_measurement_calls[0]
    assert call["skew_yaml_path"] == tmp_path / "skew.yaml"
    assert call["box_yaml_path"] == tmp_path / "box.yaml"
    assert call["clockmaster_ip"] == "192.0.2.10"
    assert set(cast(list[str], call["box_names"])) == {"BOX1", "REF"}
    assert call["estimate"] is True


def test_update_skew_uses_backend_and_returns_result(monkeypatch, tmp_path) -> None:
    """Given a skew file, when update_skew is called, then backend update is delegated and wrapped in Result."""
    backend = FakeBackendControllerWithSkew(go.FigureWidget())
    fake_manager = FakeSystemManager(
        experiment_system=SimpleNamespace(),
        backend_controller=backend,
        config_loader=SimpleNamespace(config_path=tmp_path),
    )
    (tmp_path / "skew.yaml").write_text(
        """
box_setting:
  BOX1:
    slot: 0
    wait: 0
time_to_start: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(experiment_tool, "system_manager", fake_manager)

    result = experiment_tool.update_skew(
        250,
        ["BOX1"],
        config_dir=str(tmp_path),
        backup=True,
    )

    assert isinstance(result, Result)
    assert result["wait"] == 250
    assert result["file_path"] == tmp_path / "skew.yaml"
    assert result["backup_path"] == tmp_path / "skew.yaml.bak"
    assert result["box_names"] == ["BOX1"]
    assert backend.update_skew_calls == [
        {
            "file_path": tmp_path / "skew.yaml",
            "wait": 250,
            "box_names": ["BOX1"],
            "backup": True,
        }
    ]

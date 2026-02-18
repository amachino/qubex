"""Tests for SystemManager backend settings collection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from qubex.backend.control_system import PortType
from qubex.backend.system_manager import BackendSettings, SystemManager


@dataclass(frozen=True)
class FakePort:
    """Port stub for backend settings tests."""

    number: int
    type: PortType
    channels: tuple[object, ...] = ()


@dataclass(frozen=True)
class FakeBox:
    """Box stub for backend settings tests."""

    id: str
    ports: tuple[FakePort, ...]


class FakeBackendController:
    """Backend controller stub for backend settings tests."""

    def __init__(self, configs: dict[str, dict]) -> None:
        self._configs = configs
        self._box_config_cache: dict[str, dict] = {}

    def dump_box(self, box_id: str) -> dict:
        """Return a predefined box configuration."""
        return self._configs.get(box_id, {})

    def get_box_config_cache(self) -> dict[str, dict]:
        """Return a copy of the current box-cache snapshot."""
        return deepcopy(self._box_config_cache)

    def replace_box_config_cache(self, box_configs: dict[str, dict]) -> None:
        """Replace the full box-cache snapshot."""
        self._box_config_cache = deepcopy(box_configs)

    def update_box_config_cache(self, box_configs: dict[str, dict]) -> None:
        """Update per-box entries in the box cache."""
        for box_id, box_config in box_configs.items():
            self._box_config_cache[box_id] = deepcopy(box_config)

    @property
    def hash(self) -> int:
        """Return a stable hash for controller state."""
        return 0


class FakeExperimentSystem:
    """Experiment system stub with box lookup."""

    def __init__(self, boxes: list[FakeBox]) -> None:
        self._boxes = {box.id: box for box in boxes}

    def get_box(self, box_id: str) -> FakeBox:
        """Return a fake box."""
        return self._boxes[box_id]

    @property
    def hash(self) -> int:
        """Return a stable hash for experiment system state."""
        return 0


class FakeControlSystem:
    """Control-system stub that records set_port_params calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def set_port_params(self, **kwargs) -> None:
        """Record a set_port_params call."""
        self.calls.append(kwargs)

    @property
    def hash(self) -> int:
        """Return a hash derived from recorded calls."""
        frozen = tuple(
            (
                call["box_id"],
                call["port_number"],
                call["sideband"],
                call["lo_freq"],
                call["cnco_freq"],
                tuple(call["fnco_freqs"]),
                call["fullscale_current"],
            )
            for call in self.calls
        )
        return hash(frozen)


class FakeExperimentSystemForBackendSettings:
    """Experiment-system stub with mutable control-system parameters."""

    def __init__(self, boxes: list[FakeBox]) -> None:
        self.control_system = FakeControlSystem()
        self.boxes = boxes
        self._boxes = {box.id: box for box in boxes}

    def get_box(self, box_id: str) -> FakeBox:
        """Return a fake box."""
        return self._boxes[box_id]

    @property
    def hash(self) -> int:
        """Return a hash reflecting control-system changes."""
        return self.control_system.hash


def test_backend_settings_hash_is_order_independent() -> None:
    """Given same nested content, hash is identical regardless of key insertion order."""
    settings_a = BackendSettings(
        {
            "A": {"ports": {1: {"v": 1}, 2: {"v": 2}}},
            "B": {"ports": {3: {"v": 3}}},
        }
    )
    settings_b = BackendSettings(
        {
            "B": {"ports": {3: {"v": 3}}},
            "A": {"ports": {2: {"v": 2}, 1: {"v": 1}}},
        }
    )

    assert settings_a == settings_b
    assert settings_a.hash == settings_b.hash


@pytest.mark.parametrize("parallel", [True, False])
def test_fetch_backend_settings_from_hardware_collects_ports(
    monkeypatch: pytest.MonkeyPatch,
    parallel: bool,
) -> None:
    """Given boxes, when fetching and syncing, then raw dump data is propagated."""
    # Arrange
    box_a = FakeBox(
        id="A",
        ports=(
            FakePort(number=1, type=PortType.CTRL),
            FakePort(number=2, type=PortType.MNTR_OUT),
            FakePort(number=3, type=PortType.READ_IN),
        ),
    )
    box_b = FakeBox(
        id="B",
        ports=(FakePort(number=4, type=PortType.READ_OUT),),
    )
    configs = {
        "A": {"ports": {1: {"mode": "ctrl"}, 3: {"mode": "read"}}},
        "B": {"ports": {4: {"mode": "read"}}},
    }
    manager = SystemManager.shared()
    backend_controller = FakeBackendController(configs)
    monkeypatch.setattr(manager, "_backend_controller", backend_controller)
    monkeypatch.setattr(
        manager, "_experiment_system", FakeExperimentSystem([box_a, box_b])
    )
    monkeypatch.setattr(manager, "_backend_settings", {"stale": {"ports": {}}})

    # Act
    fetched = manager._fetch_backend_settings_from_hardware(  # noqa: SLF001
        ["A", "B"],
        parallel=parallel,
    )
    manager._backend_settings = fetched  # noqa: SLF001
    manager._sync_backend_settings_to_device_controller()  # noqa: SLF001

    # Assert
    assert fetched == {
        "A": {"ports": {1: {"mode": "ctrl"}, 3: {"mode": "read"}}},
        "B": {"ports": {4: {"mode": "read"}}},
    }
    assert manager._backend_settings == fetched  # noqa: SLF001
    assert backend_controller.get_box_config_cache() == configs


def test_sync_backend_settings_to_experiment_system_updates_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given backend settings, when applying them, then the same experiment system is updated."""
    manager = SystemManager.shared()
    box = FakeBox(
        id="A",
        ports=(
            FakePort(number=1, type=PortType.READ_OUT),
            FakePort(number=2, type=PortType.READ_IN),
        ),
    )
    experiment_system = FakeExperimentSystemForBackendSettings([box])
    monkeypatch.setattr(manager, "_experiment_system", experiment_system)
    monkeypatch.setattr(manager, "_backend_controller", SimpleNamespace(hash=1))
    backend_settings = {
        "A": {
            "ports": {
                1: {
                    "direction": "out",
                    "sideband": "L",
                    "lo_freq": 10_000_000_000,
                    "cnco_freq": 1_500,
                    "fullscale_current": 40_527,
                    "channels": {
                        0: {"fnco_freq": 100},
                        1: {"fnco_freq": 200},
                    },
                },
                2: {
                    "direction": "in",
                    "lo_freq": 8_000_000_000,
                    "cnco_freq": 2_500,
                    "runits": {
                        0: {"fnco_freq": 300},
                    },
                },
            }
        }
    }
    monkeypatch.setattr(manager, "_backend_settings", backend_settings)

    original_id = id(manager.experiment_system)
    manager._sync_backend_settings_to_experiment_system()  # noqa: SLF001

    assert id(manager.experiment_system) == original_id
    assert len(experiment_system.control_system.calls) == 2
    assert experiment_system.control_system.calls[0] == {
        "box_id": "A",
        "port_number": 1,
        "sideband": "L",
        "lo_freq": 10_000_000_000,
        "cnco_freq": 1_500,
        "fnco_freqs": [100, 200],
        "fullscale_current": 40_527,
    }
    assert experiment_system.control_system.calls[1] == {
        "box_id": "A",
        "port_number": 2,
        "sideband": None,
        "lo_freq": 8_000_000_000,
        "cnco_freq": 2_500,
        "fnco_freqs": [300],
        "fullscale_current": None,
    }


def test_sync_backend_settings_to_experiment_system_skips_fnco_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given mismatched fnco count, when syncing, then that port is skipped."""
    manager = SystemManager.shared()
    box = FakeBox(
        id="A",
        ports=(FakePort(number=1, type=PortType.READ_OUT, channels=(object(),)),),
    )
    experiment_system = FakeExperimentSystemForBackendSettings([box])
    monkeypatch.setattr(manager, "_experiment_system", experiment_system)
    monkeypatch.setattr(manager, "_backend_controller", SimpleNamespace(hash=1))
    backend_settings = {
        "A": {
            "ports": {
                1: {
                    "direction": "out",
                    "sideband": "L",
                    "lo_freq": 10_000_000_000,
                    "cnco_freq": 1_500,
                    "fullscale_current": 40_527,
                    "channels": {},
                },
            }
        }
    }

    manager._sync_backend_settings_to_experiment_system(  # noqa: SLF001
        backend_settings=BackendSettings(backend_settings)
    )

    assert experiment_system.control_system.calls == []
    assert (
        "Skipping backend port sync for A:1 due to fnco count mismatch" in caplog.text
    )


def test_fetch_backend_settings_from_hardware_has_no_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given current cache, when fetching, then manager backend settings stay unchanged."""
    manager = SystemManager.shared()
    box = FakeBox(id="A", ports=(FakePort(number=1, type=PortType.CTRL),))
    monkeypatch.setattr(
        manager,
        "_backend_controller",
        FakeBackendController({"A": {"ports": {1: {"mode": "ctrl"}}}}),
    )
    monkeypatch.setattr(manager, "_experiment_system", FakeExperimentSystem([box]))
    monkeypatch.setattr(manager, "_backend_settings", {"stale": {"ports": {}}})

    fetched = manager._fetch_backend_settings_from_hardware(["A"])  # noqa: SLF001

    assert fetched == {"A": {"ports": {1: {"mode": "ctrl"}}}}
    assert manager._backend_settings == {"stale": {"ports": {}}}  # noqa: SLF001


def test_is_synced_has_no_side_effect_on_backend_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given mismatch, when checking sync, then backend settings stay unchanged."""
    manager = SystemManager.shared()
    box = FakeBox(id="A", ports=(FakePort(number=1, type=PortType.CTRL),))
    monkeypatch.setattr(
        manager,
        "_backend_controller",
        FakeBackendController({"A": {"ports": {1: {"mode": "ctrl"}}}}),
    )
    monkeypatch.setattr(manager, "_experiment_system", FakeExperimentSystem([box]))
    monkeypatch.setattr(manager, "_backend_settings", {"stale": {"ports": {}}})
    monkeypatch.setattr(manager, "_cached_state", manager.current_state)

    with pytest.warns(
        UserWarning,
        match="The current backend settings are different from the fetched backend settings.",
    ):
        result = manager.is_synced(box_ids=["A"])

    assert not result
    assert manager._backend_settings == {"stale": {"ports": {}}}  # noqa: SLF001


def test_pull_merges_partial_backend_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given partial pull, when applied, then non-target box settings are preserved."""
    manager = SystemManager.shared()
    box_a = FakeBox(id="A", ports=())
    monkeypatch.setattr(
        manager,
        "_experiment_system",
        FakeExperimentSystemForBackendSettings([box_a]),
    )
    backend_controller = FakeBackendController({"A": {"ports": {1: {"mode": "ctrl"}}}})
    backend_controller.replace_box_config_cache({"B": {"ports": {2: {"mode": "read"}}}})
    monkeypatch.setattr(manager, "_backend_controller", backend_controller)
    monkeypatch.setattr(
        manager,
        "_backend_settings",
        {
            "B": {"ports": {2: {"mode": "read"}}},
        },
    )

    manager.pull(["A"], parallel=False)

    assert manager.backend_settings == {
        "A": {"ports": {1: {"mode": "ctrl"}}},
        "B": {"ports": {2: {"mode": "read"}}},
    }
    assert backend_controller.get_box_config_cache() == {
        "A": {"ports": {1: {"mode": "ctrl"}}},
        "B": {"ports": {2: {"mode": "read"}}},
    }


def test_is_synced_compares_requested_box_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given extra cached boxes, when checking subset, then only requested boxes are compared."""
    manager = SystemManager.shared()
    box_a = FakeBox(id="A", ports=())
    monkeypatch.setattr(manager, "_experiment_system", FakeExperimentSystem([box_a]))
    monkeypatch.setattr(
        manager,
        "_backend_controller",
        FakeBackendController({"A": {"ports": {1: {"mode": "ctrl"}}}}),
    )
    monkeypatch.setattr(
        manager,
        "_backend_settings",
        {
            "A": {"ports": {1: {"mode": "ctrl"}}},
            "B": {"ports": {2: {"mode": "read"}}},
        },
    )
    monkeypatch.setattr(manager, "_cached_state", manager.current_state)

    result = manager.is_synced(box_ids=["A"])

    assert result


def test_sync_experiment_system_to_hardware_parallel_submits_per_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given parallel mode, when syncing hardware, then each box is submitted once."""
    manager = SystemManager.shared()
    boxes = [
        FakeBox(id="A", ports=()),
        FakeBox(id="B", ports=()),
    ]
    called: list[str] = []

    def _fake_sync_box(box: FakeBox) -> None:
        called.append(box.id)

    monkeypatch.setattr(manager, "_sync_box_to_hardware", _fake_sync_box)

    manager._sync_experiment_system_to_hardware(  # noqa: SLF001
        boxes=boxes,  # type: ignore[arg-type]
        parallel=True,
    )

    assert set(called) == {"A", "B"}
    assert len(called) == 2


def test_sync_experiment_system_to_hardware_sequential_calls_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given sequential mode, when syncing hardware, then boxes are processed in order."""
    manager = SystemManager.shared()
    boxes = [
        FakeBox(id="A", ports=()),
        FakeBox(id="B", ports=()),
    ]
    called: list[str] = []

    def _fake_sync_box(box: FakeBox) -> None:
        called.append(box.id)

    monkeypatch.setattr(manager, "_sync_box_to_hardware", _fake_sync_box)

    manager._sync_experiment_system_to_hardware(  # noqa: SLF001
        boxes=boxes,  # type: ignore[arg-type]
        parallel=False,
    )

    assert called == ["A", "B"]


def test_push_cancel_restores_backend_controller_cache_from_backend_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given canceled push, when confirmation is denied, then backend cache is restored from backend settings."""
    manager = SystemManager.shared()
    backend_settings = {"A": {"ports": {1: {"mode": "ctrl"}}}}
    backend_controller = FakeBackendController({})
    backend_controller.replace_box_config_cache({})
    monkeypatch.setattr(manager, "_backend_controller", backend_controller)
    monkeypatch.setattr(manager, "_backend_settings", backend_settings)

    box = SimpleNamespace(id="A", name="Alpha")
    monkeypatch.setattr(
        manager,
        "_experiment_system",
        SimpleNamespace(
            get_box=lambda box_id: box,
            hash=0,
        ),
    )
    monkeypatch.setattr("qubex.backend.system_manager.Confirm.ask", lambda _: False)

    called_sync_hardware = False

    def _should_not_run(**_: object) -> None:
        nonlocal called_sync_hardware
        called_sync_hardware = True

    monkeypatch.setattr(manager, "_sync_experiment_system_to_hardware", _should_not_run)

    manager.push(["A"], confirm=True)

    assert called_sync_hardware is False
    assert backend_controller.get_box_config_cache() == backend_settings


def test_create_backend_controller_supports_quel3() -> None:
    """Given Quel3 kind, when creating backend controller, then Quel3 measurement hint is exposed."""
    controller = SystemManager._create_backend_controller("quel3")  # noqa: SLF001

    assert getattr(controller, "MEASUREMENT_BACKEND_KIND", None) == "quel3"


def test_load_passes_backend_kind_to_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given load with backend kind, when loading, then selector is called before sync."""
    manager = SystemManager.shared()
    called: list[str] = []

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **_: object) -> None:
            pass

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    def _fake_set_backend_kind(kind: str) -> None:
        called.append(f"kind:{kind}")

    def _fake_sync() -> None:
        called.append("sync")

    monkeypatch.setattr("qubex.backend.system_manager.ConfigLoader", _FakeConfigLoader)
    monkeypatch.setattr(manager, "set_backend_kind", _fake_set_backend_kind)
    monkeypatch.setattr(
        manager, "_sync_experiment_system_to_backend_controller", _fake_sync
    )

    manager.load(
        chip_id="TEST",
        backend_kind="quel3",
        mock_mode=False,
    )

    assert called == ["kind:quel3", "sync"]


def test_load_prefers_wiring_v2_for_quel3_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quel3 backend and wiring.v2 file, when loading, then ConfigLoader uses wiring.v2.yaml."""
    manager = SystemManager.shared()
    captured: dict[str, object] = {}
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()
    (config_dir / "wiring.v2.yaml").write_text(
        "schema_version: 2\nchip_id: TEST\ncontrol: {}\nreadout: {}\n",
        encoding="utf-8",
    )

    class _FakeConfigLoader:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def load(self, **_: object) -> None:
            pass

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    monkeypatch.setattr("qubex.backend.system_manager.ConfigLoader", _FakeConfigLoader)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        backend_kind="quel3",
        mock_mode=True,
    )

    assert captured["wiring_file"] == "wiring.v2.yaml"


def test_load_falls_back_to_legacy_wiring_for_quel3_when_v2_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quel3 backend without wiring.v2 file, when loading, then ConfigLoader uses wiring.yaml."""
    manager = SystemManager.shared()
    captured: dict[str, object] = {}
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()

    class _FakeConfigLoader:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def load(self, **_: object) -> None:
            pass

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    monkeypatch.setattr("qubex.backend.system_manager.ConfigLoader", _FakeConfigLoader)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        backend_kind="quel3",
        mock_mode=True,
    )

    assert captured["wiring_file"] == "wiring.yaml"

"""Tests for SystemManager backend settings collection."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from qubex.backend.backend_controller import BACKEND_KIND_QUEL1, BACKEND_KIND_QUEL3
from qubex.backend.quel3 import Quel3BackendController
from qubex.system.control_system import PortType
from qubex.system.system_manager import BackendSettings, SystemManager


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
    """Given the same nested content, when hashing backend settings, then hash is independent of key insertion order."""
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
    manager._sync_backend_settings_to_backend_controller()  # noqa: SLF001

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
    """Given parallel mode, when syncing hardware, then SystemManager delegates to synchronizer."""
    manager = SystemManager.shared()
    boxes = [
        FakeBox(id="A", ports=()),
        FakeBox(id="B", ports=()),
    ]
    experiment_system = SimpleNamespace(hash=11)
    calls: list[tuple[object, list[str], bool | None, tuple[str, ...]]] = []

    class _SystemSyncManager:
        def sync_experiment_system_to_hardware(
            self,
            *,
            experiment_system: object,
            boxes: Sequence[FakeBox],
            parallel: bool | None = None,
            target_labels: Sequence[str] | None = None,
        ) -> None:
            calls.append(
                (
                    experiment_system,
                    [box.id for box in boxes],
                    parallel,
                    tuple(target_labels or ()),
                )
            )

    monkeypatch.setattr(manager, "_system_synchronizer", _SystemSyncManager())
    monkeypatch.setattr(manager, "_experiment_system", experiment_system)

    manager._sync_experiment_system_to_hardware(  # noqa: SLF001
        boxes=boxes,  # type: ignore[arg-type]
        parallel=True,
        target_labels=["Q00", "RQ00"],
    )

    assert calls == [(experiment_system, ["A", "B"], True, ("Q00", "RQ00"))]


def test_sync_experiment_system_to_hardware_sequential_calls_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given sequential mode, when syncing hardware, then SystemManager delegates to synchronizer."""
    manager = SystemManager.shared()
    boxes = [
        FakeBox(id="A", ports=()),
        FakeBox(id="B", ports=()),
    ]
    experiment_system = SimpleNamespace(hash=11)
    calls: list[tuple[object, list[str], bool | None, tuple[str, ...]]] = []

    class _SystemSyncManager:
        def sync_experiment_system_to_hardware(
            self,
            *,
            experiment_system: object,
            boxes: Sequence[FakeBox],
            parallel: bool | None = None,
            target_labels: Sequence[str] | None = None,
        ) -> None:
            calls.append(
                (
                    experiment_system,
                    [box.id for box in boxes],
                    parallel,
                    tuple(target_labels or ()),
                )
            )

    monkeypatch.setattr(manager, "_system_synchronizer", _SystemSyncManager())
    monkeypatch.setattr(manager, "_experiment_system", experiment_system)

    manager._sync_experiment_system_to_hardware(  # noqa: SLF001
        boxes=boxes,  # type: ignore[arg-type]
        parallel=False,
        target_labels=None,
    )

    assert calls == [(experiment_system, ["A", "B"], False, ())]


def test_sync_experiment_system_to_hardware_skips_without_system_synchronizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no system synchronizer, when syncing hardware, then SystemManager returns safely."""
    manager = SystemManager.shared()
    monkeypatch.setattr(manager, "_system_synchronizer", None)
    result = manager._sync_experiment_system_to_hardware(  # noqa: SLF001
        boxes=[FakeBox(id="A", ports=())],  # type: ignore[arg-type]
        parallel=True,
    )
    assert result is None


def test_sync_experiment_system_to_backend_controller_delegates_to_system_synchronizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given synchronizer, when loading config, then SystemManager forwards experiment-system sync."""
    manager = SystemManager.shared()
    delegated: list[object] = []
    experiment_system = SimpleNamespace(hash=11)

    class _SystemSyncManager:
        def sync_experiment_system_to_backend_controller(
            self,
            resolved_experiment_system: object,
        ) -> None:
            delegated.append(resolved_experiment_system)

    monkeypatch.setattr(manager, "_system_synchronizer", _SystemSyncManager())
    monkeypatch.setattr(manager, "_experiment_system", experiment_system)
    monkeypatch.setattr(manager, "_backend_settings", BackendSettings())

    manager._sync_experiment_system_to_backend_controller()  # noqa: SLF001

    assert delegated == [experiment_system]
    assert manager.cached_state == manager.current_state


def test_modified_backend_settings_initializes_shared_read_box_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a shared read/cap box, when applying modified backend settings, then AWG/CAP initialization runs once."""
    manager = SystemManager.shared()

    @dataclass
    class _FakePort:
        box_id: str
        number: int
        lo_freq: int | None
        cnco_freq: int

    @dataclass
    class _FakeChannel:
        port: _FakePort
        number: int
        fnco_freq: int

    class _FakeBackendController:
        def __init__(self) -> None:
            self._cache = {
                "BOX0": {
                    "ports": {
                        1: {
                            "lo_freq": 10,
                            "cnco_freq": 20,
                            "channels": {0: {"fnco_freq": 30}},
                            "runits": {0: {"fnco_freq": 30}},
                        }
                    }
                }
            }
            self.initialize_calls: list[str | list[str]] = []

        def get_box_config_cache(self) -> dict[str, dict]:
            return deepcopy(self._cache)

        def config_port(
            self,
            *,
            box_name: str,
            port: int,
            lo_freq_hz: int | None,
            cnco_freq_hz: int,
        ) -> None:
            self._cache[box_name]["ports"][port]["lo_freq"] = lo_freq_hz
            self._cache[box_name]["ports"][port]["cnco_freq"] = cnco_freq_hz

        def config_channel(
            self,
            *,
            box_name: str,
            port: int,
            channel: int,
            fnco_freq_hz: int,
        ) -> None:
            self._cache[box_name]["ports"][port]["channels"][channel]["fnco_freq"] = (
                fnco_freq_hz
            )

        def config_runit(
            self,
            *,
            box_name: str,
            port: int,
            runit: int,
            fnco_freq_hz: int,
        ) -> None:
            self._cache[box_name]["ports"][port]["runits"][runit]["fnco_freq"] = (
                fnco_freq_hz
            )

        def update_box_config_cache(self, box_configs: dict[str, dict]) -> None:
            for box_id, config in box_configs.items():
                self._cache[box_id] = deepcopy(config)

        def initialize_awg_and_capunits(self, box_names: str | list[str]) -> None:
            self.initialize_calls.append(box_names)

        def replace_box_config_cache(self, box_configs: dict[str, dict]) -> None:
            self._cache = deepcopy(box_configs)

    class _FakeExperimentSystem:
        def __init__(self) -> None:
            shared_port = _FakePort(box_id="BOX0", number=1, lo_freq=10, cnco_freq=20)
            self._target = SimpleNamespace(
                channel=_FakeChannel(port=shared_port, number=0, fnco_freq=30),
                is_read=True,
            )
            self._cap_target = SimpleNamespace(
                channel=_FakeChannel(port=shared_port, number=0, fnco_freq=30)
            )

        def get_target(self, _label: str) -> object:
            return self._target

        def get_cap_target(self, _label: str) -> object:
            return self._cap_target

        def update_port_params(self, *_args, **_kwargs) -> None:
            return

    backend_controller = _FakeBackendController()
    monkeypatch.setattr(manager, "_backend_controller", backend_controller)
    monkeypatch.setattr(manager, "_experiment_system", _FakeExperimentSystem())

    with manager.modified_backend_settings(
        "RQ00",
        lo_freq=100,
        cnco_freq=200,
        fnco_freq=300,
    ):
        pass

    assert len(backend_controller.initialize_calls) == 1
    call = backend_controller.initialize_calls[0]
    initialized = [call] if isinstance(call, str) else list(call)
    assert initialized == ["BOX0"]


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
    monkeypatch.setattr("qubex.system.system_manager.Confirm.ask", lambda _: False)

    called_sync_hardware = False

    def _should_not_run(**_: object) -> None:
        nonlocal called_sync_hardware
        called_sync_hardware = True

    monkeypatch.setattr(manager, "_sync_experiment_system_to_hardware", _should_not_run)

    manager.push(["A"], confirm=True)

    assert called_sync_hardware is False
    assert backend_controller.get_box_config_cache() == backend_settings


def test_push_does_not_reconfigure_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given user-modified system model, when pushing, then push does not reconfigure ports."""
    manager = SystemManager.shared()
    backend_controller = FakeBackendController({})
    monkeypatch.setattr(manager, "_backend_controller", backend_controller)
    monkeypatch.setattr(manager, "_backend_settings", {})

    box = SimpleNamespace(id="A", name="Alpha")

    class _ExperimentSystem:
        def get_box(self, box_id: str) -> object:
            if box_id != "A":
                raise KeyError(box_id)
            return box

        def configure_ports(self) -> None:
            raise AssertionError("push must not call configure_ports")

        @property
        def hash(self) -> int:
            return 0

    monkeypatch.setattr(manager, "_experiment_system", _ExperimentSystem())

    called_sync_hardware = False

    def _sync_hardware(**_: object) -> None:
        nonlocal called_sync_hardware
        called_sync_hardware = True

    monkeypatch.setattr(manager, "_sync_experiment_system_to_hardware", _sync_hardware)
    monkeypatch.setattr(
        manager,
        "_fetch_backend_settings_from_hardware",
        lambda **_: {"A": {"ports": {}}},
    )

    manager.push(["A"], confirm=False)

    assert called_sync_hardware is True


def test_push_without_cache_sync_still_applies_hardware_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given backend without cache sync support, when pushing, then hardware sync still runs."""
    manager = SystemManager.shared()
    box = SimpleNamespace(id="A", name="Alpha")
    monkeypatch.setattr(
        manager,
        "_experiment_system",
        SimpleNamespace(
            get_box=lambda box_id: box,
            hash=0,
        ),
    )
    monkeypatch.setattr(manager, "_supports_box_settings_cache_sync", lambda: False)

    called_sync_hardware = False

    def _sync_hardware(**_: object) -> None:
        nonlocal called_sync_hardware
        called_sync_hardware = True

    monkeypatch.setattr(manager, "_sync_experiment_system_to_hardware", _sync_hardware)

    manager.push(["A"], confirm=False)

    assert called_sync_hardware is True


def test_push_forwards_target_labels_to_hardware_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given target labels, when pushing, then SystemManager forwards them to hardware sync."""
    manager = SystemManager.shared()
    box = SimpleNamespace(id="A", name="Alpha")
    monkeypatch.setattr(
        manager,
        "_experiment_system",
        SimpleNamespace(
            get_box=lambda box_id: box,
            hash=0,
        ),
    )
    monkeypatch.setattr(manager, "_supports_box_settings_cache_sync", lambda: False)
    captured: dict[str, object] = {}

    def _sync_hardware(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(manager, "_sync_experiment_system_to_hardware", _sync_hardware)

    manager.push(["A"], confirm=False, target_labels=["Q00", "RQ00"])

    assert captured["target_labels"] == ["Q00", "RQ00"]


def test_create_backend_controller_supports_quel3() -> None:
    """Given Quel3 kind, when creating backend controller, then Quel3 controller is returned."""
    controller = SystemManager._create_backend_controller("quel3")  # noqa: SLF001

    assert isinstance(controller, Quel3BackendController)


def test_load_passes_backend_kind_to_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given load with backend kind, when loading, then selector is called before sync."""
    manager = SystemManager.shared()
    called: list[str] = []
    captured_load_kwargs: dict[str, object] = {}

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **kwargs: object) -> None:
            captured_load_kwargs.update(kwargs)

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL3

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    def _fake_set_backend_kind(kind: str) -> None:
        called.append(f"kind:{kind}")
        manager.__dict__["_backend_kind"] = kind

    def _fake_sync() -> None:
        called.append("sync")

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)
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
    assert captured_load_kwargs["backend_kind"] == BACKEND_KIND_QUEL3


def test_load_uses_config_loader_backend_kind_when_backend_kind_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given backend_kind omission, when loading, then ConfigLoader backend kind is used to select backend controller."""
    manager = SystemManager.shared()
    called: list[str] = []
    captured_load_kwargs: dict[str, object] = {}

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **kwargs: object) -> None:
            captured_load_kwargs.update(kwargs)

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL3

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    def _fake_set_backend_kind(kind: str) -> None:
        called.append(f"kind:{kind}")
        manager.__dict__["_backend_kind"] = kind

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)
    monkeypatch.setattr(manager, "set_backend_kind", _fake_set_backend_kind)

    manager.load(
        chip_id="TEST",
        mock_mode=True,
    )

    assert called == [f"kind:{BACKEND_KIND_QUEL3}"]
    assert captured_load_kwargs["backend_kind"] is None


def test_load_does_not_pass_wiring_file_to_config_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given manager load, when creating ConfigLoader, then wiring file is not forced by SystemManager."""
    manager = SystemManager.shared()
    captured_init_kwargs: dict[str, object] = {}
    captured_load_kwargs: dict[str, object] = {}
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()

    class _FakeConfigLoader:
        def __init__(self, **kwargs: object) -> None:
            captured_init_kwargs.update(kwargs)

        def load(self, **kwargs: object) -> None:
            captured_load_kwargs.update(kwargs)

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL3

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        backend_kind="quel3",
        mock_mode=True,
    )

    assert "wiring_file" not in captured_init_kwargs
    assert captured_load_kwargs["backend_kind"] == BACKEND_KIND_QUEL3


def test_load_forwards_core_paths_to_config_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given load parameters, when initializing ConfigLoader, then SystemManager forwards chip/config/params without wiring override."""
    manager = SystemManager.shared()
    captured_init_kwargs: dict[str, object] = {}
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()

    class _FakeConfigLoader:
        def __init__(self, **kwargs: object) -> None:
            captured_init_kwargs.update(kwargs)

        def load(self, **_: object) -> None:
            pass

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL3

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        backend_kind="quel3",
        mock_mode=True,
    )

    assert captured_init_kwargs["chip_id"] == "TEST"
    assert captured_init_kwargs["config_dir"] == config_dir
    assert captured_init_kwargs["params_dir"] == params_dir
    assert "wiring_file" not in captured_init_kwargs


def test_load_forwards_system_id_to_config_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given system_id input, when loading, then SystemManager forwards it to ConfigLoader."""
    manager = SystemManager.shared()
    captured_init_kwargs: dict[str, object] = {}
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()

    class _FakeConfigLoader:
        def __init__(self, **kwargs: object) -> None:
            captured_init_kwargs.update(kwargs)

        def load(self, **_: object) -> None:
            pass

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL3

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)

    manager.load(
        system_id="SYS-A",
        config_dir=config_dir,
        params_dir=params_dir,
        backend_kind="quel3",
        mock_mode=True,
    )

    assert captured_init_kwargs["system_id"] == "SYS-A"
    assert captured_init_kwargs["chip_id"] is None


def test_load_uses_provided_backend_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given an injected backend controller, load should keep that controller instance."""
    manager = SystemManager.shared()
    injected_controller = Quel3BackendController(
        client_mode="standalone",
        standalone_unit_label="quel3-02-a01",
    )

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **_: object) -> None:
            pass

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL3

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)

    manager.load(
        chip_id="TEST",
        backend_controller=injected_controller,
        mock_mode=True,
    )

    assert manager.backend_controller is injected_controller


def test_load_defaults_to_quel1_when_system_backend_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given missing system backend, when load is called without backend_kind, then backend defaults to quel1."""
    manager = SystemManager.shared()
    selected: list[str] = []
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()
    (config_dir / "chip.yaml").write_text(
        'TEST:\n  name: "Test Chip"\n  n_qubits: 4\n',
        encoding="utf-8",
    )

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **_: object) -> None:
            pass

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL1

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    def _fake_set_backend_kind(kind: str) -> None:
        selected.append(kind)
        manager.__dict__["_backend_kind"] = kind

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)
    monkeypatch.setattr(manager, "set_backend_kind", _fake_set_backend_kind)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        mock_mode=True,
    )

    assert selected == ["quel1"]


def test_load_resolves_backend_kind_from_system_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given system backend in system.yaml, when load is called without backend_kind, then backend kind is resolved from system.yaml."""
    manager = SystemManager.shared()
    selected: list[str] = []
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()
    (config_dir / "chip.yaml").write_text(
        'TEST:\n  name: "Test Chip"\n  n_qubits: 4\n  backend: unknown\n',
        encoding="utf-8",
    )
    (config_dir / "system.yaml").write_text(
        "schema_version: 1\nchip_id: TEST\nbackend: quel3\n",
        encoding="utf-8",
    )

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **_: object) -> None:
            pass

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL3

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    def _fake_set_backend_kind(kind: str) -> None:
        selected.append(kind)
        manager.__dict__["_backend_kind"] = kind

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)
    monkeypatch.setattr(manager, "set_backend_kind", _fake_set_backend_kind)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        mock_mode=True,
    )

    assert selected == ["quel3"]


def test_load_explicit_backend_kind_overrides_default_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given explicit backend_kind, when load is called, then explicit value overrides default resolution."""
    manager = SystemManager.shared()
    captured_load_kwargs: dict[str, object] = {}
    selected: list[str] = []
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()
    (config_dir / "chip.yaml").write_text(
        'TEST:\n  name: "Test Chip"\n  n_qubits: 4\n',
        encoding="utf-8",
    )

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **kwargs: object) -> None:
            captured_load_kwargs.update(kwargs)

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL1

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    def _fake_set_backend_kind(kind: str) -> None:
        selected.append(kind)
        manager.__dict__["_backend_kind"] = kind

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)
    monkeypatch.setattr(manager, "set_backend_kind", _fake_set_backend_kind)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        backend_kind="quel1",
        mock_mode=True,
    )

    assert selected == ["quel1"]
    assert captured_load_kwargs["backend_kind"] == BACKEND_KIND_QUEL1


def test_load_explicit_backend_kind_overrides_system_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given explicit backend_kind, when load is called, then explicit value overrides system.yaml backend."""
    manager = SystemManager.shared()
    captured_load_kwargs: dict[str, object] = {}
    selected: list[str] = []
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()
    (config_dir / "chip.yaml").write_text(
        'TEST:\n  name: "Test Chip"\n  n_qubits: 4\n',
        encoding="utf-8",
    )
    (config_dir / "system.yaml").write_text(
        "schema_version: 1\nchip_id: TEST\nbackend: quel3\n",
        encoding="utf-8",
    )

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **kwargs: object) -> None:
            captured_load_kwargs.update(kwargs)

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL1

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    def _fake_set_backend_kind(kind: str) -> None:
        selected.append(kind)
        manager.__dict__["_backend_kind"] = kind

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)
    monkeypatch.setattr(manager, "set_backend_kind", _fake_set_backend_kind)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        backend_kind="quel1",
        mock_mode=True,
    )

    assert selected == ["quel1"]
    assert captured_load_kwargs["backend_kind"] == BACKEND_KIND_QUEL1


def test_load_defaults_to_quel1_when_chip_and_system_backend_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no backend in system.yaml, when load is called without backend_kind, then backend defaults to quel1."""
    manager = SystemManager.shared()
    selected: list[str] = []
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()
    (config_dir / "chip.yaml").write_text(
        'TEST:\n  name: "Test Chip"\n  n_qubits: 4\n',
        encoding="utf-8",
    )

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **_: object) -> None:
            pass

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL1

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    def _fake_set_backend_kind(kind: str) -> None:
        selected.append(kind)
        manager.__dict__["_backend_kind"] = kind

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)
    monkeypatch.setattr(manager, "set_backend_kind", _fake_set_backend_kind)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        mock_mode=True,
    )

    assert selected == ["quel1"]


def test_load_ignores_unknown_backend_kind_in_chip_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given unknown backend in chip.yaml only, when load is called, then backend defaults to quel1."""
    manager = SystemManager.shared()
    selected: list[str] = []
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()
    (config_dir / "chip.yaml").write_text(
        'TEST:\n  name: "Test Chip"\n  n_qubits: 4\n  backend: unknown\n',
        encoding="utf-8",
    )

    class _FakeConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **_: object) -> None:
            pass

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL1

        def get_experiment_system(self) -> object:
            return SimpleNamespace(hash=hash("TEST"))

    def _fake_set_backend_kind(kind: str) -> None:
        selected.append(kind)
        manager.__dict__["_backend_kind"] = kind

    monkeypatch.setattr("qubex.system.system_manager.ConfigLoader", _FakeConfigLoader)
    monkeypatch.setattr(manager, "set_backend_kind", _fake_set_backend_kind)

    manager.load(
        chip_id="TEST",
        config_dir=config_dir,
        params_dir=params_dir,
        mock_mode=True,
    )

    assert selected == ["quel1"]


def test_load_raises_for_unknown_backend_kind_in_system_config(
    tmp_path: Path,
) -> None:
    """Given unknown backend in system.yaml, when load is called, then ValueError is raised."""
    manager = SystemManager.shared()
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    config_dir.mkdir()
    params_dir.mkdir()
    (config_dir / "chip.yaml").write_text(
        'TEST:\n  name: "Test Chip"\n  n_qubits: 4\n',
        encoding="utf-8",
    )
    (config_dir / "system.yaml").write_text(
        "schema_version: 1\nchip_id: TEST\nbackend: unknown\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported backend"):
        manager.load(
            system_id="TEST",
            config_dir=config_dir,
            params_dir=params_dir,
            mock_mode=True,
        )


def test_load_preserves_backend_kind_when_experiment_system_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given experiment-system resolution failure, when loading, then backend kind and config loader remain unchanged."""
    manager = SystemManager.shared()
    manager.__dict__["_backend_kind"] = BACKEND_KIND_QUEL1
    previous_loader = object()
    manager.__dict__["_config_loader"] = previous_loader
    selected: list[str] = []

    class _FailingConfigLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load(self, **_: object) -> None:
            pass

        @property
        def backend_kind(self) -> str:
            return BACKEND_KIND_QUEL3

        def get_experiment_system(self) -> object:
            raise RuntimeError("ExperimentSystem is not available for chip: TEST")

    def _fake_set_backend_kind(kind: str) -> None:
        selected.append(kind)
        manager.__dict__["_backend_kind"] = kind

    monkeypatch.setattr(
        "qubex.system.system_manager.ConfigLoader",
        _FailingConfigLoader,
    )
    monkeypatch.setattr(manager, "set_backend_kind", _fake_set_backend_kind)

    with pytest.raises(RuntimeError, match="ExperimentSystem is not available"):
        manager.load(
            chip_id="TEST",
            mock_mode=True,
        )

    assert selected == []
    assert manager.backend_kind == BACKEND_KIND_QUEL1
    assert manager.__dict__["_config_loader"] is previous_loader

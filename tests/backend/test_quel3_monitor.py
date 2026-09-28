"""Tests for QuEL-3 monitor configuration and IQ capture."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qubex.backend.quel3 import (
    Quel3BackendController,
)
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
)


class _MonitorClient:
    def __init__(self) -> None:
        self.controls = {"quel3.monitor.mode": "disabled"}
        self.allowed = ("disabled", "loopback")
        self.session_resources: tuple[str, ...] = ()
        self.configured: list[tuple[str, dict[str, str]]] = []

    async def __aenter__(self) -> _MonitorClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def list_unit_labels(self) -> list[str]:
        return ["unit-a", "unit-b"]

    async def list_resource_infos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(id="unit-a:tx_p00", category="PORT"),
            SimpleNamespace(id="unit-a:mon", category="PORT"),
            SimpleNamespace(id="unit-b:tx_p00", category="PORT"),
        ]

    async def get_unit_configuration(self, unit_label: str) -> SimpleNamespace:
        assert unit_label == "unit-a"
        return SimpleNamespace(
            supported=(
                SimpleNamespace(
                    key="quel3.monitor.mode",
                    allowed_values=self.allowed,
                    current_value=self.controls["quel3.monitor.mode"],
                ),
            )
        )

    def create_session(self, resources: tuple[str, ...]) -> _MonitorClient:
        self.session_resources = tuple(resources)
        return self

    async def configure_unit(
        self, unit_label: str, controls: dict[str, str]
    ) -> dict[str, str]:
        self.configured.append((unit_label, controls))
        self.controls.update(controls)
        return dict(self.controls)


@pytest.fixture
def monitor_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Quel3BackendController, _MonitorClient]:
    """Provide a controller backed by a monitor-capable fake quelware client."""
    client = _MonitorClient()
    manager = Quel3ConfigurationManager()
    monkeypatch.setattr(
        manager, "_load_quelware_client_factory", lambda: lambda *args: client
    )
    return Quel3BackendController(configuration_manager=manager), client


def test_configure_monitor_mode_locks_every_unit_port(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Monitor configuration should lock every port and return the applied mode."""
    controller, client = monitor_runtime

    value = controller.configuration_manager.configure_monitor_mode(
        unit_label="unit-a", mode="loopback"
    )

    assert value == "loopback"
    assert client.session_resources == ("unit-a:tx_p00", "unit-a:mon")
    assert client.configured == [("unit-a", {"quel3.monitor.mode": "loopback"})]


def test_configure_monitor_mode_rejects_unsupported_value(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Unsupported monitor values should fail before changing hardware."""
    controller, client = monitor_runtime

    with pytest.raises(ValueError, match="allowed values"):
        controller.configuration_manager.configure_monitor_mode(
            unit_label="unit-a", mode="unknown"
        )

    assert client.configured == []

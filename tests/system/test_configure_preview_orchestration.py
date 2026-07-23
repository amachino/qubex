"""Tests for configure preview orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from qubex.backend.backend_controller import BACKEND_KIND_QUEL1, BACKEND_KIND_QUEL3
from qubex.system import ConfigurePreview
from qubex.system.system_manager import SystemManager


class _PreviewSynchronizerStub:
    def __init__(self, preview: ConfigurePreview) -> None:
        self.preview = preview
        self.calls: list[dict[str, Any]] = []

    def preview_configure(self, **kwargs: Any) -> ConfigurePreview:
        """Record preview calls and return the configured preview."""
        self.calls.append(dict(kwargs))
        return self.preview


def test_system_manager_preview_configure_delegates_to_active_synchronizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given active backend, SystemManager should delegate preview to its synchronizer."""
    manager = SystemManager.shared()
    experiment_system = SimpleNamespace(
        hash=0,
        get_boxes_for_qubits=lambda _qubits: [SimpleNamespace(id="A")],
    )
    preview = ConfigurePreview(
        backend_kind=BACKEND_KIND_QUEL1,
        box_ids=("A",),
        mode="ge-cr-cr",
    )
    synchronizer = _PreviewSynchronizerStub(preview)
    monkeypatch.setattr(
        SystemManager,
        "backend_kind",
        property(lambda _manager: BACKEND_KIND_QUEL1),
    )
    monkeypatch.setattr(
        manager,
        "_resolve_system_synchronizer",
        lambda: synchronizer,
    )
    monkeypatch.setattr(
        manager,
        "_load_preview_experiment_system",
        lambda **_: (experiment_system, BACKEND_KIND_QUEL1),
        raising=False,
    )

    result = manager.preview_configure(
        chip_id="chip",
        system_id="system",
        config_dir="config",
        params_dir="params",
        targets_to_exclude=["Q00"],
        configuration_mode="ge-cr-cr",
        box_ids=None,
        parallel=False,
        target_labels=["Q00"],
        qubit_labels=["Q00"],
    )

    assert result is preview
    assert synchronizer.calls == [
        {
            "experiment_system": experiment_system,
            "box_ids": ["A"],
            "mode": "ge-cr-cr",
            "parallel": False,
            "target_labels": ["Q00"],
        }
    ]


def test_system_manager_preview_configure_rejects_backend_kind_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given preview backend differs from active session, SystemManager should fail early."""
    manager = SystemManager.shared()
    monkeypatch.setattr(
        SystemManager,
        "backend_kind",
        property(lambda _manager: BACKEND_KIND_QUEL1),
    )
    monkeypatch.setattr(
        manager,
        "_load_preview_experiment_system",
        lambda **_: (SimpleNamespace(hash=0), BACKEND_KIND_QUEL3),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="does not match the active session"):
        manager.preview_configure(
            chip_id="chip",
            system_id="system",
            config_dir="config",
            params_dir="params",
            targets_to_exclude=None,
            configuration_mode="ge-cr-cr",
            box_ids=["A"],
        )


def test_system_manager_preview_configure_rejects_quel3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QuEL-3 configure preview should remain explicitly unsupported."""
    manager = SystemManager.shared()
    preview = ConfigurePreview(
        backend_kind=BACKEND_KIND_QUEL3,
        box_ids=("A",),
        mode="ge-cr-cr",
    )
    synchronizer = _PreviewSynchronizerStub(preview)
    monkeypatch.setattr(
        SystemManager,
        "backend_kind",
        property(lambda _manager: BACKEND_KIND_QUEL3),
    )
    monkeypatch.setattr(
        manager,
        "_resolve_system_synchronizer",
        lambda: synchronizer,
    )
    monkeypatch.setattr(
        manager,
        "_load_preview_experiment_system",
        lambda **_: (SimpleNamespace(hash=0), BACKEND_KIND_QUEL3),
        raising=False,
    )

    with pytest.raises(NotImplementedError, match="backend kind: quel3"):
        manager.preview_configure(
            chip_id="chip",
            system_id="system",
            config_dir="config",
            params_dir="params",
            targets_to_exclude=None,
            configuration_mode="ge-cr-cr",
            box_ids=["A"],
        )

    assert synchronizer.calls == []

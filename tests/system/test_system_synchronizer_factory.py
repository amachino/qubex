"""Tests for extensible system synchronizer construction."""

from functools import singledispatch
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import Quel3BackendController
from qubex.system.quel1.quel1_system_synchronizer import Quel1SystemSynchronizer
from qubex.system.quel3.quel3_system_synchronizer import Quel3SystemSynchronizer
from qubex.system.system_synchronizer import SystemSynchronizer
from qubex.system.system_synchronizer_factory import create_system_synchronizer


@pytest.mark.parametrize(
    ("controller_type", "synchronizer_type"),
    [
        (Quel1BackendController, Quel1SystemSynchronizer),
        (Quel3BackendController, Quel3SystemSynchronizer),
    ],
)
def test_builtin_controller_preserves_identity(
    controller_type: type,
    synchronizer_type: type,
) -> None:
    """Built-in controllers and their subclasses should keep their identity in the matching synchronizer."""

    class DerivedController(controller_type):
        pass

    for cls in (controller_type, DerivedController):
        controller = object.__new__(cls)
        synchronizer = create_system_synchronizer(controller)
        assert synchronizer is not None
        assert isinstance(synchronizer, synchronizer_type)
        assert synchronizer.backend_controller is controller


def test_unregistered_controller_has_no_synchronizer() -> None:
    """An unsupported controller should preserve the existing no-synchronizer result."""
    assert create_system_synchronizer(object()) is None


@pytest.fixture
def isolated_factory() -> Any:
    """Copy the dispatcher so extension tests do not mutate production registrations."""
    factory = singledispatch(create_system_synchronizer.dispatch(object))
    for controller_type, builder in create_system_synchronizer.registry.items():
        factory.register(controller_type, builder)
    return factory


def test_register_additional_backend(isolated_factory: Any) -> None:
    """An unrelated backend should become constructible through registration without changing dispatch logic."""

    class AdditionalController:
        pass

    controller = AdditionalController()
    synchronizer = cast(
        SystemSynchronizer, SimpleNamespace(backend_controller=controller)
    )
    assert isolated_factory(controller) is None

    @isolated_factory.register(AdditionalController)
    def create_additional(
        backend_controller: AdditionalController,
    ) -> SystemSynchronizer:
        assert backend_controller is controller
        return synchronizer

    assert isolated_factory(controller) is synchronizer
    assert create_system_synchronizer(controller) is None


def test_subclass_registration_overrides_parent(isolated_factory: Any) -> None:
    """A specialized controller should override its parent's builder without changing the parent binding."""

    class SpecializedController(Quel3BackendController):
        pass

    controller = object.__new__(SpecializedController)
    specialized = cast(
        SystemSynchronizer, SimpleNamespace(backend_controller=controller)
    )
    assert isinstance(isolated_factory(controller), Quel3SystemSynchronizer)

    @isolated_factory.register(SpecializedController)
    def create_specialized(
        backend_controller: SpecializedController,
    ) -> SystemSynchronizer:
        assert backend_controller is controller
        return specialized

    assert isolated_factory(controller) is specialized
    assert isinstance(
        isolated_factory(object.__new__(Quel3BackendController)),
        Quel3SystemSynchronizer,
    )

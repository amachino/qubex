"""Dictionary-based QuEL-3 cable delay settings."""

from qubex.backend.quel3 import Quel3BackendController


def test_cable_delay_dictionaries_are_copied_on_assignment_and_access() -> None:
    """Caller-owned dictionaries should not silently change configured delays."""
    source = {"unit": {"tx": 20.0}}
    controller = Quel3BackendController(cable_delay_ns=source)
    source["unit"]["tx"] = 80.0
    exported = controller.cable_delay_ns
    exported["unit"]["tx"] = 100.0
    assert controller.cable_delay_ns == {"unit": {"tx": 20.0}}
    old_hash = controller.hash
    controller.cable_delay_ns = exported
    assert controller.cable_delay_ns == {"unit": {"tx": 100.0}}
    assert controller.hash != old_hash

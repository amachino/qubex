"""Tests for optional backend dependency checks."""

from importlib.metadata import PackageNotFoundError

import pytest

from tests import _backend_dependencies as dependencies


def test_missing_distribution_skips_with_package_name(monkeypatch) -> None:
    """A missing optional distribution should skip with its install name."""

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(dependencies, "version", missing)

    with pytest.raises(pytest.skip.Exception, match="quelware-client"):
        dependencies.require_distributions("quelware-client")


def test_installed_distributions_are_checked_without_importing_drivers(
    monkeypatch,
) -> None:
    """Dependency checks should inspect every distribution without loading drivers."""
    checked = []
    monkeypatch.setattr(
        dependencies, "version", lambda name: checked.append(name) or "1.0"
    )

    dependencies.require_distributions("quelware-client", "quelware-core")

    assert checked == ["quelware-client", "quelware-core"]


def test_metadata_errors_are_not_silently_skipped(monkeypatch) -> None:
    """Unexpected metadata errors should fail instead of hiding broken installations."""

    def broken(name):
        raise ValueError(f"broken metadata: {name}")

    monkeypatch.setattr(dependencies, "version", broken)

    with pytest.raises(ValueError, match="broken metadata"):
        dependencies.require_distributions("quelware-client")


@pytest.mark.parametrize(
    ("quelware_version", "distribution", "module"),
    [
        ("0.8.14", "qubecalib", "qubecalib"),
        ("0.10.8", "qxdriver-quel1", "qxdriver_quel1"),
    ],
)
def test_quel1_requirement_follows_driver_version_selection(
    monkeypatch,
    quelware_version,
    distribution,
    module,
) -> None:
    """QuEL-1 requirements should follow the supported legacy driver selection."""
    installed = {"quel-ic-config": quelware_version, distribution: "1.0"}
    checked = []

    def installed_version(name):
        checked.append(name)
        return installed[name]

    monkeypatch.setattr(dependencies, "version", installed_version)

    assert dependencies.require_quel1_backend() == module
    assert distribution in checked

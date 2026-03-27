"""Synchronize lockstep package versions from the repository VERSION file."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"
ROOT_PYPROJECT = ROOT / "pyproject.toml"
UV_EXECUTABLE = shutil.which("uv")

WORKSPACE_PACKAGES: tuple[str, ...] = (
    "qxcore",
    "qxfitting",
    "qxpulse",
    "qxschema",
    "qxsimulator",
    "qxvisualizer",
)

NON_WORKSPACE_PACKAGES: tuple[str, ...] = ("qxdriver-quel1",)

PACKAGE_PYPROJECTS: dict[str, Path] = {
    "qubex": ROOT_PYPROJECT,
    "qxcore": ROOT / "packages/qxcore/pyproject.toml",
    "qxfitting": ROOT / "packages/qxfitting/pyproject.toml",
    "qxpulse": ROOT / "packages/qxpulse/pyproject.toml",
    "qxschema": ROOT / "packages/qxschema/pyproject.toml",
    "qxsimulator": ROOT / "packages/qxsimulator/pyproject.toml",
    "qxvisualizer": ROOT / "packages/qxvisualizer/pyproject.toml",
    "qxdriver-quel1": ROOT / "packages/qxdriver-quel1/pyproject.toml",
}

PINNED_DEPENDENCIES: dict[Path, tuple[str, ...]] = {
    ROOT_PYPROJECT: (
        "qxcore",
        "qxfitting",
        "qxpulse",
        "qxschema",
        "qxsimulator",
        "qxvisualizer",
        "qxdriver-quel1",
    ),
    ROOT / "packages/qxpulse/pyproject.toml": ("qxvisualizer",),
    ROOT / "packages/qxschema/pyproject.toml": ("qxcore",),
    ROOT / "packages/qxsimulator/pyproject.toml": ("qxpulse", "qxvisualizer"),
}


def read_release_version() -> str:
    """Return the normalized release version from the VERSION file."""
    version = VERSION_FILE.read_text().strip()
    if not version:
        raise ValueError(f"{VERSION_FILE} is empty.")
    return version


def write_release_version(version: str) -> None:
    """Write one normalized release version into VERSION."""
    VERSION_FILE.write_text(f"{version}\n")


def _run(cmd: list[str]) -> None:
    """Run one command in the repository root."""
    if UV_EXECUTABLE is None:
        raise FileNotFoundError("uv executable was not found on PATH.")
    subprocess.run([UV_EXECUTABLE, *cmd], cwd=ROOT, check=True)  # noqa: S603


def set_pyproject_version(path: Path, *, version: str) -> None:
    """Rewrite one static project.version field in place."""
    text = path.read_text()
    pattern = re.compile(r'(^version\s*=\s*")([^"]+)(")', re.MULTILINE)

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}{version}{match.group(3)}"

    updated_text, count = pattern.subn(_replace, text, count=1)
    if count == 0:
        raise ValueError(f"Static project.version not found in {path}.")
    path.write_text(updated_text)


def set_workspace_versions(version: str) -> None:
    """Set root and companion package versions to one shared value."""
    _run(["version", "--frozen", version])
    for package in WORKSPACE_PACKAGES:
        _run(["version", "--frozen", "--package", package, version])
    for package in NON_WORKSPACE_PACKAGES:
        set_pyproject_version(PACKAGE_PYPROJECTS[package], version=version)


def pin_dependency_version(
    text: str,
    *,
    package: str,
    version: str,
    path: Path,
) -> str:
    """Replace only the pinned version string for one dependency entry."""
    pattern = re.compile(rf'("{re.escape(package)}\s*==\s*)([^"]+)(")')

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}{version}{match.group(3)}"

    updated_text, count = pattern.subn(_replace, text)
    if count == 0:
        raise ValueError(f"Dependency entry for {package!r} not found in {path}.")
    return updated_text


def sync_dependency_pins(version: str) -> None:
    """Rewrite exact intra-workspace dependency pins to the shared version."""
    for path, packages in PINNED_DEPENDENCIES.items():
        text = path.read_text()
        updated_text = text
        for package in packages:
            updated_text = pin_dependency_version(
                updated_text,
                package=package,
                version=version,
                path=path,
            )
        if updated_text != text:
            path.write_text(updated_text)


def sync_release_version(version: str) -> None:
    """Synchronize all workspace package versions and exact pins."""
    set_workspace_versions(version)
    sync_dependency_pins(version)
    _run(["lock"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synchronize workspace package versions from VERSION."
    )
    parser.add_argument(
        "--version",
        help="Set VERSION to this value before synchronizing.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the synchronization entrypoint."""
    args = parse_args()
    version = args.version or read_release_version()
    if args.version:
        write_release_version(version)
    sync_release_version(version)
    print(f"Synchronized workspace release version to {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

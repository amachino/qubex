"""Check lockstep workspace versions against the repository VERSION file."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import sync_release_version as release_version

UV_EXECUTABLE = shutil.which("uv")


def _read_version_from_pyproject(path: Path) -> str:
    """Extract one static project version from a pyproject file."""
    match = re.search(r'^version\s*=\s*"([^"]+)"', path.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError(f"Static project.version not found in {path}.")
    return match.group(1)


def _read_pinned_dependency(path: Path, package: str) -> str:
    """Extract one exact intra-workspace dependency pin from a pyproject file."""
    match = re.search(
        rf'"{re.escape(package)}\s*==\s*([^"]+)"',
        path.read_text(),
    )
    if match is None:
        raise ValueError(f"Exact dependency pin for {package!r} not found in {path}.")
    return match.group(1)


def _check_project_versions(expected_version: str) -> list[str]:
    """Return version drift messages for project.version fields."""
    errors: list[str] = []
    for package, path in release_version.PACKAGE_PYPROJECTS.items():
        actual_version = _read_version_from_pyproject(path)
        if actual_version != expected_version:
            errors.append(
                f"{package}: expected version {expected_version}, found {actual_version} in {path}"
            )
    return errors


def _check_dependency_pins(expected_version: str) -> list[str]:
    """Return drift messages for exact intra-workspace dependency pins."""
    errors: list[str] = []
    for path, packages in release_version.PINNED_DEPENDENCIES.items():
        for package in packages:
            actual_version = _read_pinned_dependency(path, package)
            if actual_version != expected_version:
                errors.append(
                    f"{path}: expected {package} == {expected_version}, found {actual_version}"
                )
    return errors


def _check_lockfile() -> list[str]:
    """Return lockfile drift messages."""
    if UV_EXECUTABLE is None:
        return ["uv executable was not found on PATH."]
    result = subprocess.run(  # noqa: S603
        [UV_EXECUTABLE, "lock", "--check"],
        cwd=release_version.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return []
    output = result.stderr.strip() or result.stdout.strip() or "uv lock --check failed"
    return [output]


def main() -> int:
    """Run the release-version drift check."""
    expected_version = release_version.read_release_version()
    errors = [
        *_check_project_versions(expected_version),
        *_check_dependency_pins(expected_version),
        *_check_lockfile(),
    ]
    if not errors:
        print(f"Release version is synchronized at {expected_version}")
        return 0

    print("Release version drift detected:", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

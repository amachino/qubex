"""Version helpers for Qubex."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse


def _get_installed_version(package_name: str) -> str:
    """Get the installed version for the given package."""
    return importlib.metadata.version(package_name)


def _get_editable_source_dir(package_name: str) -> Path | None:
    """
    Return the editable source directory for a package if available.

    Parameters
    ----------
    package_name : str
        The name of the package to check.

    Returns
    -------
    Path | None
        The editable source directory when installed in editable mode.
    """
    try:
        dist = importlib.metadata.distribution(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    if dist.files is None:
        return None

    for file in dist.files:
        if file.name != "direct_url.json":
            continue
        direct_url_path = dist.locate_file(file)
        try:
            data = json.loads(direct_url_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

        dir_info = data.get("dir_info", {})
        if not dir_info.get("editable", False):
            return None

        url = data.get("url", "")
        if not url:
            return None

        parsed = urlparse(url)
        if parsed.scheme and parsed.scheme != "file":
            return None

        return Path(unquote(parsed.path))

    return None


def _get_git_commit(source_dir: Path) -> str | None:
    """
    Return the short git commit hash for a source directory.

    Parameters
    ----------
    source_dir : Path
        Source directory to resolve the git repository.

    Returns
    -------
    str | None
        Short git commit hash if available.
    """
    git_path = shutil.which("git")
    if git_path is None:
        return None

    try:
        return (
            subprocess.check_output(  # noqa: S603
                [git_path, "-C", str(source_dir), "rev-parse", "--short", "HEAD"]
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def get_version(package_name: str | None = None) -> str:
    """
    Return the package version, appending a git hash for editable installs.

    Parameters
    ----------
    package_name : str | None
        The package name. If None, defaults to `qubex`.

    Returns
    -------
    str
        Version string with a local git hash suffix when available.
    """
    name = package_name or "qubex"
    try:
        version = _get_installed_version(name)
    except importlib.metadata.PackageNotFoundError:
        return f"Package '{name}' is not installed."
    source_dir = _get_editable_source_dir(name)
    if source_dir is None:
        return version

    commit_hash = _get_git_commit(source_dir)
    if commit_hash is None:
        return version

    return f"{version}+g{commit_hash}"


def get_optional_version(package_name: str) -> str | None:
    """
    Return package version with editable git hash suffix, or `None` if missing.

    Parameters
    ----------
    package_name : str
        Distribution name.

    Returns
    -------
    str | None
        Version string when installed, otherwise `None`.
    """
    try:
        version = _get_installed_version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    source_dir = _get_editable_source_dir(package_name)
    if source_dir is None:
        return version

    commit_hash = _get_git_commit(source_dir)
    if commit_hash is None:
        return version

    return f"{version}+g{commit_hash}"


VERSION = get_version()

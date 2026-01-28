import importlib.metadata
import subprocess

from typing_extensions import deprecated

VERSION = "1.5.0a1"


@deprecated("get_version is deprecated, use VERSION constant instead.")
def get_version() -> str:
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return f"{VERSION}+{commit_hash}"
    except Exception:
        return VERSION


def get_package_version(package_name: str) -> str:
    """
    Get the installed version of a specific package using the standard library.

    Parameters
    ----------
    package_name : str
        The name of the package to check.

    Returns
    -------
    str
        The installed version of the package.
    """
    try:
        version = importlib.metadata.version(package_name)
        return version
    except importlib.metadata.PackageNotFoundError:
        return f"Package '{package_name}' is not installed."

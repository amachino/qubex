import importlib.metadata
import subprocess

VERSION = "1.4.6"


def get_version():
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

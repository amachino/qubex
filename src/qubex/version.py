import subprocess

VERSION = "1.0.1"


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

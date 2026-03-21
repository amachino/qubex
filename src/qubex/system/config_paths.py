"""Helpers for resolving default configuration-related filesystem paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

QUBEX_CONFIG_ROOT_ENV_VAR: Final = "QUBEX_CONFIG_ROOT"
USER_CONFIG_ROOT_NAME: Final = "qubex-config"
SHARED_CONFIG_ROOT: Path = Path("/opt/qubex-config")
LEGACY_SHARED_CONFIG_ROOT: Path = Path("/home/shared/qubex-config")
CALIBRATION_NOTE_FILE: Final = "calib_note.json"


def _get_home_dir() -> Path:
    """Return the current user's home directory."""
    return Path.home()


def _candidate_ids(*, system_id: str | None, chip_id: str | None) -> list[str]:
    """Return identifier candidates in priority order without duplicates."""
    result: list[str] = []
    for value in (system_id, chip_id):
        if value is not None and value not in result:
            result.append(value)
    return result


def _first_existing_path(*paths: Path) -> Path | None:
    """Return the first path that exists, or `None` when none exist."""
    for path in paths:
        if path.exists():
            return path
    return None


def get_config_root_dir() -> Path:
    """
    Return the root directory used for default configuration discovery.

    Priority
    --------
    1. `QUBEX_CONFIG_ROOT`
    2. `~/qubex-config` when it exists
    3. `/opt/qubex-config` when it exists
    4. legacy `/home/shared/qubex-config` when it exists
    5. `~/qubex-config`
    """
    env_root = os.getenv(QUBEX_CONFIG_ROOT_ENV_VAR)
    if env_root:
        return Path(env_root).expanduser()

    user_root = _get_home_dir() / USER_CONFIG_ROOT_NAME
    existing = _first_existing_path(
        user_root, SHARED_CONFIG_ROOT, LEGACY_SHARED_CONFIG_ROOT
    )
    return existing or user_root


def resolve_default_config_dir(
    *,
    system_id: str | None,
    chip_id: str | None,
) -> Path:
    """
    Return the default config directory for a system or chip.

    Preferred shared layout is `<root>/config`. Legacy layout keeps config
    under `<root>/<id>/config`, preferring `system_id` and then `chip_id`.
    """
    root = get_config_root_dir()
    preferred = root / "config"
    legacy_candidates = [
        root / config_id / "config"
        for config_id in _candidate_ids(system_id=system_id, chip_id=chip_id)
    ]
    return _first_existing_path(preferred, *legacy_candidates) or preferred


def resolve_default_params_dir(
    *,
    system_id: str | None,
    chip_id: str | None,
) -> Path:
    """
    Return the default parameter directory for a system or chip.

    Preferred shared layout is `<root>/params/<id>`, preferring `system_id`
    and then `chip_id`. Legacy layout keeps params under `<root>/<id>/params`
    with the same identifier priority.
    """
    candidate_ids = _candidate_ids(system_id=system_id, chip_id=chip_id)
    if not candidate_ids:
        raise ValueError("Either `system_id` or `chip_id` must be provided.")

    root = get_config_root_dir()
    shared_candidates = [root / "params" / config_id for config_id in candidate_ids]
    legacy_candidates = [root / config_id / "params" for config_id in candidate_ids]
    preferred = shared_candidates[0]
    return _first_existing_path(*shared_candidates, *legacy_candidates) or preferred


def resolve_default_calibration_note_path(
    *,
    system_id: str | None,
    chip_id: str | None,
) -> Path:
    """
    Return the default calibration note path for a system or chip.

    Preferred shared layout is `<root>/calibration/<id>/calib_note.json`,
    preferring `system_id` and then `chip_id`. Legacy layout keeps calibration
    under `<root>/<id>/calibration/calib_note.json` with the same priority.
    """
    candidate_ids = _candidate_ids(system_id=system_id, chip_id=chip_id)
    if not candidate_ids:
        raise ValueError("Either `system_id` or `chip_id` must be provided.")

    root = get_config_root_dir()
    shared_candidates = [
        root / "calibration" / config_id / CALIBRATION_NOTE_FILE
        for config_id in candidate_ids
    ]
    legacy_candidates = [
        root / config_id / "calibration" / CALIBRATION_NOTE_FILE
        for config_id in candidate_ids
    ]
    preferred = shared_candidates[0]
    return _first_existing_path(*shared_candidates, *legacy_candidates) or preferred

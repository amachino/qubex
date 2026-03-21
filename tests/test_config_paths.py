"""Tests for config path resolution helpers."""

from __future__ import annotations

from pathlib import Path

import qubex.system.config_paths as config_paths


def test_get_config_root_dir_prefers_env_var(monkeypatch, tmp_path: Path) -> None:
    """Given env var, when resolving config root, then it takes precedence."""
    env_root = tmp_path / "env-root"
    home_root = tmp_path / "home" / "qubex-config"
    shared_root = tmp_path / "opt-root"
    legacy_root = tmp_path / "legacy-root"
    env_root.mkdir(parents=True)
    home_root.mkdir(parents=True)
    shared_root.mkdir(parents=True)
    legacy_root.mkdir(parents=True)

    monkeypatch.setattr(config_paths, "SHARED_CONFIG_ROOT", shared_root)
    monkeypatch.setattr(config_paths, "LEGACY_SHARED_CONFIG_ROOT", legacy_root)
    monkeypatch.setattr(config_paths, "_get_home_dir", lambda: tmp_path / "home")
    monkeypatch.setenv(config_paths.QUBEX_CONFIG_ROOT_ENV_VAR, str(env_root))

    assert config_paths.get_config_root_dir() == env_root


def test_get_config_root_dir_falls_back_to_shared_root_when_home_root_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Given no env var and no home root, when resolving config root, then shared root is used."""
    shared_root = tmp_path / "opt-root"
    legacy_root = tmp_path / "legacy-root"
    shared_root.mkdir(parents=True)
    legacy_root.mkdir(parents=True)

    monkeypatch.delenv(config_paths.QUBEX_CONFIG_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(config_paths, "SHARED_CONFIG_ROOT", shared_root)
    monkeypatch.setattr(config_paths, "LEGACY_SHARED_CONFIG_ROOT", legacy_root)
    monkeypatch.setattr(config_paths, "_get_home_dir", lambda: tmp_path / "home")

    assert config_paths.get_config_root_dir() == shared_root


def test_get_config_root_dir_falls_back_to_legacy_when_shared_root_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Given no env var and no newer roots, when resolving config root, then legacy root is used."""
    legacy_root = tmp_path / "legacy-root"
    legacy_root.mkdir(parents=True)

    monkeypatch.delenv(config_paths.QUBEX_CONFIG_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(config_paths, "SHARED_CONFIG_ROOT", tmp_path / "opt-root")
    monkeypatch.setattr(config_paths, "LEGACY_SHARED_CONFIG_ROOT", legacy_root)
    monkeypatch.setattr(config_paths, "_get_home_dir", lambda: tmp_path / "home")

    assert config_paths.get_config_root_dir() == legacy_root


def test_resolve_default_config_dir_prefers_shared_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Given shared config dir, when resolving config dir, then shared layout is preferred."""
    root = tmp_path / "qubex-config"
    shared_config_dir = root / "config"
    legacy_config_dir = root / "CHIP_A" / "config"
    shared_config_dir.mkdir(parents=True)
    legacy_config_dir.mkdir(parents=True)

    monkeypatch.setenv(config_paths.QUBEX_CONFIG_ROOT_ENV_VAR, str(root))

    assert (
        config_paths.resolve_default_config_dir(
            system_id="SYSTEM_A",
            chip_id="CHIP_A",
        )
        == shared_config_dir
    )


def test_resolve_default_params_dir_prefers_system_shared_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Given shared params dir for a system, when resolving params dir, then that path is used."""
    root = tmp_path / "qubex-config"
    system_params_dir = root / "params" / "SYSTEM_A"
    chip_params_dir = root / "CHIP_A" / "params"
    system_params_dir.mkdir(parents=True)
    chip_params_dir.mkdir(parents=True)

    monkeypatch.setenv(config_paths.QUBEX_CONFIG_ROOT_ENV_VAR, str(root))

    assert (
        config_paths.resolve_default_params_dir(
            system_id="SYSTEM_A",
            chip_id="CHIP_A",
        )
        == system_params_dir
    )


def test_resolve_default_calibration_note_path_prefers_system_shared_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Given shared calibration note path, when resolving it, then system shared layout is used."""
    root = tmp_path / "qubex-config"
    shared_note_path = root / "calibration" / "SYSTEM_A" / "calib_note.json"
    legacy_note_path = root / "CHIP_A" / "calibration" / "calib_note.json"
    shared_note_path.parent.mkdir(parents=True)
    legacy_note_path.parent.mkdir(parents=True)
    shared_note_path.write_text("{}", encoding="utf-8")
    legacy_note_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv(config_paths.QUBEX_CONFIG_ROOT_ENV_VAR, str(root))

    assert (
        config_paths.resolve_default_calibration_note_path(
            system_id="SYSTEM_A",
            chip_id="CHIP_A",
        )
        == shared_note_path
    )


def test_resolve_default_params_dir_uses_legacy_system_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Given legacy nested system params, when resolving params dir, then system layout is used."""
    root = tmp_path / "legacy-root"
    legacy_system_params_dir = root / "SYSTEM_A" / "params"
    legacy_chip_params_dir = root / "CHIP_A" / "params"
    legacy_system_params_dir.mkdir(parents=True)
    legacy_chip_params_dir.mkdir(parents=True)

    monkeypatch.delenv(config_paths.QUBEX_CONFIG_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(config_paths, "SHARED_CONFIG_ROOT", tmp_path / "opt-root")
    monkeypatch.setattr(config_paths, "LEGACY_SHARED_CONFIG_ROOT", root)
    monkeypatch.setattr(config_paths, "_get_home_dir", lambda: tmp_path / "home")

    assert (
        config_paths.resolve_default_params_dir(
            system_id="SYSTEM_A",
            chip_id="CHIP_A",
        )
        == legacy_system_params_dir
    )

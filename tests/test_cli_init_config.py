"""Tests for `qubex` CLI config initialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qubex.cli import main


@pytest.mark.parametrize("chip_id", ["64Qv2", "144Qv2"])
def test_init_config_creates_expected_files(tmp_path: Path, chip_id: str) -> None:
    """Given a target directory, when init-config runs, then required YAML templates are created."""
    # Arrange
    output_dir = tmp_path / "config-root"

    # Act
    exit_code = main(
        ["init-config", "--chip-id", chip_id, "--output-dir", str(output_dir)]
    )

    # Assert
    assert exit_code == 0

    config_dir = output_dir / chip_id / "config"
    params_dir = output_dir / chip_id / "params"
    assert (config_dir / "chip.yaml").exists()
    assert (config_dir / "box.yaml").exists()
    assert (config_dir / "wiring.yaml").exists()
    assert (config_dir / "skew.yaml").exists()
    assert (params_dir / "params.yaml").exists()
    assert (params_dir / "props.yaml").exists()

    chip_payload = yaml.safe_load((config_dir / "chip.yaml").read_text())
    assert chip_id in chip_payload
    assert chip_payload[chip_id]["backend"] == "quel1"


def test_init_config_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    """Given existing files, when init-config runs without force, then it fails without overwriting."""
    # Arrange
    output_dir = tmp_path / "config-root"
    chip_id = "64Qv2"
    first_exit_code = main(
        ["init-config", "--chip-id", chip_id, "--output-dir", str(output_dir)]
    )
    assert first_exit_code == 0

    target_file = output_dir / chip_id / "config" / "chip.yaml"
    original_content = target_file.read_text()

    # Act
    second_exit_code = main(
        ["init-config", "--chip-id", chip_id, "--output-dir", str(output_dir)]
    )

    # Assert
    assert second_exit_code == 1
    assert target_file.read_text() == original_content


def test_init_config_overwrites_when_force_enabled(tmp_path: Path) -> None:
    """Given existing files, when init-config runs with force, then templates are overwritten."""
    # Arrange
    output_dir = tmp_path / "config-root"
    chip_id = "64Qv2"
    first_exit_code = main(
        ["init-config", "--chip-id", chip_id, "--output-dir", str(output_dir)]
    )
    assert first_exit_code == 0

    target_file = output_dir / chip_id / "config" / "chip.yaml"
    target_file.write_text("broken: true\n")

    # Act
    second_exit_code = main(
        [
            "init-config",
            "--chip-id",
            chip_id,
            "--output-dir",
            str(output_dir),
            "--force",
        ]
    )

    # Assert
    assert second_exit_code == 0
    chip_payload = yaml.safe_load(target_file.read_text())
    assert chip_id in chip_payload

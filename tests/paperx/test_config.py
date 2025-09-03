"""Tests for paperx configuration management."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qubex.paperx.config import PaperConfig


def test_paper_config_initialization(temp_config_dir: Path):
    """Test PaperConfig initialization."""
    config = PaperConfig(config_dir=temp_config_dir)
    
    assert config._config_dir == temp_config_dir
    assert config.template_name == "default"
    assert config.preamble_file == "preamble.yaml"
    assert config.macros_file == "macros.yaml"
    assert config.references_file == "references.yaml"


def test_paper_config_default_directory():
    """Test default configuration directory."""
    config = PaperConfig()
    expected_dir = Path.home() / ".paperx" / "config"
    assert config._config_dir == expected_dir


def test_load_config_file_success(setup_config_files: Path):
    """Test successful loading of configuration files."""
    config = PaperConfig(config_dir=setup_config_files)
    
    # Test loading each configuration type
    preamble = config.preamble_config
    assert "packages" in preamble
    assert "amsmath" in preamble["packages"]
    
    macros = config.macros_config
    assert "commands" in macros
    assert "qubex" in macros["commands"]
    
    references = config.references_config
    assert "entries" in references
    assert "test_ref_1" in references["entries"]


def test_load_config_file_not_found(temp_config_dir: Path):
    """Test handling of missing configuration files."""
    config = PaperConfig(config_dir=temp_config_dir)
    
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        _ = config.preamble_config


def test_load_invalid_yaml(temp_config_dir: Path):
    """Test handling of invalid YAML files."""
    # Create invalid YAML file
    invalid_yaml = temp_config_dir / "preamble.yaml"
    with open(invalid_yaml, "w") as f:
        f.write("invalid: yaml: content: [")
    
    config = PaperConfig(config_dir=temp_config_dir)
    
    with pytest.raises(yaml.YAMLError, match="Error loading configuration file"):
        _ = config.preamble_config


def test_get_template_config(setup_config_files: Path):
    """Test template configuration loading."""
    config = PaperConfig(config_dir=setup_config_files)
    
    template_config = config.get_template_config("default")
    assert template_config["documentclass"] == "article"
    assert "11pt" in template_config["documentclass_options"]


def test_get_template_config_not_found(temp_config_dir: Path):
    """Test handling of missing template configuration."""
    config = PaperConfig(config_dir=temp_config_dir)
    
    with pytest.raises(FileNotFoundError):
        config.get_template_config("nonexistent")


def test_config_caching(setup_config_files: Path):
    """Test that configurations are cached after first load."""
    config = PaperConfig(config_dir=setup_config_files)
    
    # First access loads the config
    preamble1 = config.preamble_config
    preamble2 = config.preamble_config
    
    # Should be the same object (cached)
    assert preamble1 is preamble2


def test_custom_file_names(temp_config_dir: Path):
    """Test custom configuration file names."""
    # Create custom config files
    custom_preamble = {"packages": ["test"]}
    with open(temp_config_dir / "custom_preamble.yaml", "w") as f:
        yaml.dump(custom_preamble, f)
    
    config = PaperConfig(
        config_dir=temp_config_dir,
        preamble_file="custom_preamble.yaml",
    )
    
    preamble = config.preamble_config
    assert preamble["packages"] == ["test"]


def test_empty_config_file(temp_config_dir: Path):
    """Test handling of empty configuration files."""
    # Create empty config file
    empty_file = temp_config_dir / "preamble.yaml"
    empty_file.touch()
    
    config = PaperConfig(config_dir=temp_config_dir)
    preamble = config.preamble_config
    
    # Should return empty dict for empty file
    assert preamble == {}
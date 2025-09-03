"""Tests for paperx template management."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qubex.paperx.config import PaperConfig
from qubex.paperx.templates import TemplateManager


def test_template_manager_initialization(setup_config_files: Path):
    """Test TemplateManager initialization."""
    config = PaperConfig(config_dir=setup_config_files)
    template_manager = TemplateManager(config)
    
    assert template_manager.config is config


def test_load_template(setup_config_files: Path):
    """Test loading template configuration."""
    config = PaperConfig(config_dir=setup_config_files)
    template_manager = TemplateManager(config)
    
    template = template_manager.load_template("default")
    assert template["documentclass"] == "article"
    assert "11pt" in template["documentclass_options"]


def test_load_default_template(setup_config_files: Path):
    """Test loading default template."""
    config = PaperConfig(config_dir=setup_config_files, template_name="default")
    template_manager = TemplateManager(config)
    
    template = template_manager.load_template()
    assert template["documentclass"] == "article"


def test_generate_preamble_basic(setup_config_files: Path):
    """Test basic preamble generation."""
    config = PaperConfig(config_dir=setup_config_files)
    template_manager = TemplateManager(config)
    
    preamble = template_manager.generate_preamble()
    
    # Check document class
    assert "\\documentclass[11pt,a4paper]{article}" in preamble
    
    # Check packages
    assert "\\usepackage{amsmath}" in preamble
    assert "\\usepackage{amsfonts}" in preamble
    assert "\\usepackage[margin=1in]{geometry}" in preamble
    
    # Check template additions
    assert "% Test template" in preamble


def test_generate_preamble_no_options(temp_config_dir: Path):
    """Test preamble generation without document class options."""
    # Create minimal template config
    template_config = {"documentclass": "article"}
    with open(temp_config_dir / "template_minimal.yaml", "w") as f:
        yaml.dump(template_config, f)
    
    # Create minimal preamble config
    preamble_config = {"packages": ["amsmath"]}
    with open(temp_config_dir / "preamble.yaml", "w") as f:
        yaml.dump(preamble_config, f)
    
    # Create empty macros config
    with open(temp_config_dir / "macros.yaml", "w") as f:
        yaml.dump({}, f)
    
    config = PaperConfig(config_dir=temp_config_dir)
    template_manager = TemplateManager(config)
    
    preamble = template_manager.generate_preamble("minimal")
    assert "\\documentclass{article}" in preamble
    assert "\\usepackage{amsmath}" in preamble


def test_generate_macros(setup_config_files: Path):
    """Test macro generation."""
    config = PaperConfig(config_dir=setup_config_files)
    template_manager = TemplateManager(config)
    
    macros = template_manager.generate_macros()
    
    # Check simple commands
    assert "\\newcommand{\\qubex}{\\texttt{qubex}}" in macros
    
    # Check commands with arguments
    assert "\\newcommand{\\code}[1]{\\texttt{#1}}" in macros
    
    # Check environments
    assert "\\newenvironment{theorem}" in macros
    
    # Check direct LaTeX
    assert "\\newcommand{\\ket}[1]{|#1\\rangle}" in macros


def test_generate_macros_empty_config(temp_config_dir: Path):
    """Test macro generation with empty configuration."""
    with open(temp_config_dir / "macros.yaml", "w") as f:
        yaml.dump({}, f)
    
    config = PaperConfig(config_dir=temp_config_dir)
    template_manager = TemplateManager(config)
    
    macros = template_manager.generate_macros()
    assert macros == ""


def test_get_document_structure(setup_config_files: Path):
    """Test getting document structure."""
    config = PaperConfig(config_dir=setup_config_files)
    template_manager = TemplateManager(config)
    
    structure = template_manager.get_document_structure()
    assert "sections" in structure
    assert "Introduction" in structure["sections"]
    assert structure["include_abstract"] is True


def test_list_available_templates(temp_config_dir: Path):
    """Test listing available templates."""
    # Create multiple template files
    template_names = ["default", "ieee", "aps"]
    for name in template_names:
        template_file = temp_config_dir / f"template_{name}.yaml"
        with open(template_file, "w") as f:
            yaml.dump({"documentclass": "article"}, f)
    
    config = PaperConfig(config_dir=temp_config_dir)
    template_manager = TemplateManager(config)
    
    available = template_manager.list_available_templates()
    assert sorted(available) == sorted(template_names)


def test_validate_template_valid(setup_config_files: Path):
    """Test template validation for valid template."""
    config = PaperConfig(config_dir=setup_config_files)
    template_manager = TemplateManager(config)
    
    is_valid, errors = template_manager.validate_template("default")
    assert is_valid is True
    assert len(errors) == 0


def test_validate_template_missing_documentclass(temp_config_dir: Path):
    """Test template validation for missing documentclass."""
    # Create template without documentclass
    template_config = {"structure": {"sections": ["Introduction"]}}
    with open(temp_config_dir / "template_invalid.yaml", "w") as f:
        yaml.dump(template_config, f)
    
    config = PaperConfig(config_dir=temp_config_dir)
    template_manager = TemplateManager(config)
    
    is_valid, errors = template_manager.validate_template("invalid")
    assert is_valid is False
    assert "Missing required field: documentclass" in errors


def test_validate_template_invalid_options(temp_config_dir: Path):
    """Test template validation for invalid options."""
    # Create template with invalid options
    template_config = {
        "documentclass": "article",
        "documentclass_options": "invalid",  # Should be list
    }
    with open(temp_config_dir / "template_invalid.yaml", "w") as f:
        yaml.dump(template_config, f)
    
    config = PaperConfig(config_dir=temp_config_dir)
    template_manager = TemplateManager(config)
    
    is_valid, errors = template_manager.validate_template("invalid")
    assert is_valid is False
    assert "documentclass_options must be a list" in errors


def test_validate_template_nonexistent(temp_config_dir: Path):
    """Test template validation for nonexistent template."""
    config = PaperConfig(config_dir=temp_config_dir)
    template_manager = TemplateManager(config)
    
    is_valid, errors = template_manager.validate_template("nonexistent")
    assert is_valid is False
    assert len(errors) > 0
    assert "Error loading template" in errors[0]
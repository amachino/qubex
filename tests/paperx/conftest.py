"""Fixtures for paperx tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import yaml


@pytest.fixture
def temp_config_dir():
    """Create a temporary configuration directory."""
    with TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        yield config_dir


@pytest.fixture  
def sample_preamble_config() -> dict[str, Any]:
    """Sample preamble configuration."""
    return {
        "packages": [
            "amsmath",
            "amsfonts", 
            {"name": "geometry", "options": ["margin=1in"]},
        ],
        "fonts": {"encoding": "utf8"},
    }


@pytest.fixture
def sample_macros_config() -> dict[str, Any]:
    """Sample macros configuration."""
    return {
        "commands": {
            "qubex": "\\texttt{qubex}",
            "code": {"args": 1, "definition": "\\texttt{#1}"},
        },
        "environments": {
            "theorem": {
                "begin": "\\textbf{Theorem.} \\itshape",
                "end": "\\upshape",
            }
        },
        "direct_latex": ["\\newcommand{\\ket}[1]{|#1\\rangle}"],
    }


@pytest.fixture
def sample_references_config() -> dict[str, Any]:
    """Sample references configuration."""
    return {
        "entries": {
            "test_ref_1": {
                "type": "article",
                "title": "Test Article",
                "author": "Test Author",
                "journal": "Test Journal",
                "year": "2024",
            },
            "test_ref_2": {
                "type": "book",
                "title": "Test Book",
                "author": "Test Author",
                "publisher": "Test Publisher",
                "year": "2023",
            },
        }
    }


@pytest.fixture
def sample_template_config() -> dict[str, Any]:
    """Sample template configuration."""
    return {
        "documentclass": "article",
        "documentclass_options": ["11pt", "a4paper"],
        "structure": {
            "sections": ["Introduction", "Methods", "Results"],
            "include_abstract": True,
        },
        "preamble_additions": ["% Test template"],
    }


@pytest.fixture
def setup_config_files(
    temp_config_dir: Path,
    sample_preamble_config: dict[str, Any],
    sample_macros_config: dict[str, Any],
    sample_references_config: dict[str, Any],
    sample_template_config: dict[str, Any],
):
    """Set up configuration files in temporary directory."""
    # Write preamble config
    with open(temp_config_dir / "preamble.yaml", "w") as f:
        yaml.dump(sample_preamble_config, f)
    
    # Write macros config
    with open(temp_config_dir / "macros.yaml", "w") as f:
        yaml.dump(sample_macros_config, f)
    
    # Write references config
    with open(temp_config_dir / "references.yaml", "w") as f:
        yaml.dump(sample_references_config, f)
    
    # Write template config
    with open(temp_config_dir / "template_default.yaml", "w") as f:
        yaml.dump(sample_template_config, f)
    
    return temp_config_dir
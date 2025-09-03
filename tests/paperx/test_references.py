"""Tests for paperx reference management."""

from __future__ import annotations

from pathlib import Path

import pytest

from qubex.paperx.config import PaperConfig
from qubex.paperx.references import ReferenceManager


def test_reference_manager_initialization(setup_config_files: Path):
    """Test ReferenceManager initialization."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    assert ref_manager.config is config
    assert len(ref_manager._reference_database) > 0


def test_load_references_from_config(setup_config_files: Path):
    """Test loading references from configuration."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    # Check that references were loaded
    assert "test_ref_1" in ref_manager._reference_database
    assert "test_ref_2" in ref_manager._reference_database
    
    # Check reference data
    ref1 = ref_manager._reference_database["test_ref_1"]
    assert ref1["title"] == "Test Article"
    assert ref1["type"] == "article"


def test_get_references_for_paper(setup_config_files: Path):
    """Test getting references for a specific paper."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    paper_refs = ["test_ref_1", "test_ref_2"]
    selected_refs = ref_manager.get_references_for_paper(paper_refs)
    
    assert len(selected_refs) == 2
    assert "test_ref_1" in selected_refs
    assert "test_ref_2" in selected_refs
    assert selected_refs["test_ref_1"]["title"] == "Test Article"


def test_get_references_for_paper_missing_key(setup_config_files: Path):
    """Test handling of missing reference keys."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    with pytest.raises(KeyError, match="Reference 'nonexistent' not found"):
        ref_manager.get_references_for_paper(["nonexistent"])


def test_generate_bibtex(setup_config_files: Path):
    """Test BibTeX generation."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    paper_refs = ["test_ref_1"]
    bibtex = ref_manager.generate_bibtex(paper_refs)
    
    # Check BibTeX format
    assert "@article{test_ref_1," in bibtex
    assert "title = {Test Article}," in bibtex
    assert "author = {Test Author}," in bibtex
    assert "journal = {Test Journal}," in bibtex
    assert "year = {2024}," in bibtex


def test_generate_bibtex_multiple_refs(setup_config_files: Path):
    """Test BibTeX generation for multiple references."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    paper_refs = ["test_ref_1", "test_ref_2"]
    bibtex = ref_manager.generate_bibtex(paper_refs)
    
    # Check both references are included
    assert "@article{test_ref_1," in bibtex
    assert "@book{test_ref_2," in bibtex
    assert "Test Article" in bibtex
    assert "Test Book" in bibtex


def test_list_available_references(setup_config_files: Path):
    """Test listing available references."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    available = ref_manager.list_available_references()
    assert "test_ref_1" in available
    assert "test_ref_2" in available
    assert len(available) == 2


def test_search_references_by_title(setup_config_files: Path):
    """Test searching references by title."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    # Search by title
    results = ref_manager.search_references("Test Article")
    assert "test_ref_1" in results
    assert len(results) == 1


def test_search_references_by_author(setup_config_files: Path):
    """Test searching references by author."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    # Search by author (should find both references)
    results = ref_manager.search_references("Test Author")
    assert "test_ref_1" in results
    assert "test_ref_2" in results
    assert len(results) == 2


def test_search_references_case_insensitive(setup_config_files: Path):
    """Test case-insensitive reference searching."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    # Search with different case
    results = ref_manager.search_references("test article")
    assert "test_ref_1" in results


def test_search_references_by_key(setup_config_files: Path):
    """Test searching references by key."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    # Search by partial key
    results = ref_manager.search_references("ref_1")
    assert "test_ref_1" in results


def test_search_references_no_matches(setup_config_files: Path):
    """Test searching with no matches."""
    config = PaperConfig(config_dir=setup_config_files)
    ref_manager = ReferenceManager(config)
    
    results = ref_manager.search_references("nonexistent query")
    assert len(results) == 0


def test_parse_bibtex_simple():
    """Test simple BibTeX parsing."""
    from qubex.paperx.references import ReferenceManager
    
    # Create a dummy config and reference manager
    config = PaperConfig()
    ref_manager = ReferenceManager.__new__(ReferenceManager)
    
    bibtex_content = """
    @article{test2024,
      title={Test Article},
      author={Test Author},
      journal={Test Journal},
      year={2024}
    }
    
    @book{book2023,
      title={Test Book},
      author={Book Author},
      publisher={Test Publisher},
      year={2023}
    }
    """
    
    entries = ref_manager._parse_bibtex_simple(bibtex_content)
    
    assert "test2024" in entries
    assert "book2023" in entries
    assert entries["test2024"]["type"] == "article"
    assert entries["test2024"]["title"] == "Test Article"
    assert entries["book2023"]["type"] == "book"


def test_load_bibtex_database_missing_file(temp_config_dir: Path):
    """Test loading BibTeX database from missing file."""
    config = PaperConfig(config_dir=temp_config_dir)
    ref_manager = ReferenceManager.__new__(ReferenceManager)
    ref_manager.config = config
    
    # Try to load non-existent file
    missing_path = temp_config_dir / "missing.bib"
    refs = ref_manager._load_bibtex_database(missing_path)
    
    # Should return empty dict for missing file
    assert refs == {}
"""Tests for paperx document generation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from qubex.paperx.config import PaperConfig
from qubex.paperx.document import DocumentGenerator


def test_document_generator_initialization(setup_config_files: Path):
    """Test DocumentGenerator initialization."""
    config = PaperConfig(config_dir=setup_config_files)
    doc_gen = DocumentGenerator(config)
    
    assert doc_gen.config is config
    assert doc_gen.template_manager is not None
    assert doc_gen.reference_manager is not None


def test_document_generator_initialization_no_config(setup_config_files: Path):
    """Test DocumentGenerator initialization without explicit config."""
    doc_gen = DocumentGenerator(config_dir=setup_config_files)
    
    assert doc_gen.config is not None
    assert doc_gen.config._config_dir == setup_config_files


def test_generate_document_basic(setup_config_files: Path):
    """Test basic document generation."""
    config = PaperConfig(config_dir=setup_config_files)
    doc_gen = DocumentGenerator(config)
    
    content = "This is test content."
    
    with TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.tex"
        
        result_path = doc_gen.generate_document(
            content=content,
            output_path=output_path,
            title="Test Paper",
            author="Test Author",
        )
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Check document content
        with open(output_path, "r") as f:
            doc_content = f.read()
        
        assert "\\documentclass[11pt,a4paper]{article}" in doc_content
        assert "\\title{Test Paper}" in doc_content
        assert "\\author{Test Author}" in doc_content
        assert "\\maketitle" in doc_content
        assert "This is test content." in doc_content
        assert "\\end{document}" in doc_content


def test_generate_document_with_abstract(setup_config_files: Path):
    """Test document generation with abstract."""
    config = PaperConfig(config_dir=setup_config_files)
    doc_gen = DocumentGenerator(config)
    
    content = "Main content."
    abstract = "This is the abstract."
    
    with TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.tex"
        
        doc_gen.generate_document(
            content=content,
            output_path=output_path,
            abstract=abstract,
        )
        
        with open(output_path, "r") as f:
            doc_content = f.read()
        
        assert "\\begin{abstract}" in doc_content
        assert "This is the abstract." in doc_content
        assert "\\end{abstract}" in doc_content


def test_generate_document_with_references(setup_config_files: Path):
    """Test document generation with references."""
    config = PaperConfig(config_dir=setup_config_files)
    doc_gen = DocumentGenerator(config)
    
    content = "This cites \\cite{test_ref_1}."
    references = ["test_ref_1", "test_ref_2"]
    
    with TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.tex"
        
        doc_gen.generate_document(
            content=content,
            output_path=output_path,
            references=references,
        )
        
        # Check main document
        with open(output_path, "r") as f:
            doc_content = f.read()
        
        assert "\\bibliography{\\jobname}" in doc_content
        assert "\\bibliographystyle{plain}" in doc_content
        
        # Check that .bib file was created
        bib_path = output_path.with_suffix(".bib")
        assert bib_path.exists()
        
        with open(bib_path, "r") as f:
            bib_content = f.read()
        
        assert "@article{test_ref_1," in bib_content
        assert "@book{test_ref_2," in bib_content


def test_generate_document_custom_date(setup_config_files: Path):
    """Test document generation with custom date."""
    config = PaperConfig(config_dir=setup_config_files)
    doc_gen = DocumentGenerator(config)
    
    content = "Test content."
    
    with TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.tex"
        
        doc_gen.generate_document(
            content=content,
            output_path=output_path,
            date="January 2024",
        )
        
        with open(output_path, "r") as f:
            doc_content = f.read()
        
        assert "\\date{January 2024}" in doc_content


def test_generate_document_no_date(setup_config_files: Path):
    """Test document generation without explicit date."""
    config = PaperConfig(config_dir=setup_config_files)
    doc_gen = DocumentGenerator(config)
    
    content = "Test content."
    
    with TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.tex"
        
        doc_gen.generate_document(
            content=content,
            output_path=output_path,
        )
        
        with open(output_path, "r") as f:
            doc_content = f.read()
        
        assert "\\date{\\today}" in doc_content


def test_generate_document_creates_directories(setup_config_files: Path):
    """Test that document generation creates output directories."""
    config = PaperConfig(config_dir=setup_config_files)
    doc_gen = DocumentGenerator(config)
    
    content = "Test content."
    
    with TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "subdir" / "test.tex"
        
        doc_gen.generate_document(
            content=content,
            output_path=output_path,
        )
        
        assert output_path.exists()
        assert output_path.parent.exists()


def test_create_project(setup_config_files: Path):
    """Test creating a new paperx project."""
    config = PaperConfig(config_dir=setup_config_files)
    doc_gen = DocumentGenerator(config)
    
    with TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir) / "test_project"
        
        result_path = doc_gen.create_project(project_path)
        
        assert result_path == project_path
        assert project_path.exists()
        
        # Check that required files were created
        assert (project_path / "content.tex").exists()
        assert (project_path / "paper.yaml").exists()
        assert (project_path / "README.md").exists()
        
        # Check content.tex has sample content
        with open(project_path / "content.tex", "r") as f:
            content = f.read()
        assert "\\section{Introduction}" in content
        
        # Check paper.yaml has sample configuration
        with open(project_path / "paper.yaml", "r") as f:
            config_content = f.read()
        assert "template: default" in config_content
        assert "title:" in config_content


def test_create_project_custom_template(setup_config_files: Path):
    """Test creating project with custom template."""
    config = PaperConfig(config_dir=setup_config_files)
    doc_gen = DocumentGenerator(config)
    
    with TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir) / "test_project"
        
        doc_gen.create_project(project_path, template_name="ieee")
        
        with open(project_path / "paper.yaml", "r") as f:
            config_content = f.read()
        assert "template: ieee" in config_content


def test_generate_document_content_minimal(temp_config_dir: Path):
    """Test document content generation with minimal parameters."""
    # Create minimal config files
    for filename in ["preamble.yaml", "macros.yaml", "references.yaml"]:
        with open(temp_config_dir / filename, "w") as f:
            yaml.dump({}, f)
    
    # Create minimal template
    with open(temp_config_dir / "template_default.yaml", "w") as f:
        yaml.dump({"documentclass": "article"}, f)
    
    doc_gen = DocumentGenerator(config_dir=temp_config_dir)
    
    content = doc_gen._generate_document_content("Test content.")
    
    assert "\\begin{document}" in content
    assert "Test content." in content
    assert "\\date{\\today}" in content


def test_generate_document_content_full(temp_config_dir: Path):
    """Test document content generation with all parameters."""
    # Create minimal config files
    for filename in ["preamble.yaml", "macros.yaml", "references.yaml"]:
        with open(temp_config_dir / filename, "w") as f:
            yaml.dump({}, f)
    
    # Create minimal template
    with open(temp_config_dir / "template_default.yaml", "w") as f:
        yaml.dump({"documentclass": "article"}, f)
    
    doc_gen = DocumentGenerator(config_dir=temp_config_dir)
    
    content = doc_gen._generate_document_content(
        content="Main content.",
        title="Test Title",
        author="Test Author",
        date="2024",
        abstract="Test abstract.",
    )
    
    assert "\\title{Test Title}" in content
    assert "\\author{Test Author}" in content
    assert "\\date{2024}" in content
    assert "\\maketitle" in content
    assert "\\begin{abstract}" in content
    assert "Test abstract." in content
    assert "Main content." in content


def test_generate_bibliography_empty(temp_config_dir: Path):
    """Test bibliography generation with empty references."""
    # Create minimal config files
    for filename in ["preamble.yaml", "macros.yaml", "references.yaml"]:
        with open(temp_config_dir / filename, "w") as f:
            yaml.dump({}, f)
    
    # Create minimal template
    with open(temp_config_dir / "template_default.yaml", "w") as f:
        yaml.dump({"documentclass": "article"}, f)
    
    doc_gen = DocumentGenerator(config_dir=temp_config_dir)
    
    bibliography = doc_gen._generate_bibliography([])
    assert bibliography == ""


def test_generate_bibliography_with_refs(temp_config_dir: Path):
    """Test bibliography generation with references."""
    # Create minimal config files
    for filename in ["preamble.yaml", "macros.yaml", "references.yaml"]:
        with open(temp_config_dir / filename, "w") as f:
            yaml.dump({}, f)
    
    # Create minimal template
    with open(temp_config_dir / "template_default.yaml", "w") as f:
        yaml.dump({"documentclass": "article"}, f)
    
    doc_gen = DocumentGenerator(config_dir=temp_config_dir)
    
    bibliography = doc_gen._generate_bibliography(["ref1", "ref2"])
    
    assert "\\bibliographystyle{plain}" in bibliography
    assert "\\bibliography{\\jobname}" in bibliography


def test_combine_document_parts(temp_config_dir: Path):
    """Test combining document parts."""
    # Create minimal config files
    for filename in ["preamble.yaml", "macros.yaml", "references.yaml"]:
        with open(temp_config_dir / filename, "w") as f:
            yaml.dump({}, f)
    
    # Create minimal template
    with open(temp_config_dir / "template_default.yaml", "w") as f:
        yaml.dump({"documentclass": "article"}, f)
    
    doc_gen = DocumentGenerator(config_dir=temp_config_dir)
    
    preamble = "\\documentclass{article}"
    content = "\\begin{document}\nContent\n"
    bibliography = "\\bibliography{refs}"
    
    full_doc = doc_gen._combine_document_parts(preamble, content, bibliography)
    
    assert preamble in full_doc
    assert content in full_doc
    assert bibliography in full_doc
    assert "\\end{document}" in full_doc
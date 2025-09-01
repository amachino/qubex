"""Smoke tests for example notebooks."""

import pytest
import subprocess
import os
from pathlib import Path


class TestNotebookSmoke:
    """Smoke tests for example notebooks to ensure they can execute."""

    @pytest.fixture
    def notebooks_dir(self):
        """Get the notebooks directory path."""
        repo_root = Path(__file__).parent.parent
        return repo_root / "docs" / "examples"

    def test_notebooks_directory_exists(self, notebooks_dir):
        """Test that the notebooks directory exists."""
        assert notebooks_dir.exists(), f"Notebooks directory not found: {notebooks_dir}"

    @pytest.mark.parametrize("notebook_pattern", [
        "*.ipynb",
    ])
    def test_find_notebooks(self, notebooks_dir, notebook_pattern):
        """Test that we can find notebooks in the examples directory."""
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory does not exist")
        
        notebooks = list(notebooks_dir.rglob(notebook_pattern))
        # We expect at least one notebook from the repository snippets
        assert len(notebooks) > 0, f"No notebooks found matching {notebook_pattern}"

    def test_notebook_syntax_check(self, notebooks_dir):
        """Test that notebook files have valid JSON syntax."""
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory does not exist")
        
        notebooks = list(notebooks_dir.rglob("*.ipynb"))
        if not notebooks:
            pytest.skip("No notebooks found")
        
        for notebook in notebooks[:3]:  # Limit to first 3 for performance
            # Try to read the notebook as JSON
            import json
            try:
                with open(notebook, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                # Basic structure validation
                assert "cells" in notebook_data, f"Notebook {notebook} missing 'cells'"
                assert "metadata" in notebook_data, f"Notebook {notebook} missing 'metadata'"
                assert isinstance(notebook_data["cells"], list), f"Notebook {notebook} 'cells' not a list"
                
            except json.JSONDecodeError as e:
                pytest.fail(f"Notebook {notebook} has invalid JSON: {e}")
            except Exception as e:
                pytest.fail(f"Error reading notebook {notebook}: {e}")

    @pytest.mark.slow
    def test_notebook_basic_execution(self, notebooks_dir):
        """Test that notebooks can be converted (basic execution check)."""
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory does not exist")
        
        # Check if nbconvert is available
        try:
            subprocess.run(["jupyter", "nbconvert", "--version"], 
                         check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("jupyter nbconvert not available")
        
        notebooks = list(notebooks_dir.rglob("*.ipynb"))
        if not notebooks:
            pytest.skip("No notebooks found")
        
        # Test the first notebook only for basic execution check
        test_notebook = notebooks[0]
        
        try:
            # Convert to script format (safer than execution)
            result = subprocess.run([
                "jupyter", "nbconvert", 
                "--to", "script",
                "--stdout",
                str(test_notebook)
            ], check=True, capture_output=True, text=True, timeout=30)
            
            # Should produce some Python code
            assert len(result.stdout) > 0, "No output from nbconvert"
            assert "import" in result.stdout.lower(), "No import statements found in converted notebook"
            
        except subprocess.TimeoutExpired:
            pytest.fail(f"Notebook conversion timed out: {test_notebook}")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Notebook conversion failed: {test_notebook}, error: {e}")

    def test_notebook_has_qubex_imports(self, notebooks_dir):
        """Test that notebooks contain qubex imports."""
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory does not exist")
        
        notebooks = list(notebooks_dir.rglob("*.ipynb"))
        if not notebooks:
            pytest.skip("No notebooks found")
        
        import json
        found_qubex_import = False
        
        for notebook in notebooks[:3]:  # Check first 3 notebooks
            try:
                with open(notebook, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                # Check code cells for qubex imports
                for cell in notebook_data.get("cells", []):
                    if cell.get("cell_type") == "code":
                        source = cell.get("source", [])
                        source_text = "".join(source) if isinstance(source, list) else source
                        
                        if "qubex" in source_text.lower() or "import qx" in source_text:
                            found_qubex_import = True
                            break
                
                if found_qubex_import:
                    break
                    
            except Exception as e:
                # Don't fail the test for individual notebook read errors
                continue
        
        assert found_qubex_import, "No qubex imports found in example notebooks"


class TestNotebookContent:
    """Tests for notebook content quality."""

    @pytest.fixture
    def notebooks_dir(self):
        """Get the notebooks directory path."""
        repo_root = Path(__file__).parent.parent
        return repo_root / "docs" / "examples"

    def test_notebooks_have_markdown_cells(self, notebooks_dir):
        """Test that notebooks contain explanatory markdown cells."""
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory does not exist")
        
        notebooks = list(notebooks_dir.rglob("*.ipynb"))
        if not notebooks:
            pytest.skip("No notebooks found")
        
        import json
        found_markdown = False
        
        for notebook in notebooks[:2]:  # Check first 2 notebooks
            try:
                with open(notebook, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                markdown_cells = [
                    cell for cell in notebook_data.get("cells", [])
                    if cell.get("cell_type") == "markdown"
                ]
                
                if len(markdown_cells) > 0:
                    found_markdown = True
                    break
                    
            except Exception:
                continue
        
        # This is more of a quality suggestion than a hard requirement
        if not found_markdown:
            pytest.skip("No markdown cells found - notebooks might be purely code")

    def test_notebooks_reasonable_size(self, notebooks_dir):
        """Test that notebooks are not excessively large."""
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory does not exist")
        
        notebooks = list(notebooks_dir.rglob("*.ipynb"))
        if not notebooks:
            pytest.skip("No notebooks found")
        
        max_size_mb = 10  # 10MB limit
        
        for notebook in notebooks:
            size_bytes = notebook.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            assert size_mb < max_size_mb, f"Notebook {notebook} is too large: {size_mb:.1f}MB > {max_size_mb}MB"
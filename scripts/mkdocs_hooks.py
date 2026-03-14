"""MkDocs build hooks for the Qubex documentation site."""

from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig

from qubex.devtools.api_reference import generate_api_reference_docs


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    """Generate API reference pages before MkDocs scans the docs directory."""
    repo_root = Path(config.config_file_path).parent
    src_dir = repo_root / "src"
    output_dir = Path(config.docs_dir) / "api-reference"
    generate_api_reference_docs(src_dir=src_dir, output_dir=output_dir)
    return config

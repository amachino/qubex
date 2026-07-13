"""Tests for MkDocs API reference generation."""

from pathlib import Path

from qubex.devtools.api_reference import generate_api_reference_docs


def test_generate_api_reference_docs_writes_package_pages_and_summary(
    tmp_path: Path,
) -> None:
    """Generator should create package pages and a literate navigation summary."""
    # Arrange
    src_dir = tmp_path / "src"
    output_dir = tmp_path / "docs" / "api-reference"
    package_dir = src_dir / "sample_pkg"
    nested_dir = package_dir / "analysis"
    nested_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "cli.py").write_text("", encoding="utf-8")
    (nested_dir / "__init__.py").write_text("", encoding="utf-8")
    (nested_dir / "fit_result.py").write_text("", encoding="utf-8")

    # Act
    generate_api_reference_docs(src_dir=src_dir, output_dir=output_dir)

    # Assert
    assert (output_dir / "sample_pkg" / "index.md").read_text(
        encoding="utf-8"
    ) == "::: sample_pkg\n"
    assert (output_dir / "sample_pkg" / "cli.md").read_text(
        encoding="utf-8"
    ) == "::: sample_pkg.cli\n"
    assert (output_dir / "sample_pkg" / "analysis" / "index.md").read_text(
        encoding="utf-8"
    ) == "::: sample_pkg.analysis\n"
    assert (output_dir / "sample_pkg" / "analysis" / "fit_result.md").read_text(
        encoding="utf-8"
    ) == "::: sample_pkg.analysis.fit_result\n"
    assert (output_dir / "summary.md").read_text(encoding="utf-8") == "\n".join(
        [
            "* [sample_pkg](sample_pkg/index.md)",
            "  * [analysis](sample_pkg/analysis/index.md)",
            "    * [fit_result](sample_pkg/analysis/fit_result.md)",
            "  * [cli](sample_pkg/cli.md)",
            "",
        ]
    )


def test_generate_api_reference_docs_removes_stale_generated_files(
    tmp_path: Path,
) -> None:
    """Generator should remove stale files from the output directory."""
    # Arrange
    src_dir = tmp_path / "src"
    output_dir = tmp_path / "docs" / "api-reference"
    package_dir = src_dir / "sample_pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    stale_file = output_dir / "stale.md"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale\n", encoding="utf-8")

    # Act
    generate_api_reference_docs(src_dir=src_dir, output_dir=output_dir)

    # Assert
    assert not stale_file.exists()

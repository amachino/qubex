"""API reference page generation helpers for MkDocs builds."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class _ApiReferenceNode:
    title: str
    doc_path: Path | None = None
    children: dict[str, _ApiReferenceNode] = field(default_factory=dict)


def generate_api_reference_docs(src_dir: Path, output_dir: Path) -> None:
    """Generate MkDocs API reference pages from Python source files."""
    module_files = _collect_module_files(src_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nav_root = _ApiReferenceNode(title="")
    for module_parts, source_file in module_files.items():
        module_name = _get_module_name(module_parts, source_file)
        doc_relative_path = _get_doc_relative_path(module_parts, source_file)
        doc_path = output_dir / doc_relative_path
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(f"::: {module_name}\n", encoding="utf-8")
        _add_nav_entry(nav_root, module_parts, doc_relative_path)

    summary_lines = list(_render_summary_lines(nav_root))
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def _collect_module_files(src_dir: Path) -> dict[tuple[str, ...], Path]:
    """Collect Python modules under the source tree."""
    module_files: dict[tuple[str, ...], Path] = {}

    for source_file in sorted(src_dir.rglob("*.py")):
        if "__pycache__" in source_file.parts:
            continue

        module_parts = source_file.relative_to(src_dir).with_suffix("").parts
        if not module_parts:
            continue

        module_files[module_parts] = source_file

    return module_files


def _get_doc_relative_path(module_parts: tuple[str, ...], source_file: Path) -> Path:
    """Return the markdown path for a Python module."""
    if source_file.stem == "__init__":
        package_parts = module_parts[:-1]
        if not package_parts:
            msg = "Top-level __init__.py is not supported for API reference generation."
            raise ValueError(msg)
        return Path(*package_parts) / "index.md"

    return Path(*module_parts).with_suffix(".md")


def _get_module_name(module_parts: tuple[str, ...], source_file: Path) -> str:
    """Return the mkdocstrings module identifier for a Python file."""
    if source_file.stem == "__init__":
        return ".".join(module_parts[:-1])

    return ".".join(module_parts)


def _add_nav_entry(
    root: _ApiReferenceNode, module_parts: tuple[str, ...], doc_path: Path
) -> None:
    """Add a documentation page to the summary tree."""
    current = root
    normalized_parts = (
        module_parts[:-1] if module_parts[-1] == "__init__" else module_parts
    )

    for part in normalized_parts:
        current = current.children.setdefault(part, _ApiReferenceNode(title=part))

    current.doc_path = doc_path


def _render_summary_lines(root: _ApiReferenceNode) -> list[str]:
    """Render literate navigation lines for generated API pages."""
    return list(_walk_summary(root, depth=0))


def _walk_summary(node: _ApiReferenceNode, depth: int) -> list[str]:
    """Walk the navigation tree and render bullet items."""
    lines: list[str] = []
    for child in sorted(node.children.values(), key=_nav_sort_key):
        if child.doc_path is None:
            continue

        indent = "  " * depth
        lines.append(f"{indent}* [{child.title}]({child.doc_path.as_posix()})")
        lines.extend(_walk_summary(child, depth + 1))

    return lines


def _nav_sort_key(node: _ApiReferenceNode) -> tuple[int, str]:
    """Sort packages before modules and keep names stable."""
    is_package = int(not (node.doc_path and node.doc_path.name == "index.md"))
    return (is_package, node.title)

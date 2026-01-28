from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files


def main() -> None:
    """Generate API reference pages for all modules under ``src/qubex``."""
    nav = mkdocs_gen_files.Nav()  # pyright: ignore[reportPrivateImportUsage]

    package_root = Path("src/qubex")
    for path in sorted(package_root.rglob("*.py")):
        if "egg-info" in path.parts or "__pycache__" in path.parts:
            continue

        module_path = path.relative_to("src").with_suffix("")
        parts = list(module_path.parts)

        if parts[-1] == "__init__":
            parts.pop()
            doc_path = Path("reference").joinpath(*parts, "index.md")
            identifier = ".".join(parts) if parts else "qubex"
            nav_parts = [*parts, "index"]
        else:
            doc_path = Path("reference").joinpath(*parts).with_suffix(".md")
            identifier = ".".join(parts)
            nav_parts = parts

        nav[tuple(nav_parts)] = doc_path.as_posix()

        with mkdocs_gen_files.open(doc_path, "w") as file:
            file.write(f"# {identifier}\n\n")
            file.write(f"::: {identifier}\n")

    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


main()

"""MkDocs build hooks for the Qubex documentation site."""

import re
from collections.abc import Callable
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.livereload import LiveReloadServer
from mkdocs.plugins import CombinedEvent, event_priority
from mkdocs.structure.files import File, Files, InclusionLevel
from mkdocs.structure.pages import Page
from mkdocs_jupyter.plugin import NotebookFile

from qubex.devtools.api_reference import build_api_reference_documents

_GENERATED_API_REFERENCE_DOCUMENTS: dict[str, str] = {}


@event_priority(100)
def _on_files_add_api_reference(files: Files, config: MkDocsConfig) -> Files:
    """Add generated API reference pages to the MkDocs file collection."""
    repo_root = Path(config.config_file_path).parent
    src_dir = repo_root / "src"
    generated_documents = build_api_reference_documents(src_dir=src_dir)
    _GENERATED_API_REFERENCE_DOCUMENTS.clear()
    _GENERATED_API_REFERENCE_DOCUMENTS.update(generated_documents)

    for src_uri in generated_documents:
        existing_file = files.get_file_from_path(src_uri)
        if existing_file is not None:
            files.remove(existing_file)
        generated_file = File(
            src_uri,
            src_dir=config.docs_dir,
            dest_dir=config.site_dir,
            use_directory_urls=config.use_directory_urls,
            inclusion=InclusionLevel.NOT_IN_NAV,
        )
        files.append(generated_file)

    return files


@event_priority(-200)
def _on_files_restore_notebooks(files: Files, config: MkDocsConfig) -> Files:
    """Re-wrap notebook files after mkdocs-static-i18n recreates file objects."""
    jupyter_plugin = config.plugins.get("mkdocs-jupyter")
    if jupyter_plugin is None:
        return files

    files_to_wrap = [
        file
        for file in files
        if jupyter_plugin.should_include(file) and not isinstance(file, NotebookFile)
    ]
    for file in files_to_wrap:
        files.remove(file)
        files.append(NotebookFile(file, **config))
    return files


on_files = CombinedEvent(_on_files_add_api_reference, _on_files_restore_notebooks)


_MATHJAX_V2_PATTERN = re.compile(
    r"<!-- Load mathjax -->.*?<!-- End of mathjax configuration -->",
    re.DOTALL,
)

_DISPLAY_MATH_PATTERN = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


def _wrap_notebook_math(html: str) -> str:
    """
    Wrap $...$ and $$...$$ in notebook HTML with arithmatex spans.

    nbconvert emits bare $...$ delimiters that MathJax cannot find because
    the global ignoreHtmlClass pattern blocks classless elements. Wrapping
    them in <span class="arithmatex"> makes them visible to MathJax, just
    like pymdownx.arithmatex does for regular Markdown pages.
    """
    html = _DISPLAY_MATH_PATTERN.sub(r'<span class="arithmatex">\[\1\]</span>', html)
    html = _INLINE_MATH_PATTERN.sub(r'<span class="arithmatex">\(\1\)</span>', html)
    return html


def on_page_content(
    html: str, *, page: Page, config: MkDocsConfig, files: Files
) -> str:
    """Post-process notebook HTML for MathJax v3 compatibility."""
    html = _MATHJAX_V2_PATTERN.sub("", html)
    if isinstance(page.file, NotebookFile):
        html = _wrap_notebook_math(html)
    return html


def on_page_read_source(*, page: Page, config: MkDocsConfig) -> str | None:
    """Provide generated API reference content without writing files to docs_dir."""
    return _GENERATED_API_REFERENCE_DOCUMENTS.get(page.file.src_uri)


def on_serve(
    server: LiveReloadServer,
    *,
    config: MkDocsConfig,
    builder: Callable[..., None],
) -> LiveReloadServer:
    """Watch the source tree so API docs rebuild during mkdocs serve."""
    repo_root = Path(config.config_file_path).parent
    server.watch(str(repo_root / "src"))
    return server

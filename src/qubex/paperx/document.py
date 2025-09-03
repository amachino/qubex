"""Document generation for paperx framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from .config import PaperConfig
from .references import ReferenceManager
from .templates import TemplateManager

PathLike = Union[str, Path]


class DocumentGenerator:
    """Main document generator for the paperx framework.
    
    Orchestrates template management, reference handling, and document
    generation to produce complete LaTeX documents from content and
    configuration.
    """
    
    def __init__(
        self,
        config: PaperConfig | None = None,
        *,
        config_dir: PathLike | None = None,
        template_name: str = "default",
    ) -> None:
        """Initialize document generator.
        
        Parameters
        ----------
        config : PaperConfig, optional
            Configuration instance. If None, creates a new one.
        config_dir : PathLike, optional
            Configuration directory. Used only if config is None.
        template_name : str, optional
            Default template name, by default "default".
        """
        if config is None:
            config = PaperConfig(
                config_dir=config_dir,
                template_name=template_name,
            )
        
        self.config = config
        self.template_manager = TemplateManager(config)
        self.reference_manager = ReferenceManager(config)
    
    def generate_document(
        self,
        content: str,
        *,
        output_path: PathLike,
        references: list[str] | None = None,
        template_name: str | None = None,
        title: str | None = None,
        author: str | None = None,
        date: str | None = None,
        abstract: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Generate a complete LaTeX document.
        
        Parameters
        ----------
        content : str
            Main document content (LaTeX body).
        output_path : PathLike
            Output file path for the generated document.
        references : list[str], optional
            List of reference keys to include. If None, no references.
        template_name : str, optional
            Template to use. If None, uses default template.
        title : str, optional
            Document title.
        author : str, optional
            Document author(s).
        date : str, optional
            Document date.
        abstract : str, optional
            Document abstract.
        metadata : dict[str, Any], optional
            Additional metadata for the document.
            
        Returns
        -------
        Path
            Path to the generated LaTeX file.
        """
        output_path = Path(output_path)
        
        # Generate document parts
        preamble = self.template_manager.generate_preamble(template_name)
        document_content = self._generate_document_content(
            content=content,
            title=title,
            author=author,
            date=date,
            abstract=abstract,
            template_name=template_name,
            metadata=metadata,
        )
        
        # Generate references if requested
        bibliography = ""
        if references:
            bibliography = self._generate_bibliography(references)
        
        # Combine all parts
        full_document = self._combine_document_parts(
            preamble=preamble,
            content=document_content,
            bibliography=bibliography,
        )
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_document)
        
        # Also generate .bib file if references are used
        if references:
            bib_path = output_path.with_suffix(".bib")
            bib_content = self.reference_manager.generate_bibtex(references)
            with open(bib_path, "w", encoding="utf-8") as f:
                f.write(bib_content)
        
        return output_path
    
    def _generate_document_content(
        self,
        content: str,
        title: str | None = None,
        author: str | None = None,
        date: str | None = None,
        abstract: str | None = None,
        template_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate the main document content section.
        
        Parameters
        ----------
        content : str
            Main document content.
        title : str, optional
            Document title.
        author : str, optional
            Document author(s).
        date : str, optional
            Document date.
        abstract : str, optional
            Document abstract.
        template_name : str, optional
            Template name for structure guidance.
        metadata : dict[str, Any], optional
            Additional metadata.
            
        Returns
        -------
        str
            Generated document content section.
        """
        content_parts = []
        
        # Document metadata
        if title:
            content_parts.append(f"\\title{{{title}}}")
        if author:
            content_parts.append(f"\\author{{{author}}}")
        if date:
            content_parts.append(f"\\date{{{date}}}")
        elif date is None:
            content_parts.append("\\date{\\today}")
        
        # Begin document
        content_parts.append("")
        content_parts.append("\\begin{document}")
        content_parts.append("")
        
        # Title and front matter
        if title or author or date:
            content_parts.append("\\maketitle")
            content_parts.append("")
        
        # Abstract
        if abstract:
            content_parts.append("\\begin{abstract}")
            content_parts.append(abstract)
            content_parts.append("\\end{abstract}")
            content_parts.append("")
        
        # Main content
        content_parts.append(content)
        
        return "\n".join(content_parts)
    
    def _generate_bibliography(self, references: list[str]) -> str:
        """Generate bibliography section.
        
        Parameters
        ----------
        references : list[str]
            List of reference keys.
            
        Returns
        -------
        str
            Bibliography LaTeX content.
        """
        if not references:
            return ""
        
        bib_parts = []
        bib_parts.append("")
        bib_parts.append("% Bibliography")
        bib_parts.append("\\bibliographystyle{plain}")
        bib_parts.append("\\bibliography{\\jobname}")  # Use same name as main file
        
        return "\n".join(bib_parts)
    
    def _combine_document_parts(
        self,
        preamble: str,
        content: str,
        bibliography: str = "",
    ) -> str:
        """Combine all document parts into final LaTeX document.
        
        Parameters
        ----------
        preamble : str
            Document preamble.
        content : str
            Document content.
        bibliography : str, optional
            Bibliography section.
            
        Returns
        -------
        str
            Complete LaTeX document.
        """
        parts = [preamble, content]
        
        if bibliography:
            parts.append(bibliography)
        
        parts.extend(["", "\\end{document}"])
        
        return "\n".join(parts)
    
    def create_project(
        self,
        project_dir: PathLike,
        template_name: str = "default",
    ) -> Path:
        """Create a new paperx project directory with example files.
        
        Parameters
        ----------
        project_dir : PathLike
            Directory to create the project in.
        template_name : str, optional
            Template to use for the project, by default "default".
            
        Returns
        -------
        Path
            Path to the created project directory.
        """
        project_path = Path(project_dir)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create main content file
        content_file = project_path / "content.tex"
        sample_content = """\\section{Introduction}

This is a sample introduction section. Replace this with your actual content.

\\section{Methods}

Describe your methods here.

\\section{Results}

Present your results here.

\\section{Conclusion}

Conclude your paper here."""
        
        with open(content_file, "w", encoding="utf-8") as f:
            f.write(sample_content)
        
        # Create paper configuration file
        paper_config = project_path / "paper.yaml"
        config_content = f"""# Paper configuration for paperx
template: {template_name}
title: "Sample Paper Title"
author: "Your Name"
abstract: |
  This is a sample abstract. Replace this with your actual abstract
  that summarizes your paper's contributions and findings.

# References to include (keys from your reference database)
references:
  - sample_ref_1
  - sample_ref_2

# Additional metadata
keywords:
  - keyword1
  - keyword2
  - keyword3
"""
        
        with open(paper_config, "w", encoding="utf-8") as f:
            f.write(config_content)
        
        # Create README
        readme_file = project_path / "README.md"
        readme_content = f"""# Paperx Project

This is a paperx project using the `{template_name}` template.

## Files

- `content.tex`: Main paper content (edit this file)
- `paper.yaml`: Paper configuration (title, author, references, etc.)
- `README.md`: This file

## Usage

1. Edit `content.tex` with your paper content
2. Update `paper.yaml` with your paper metadata
3. Generate the document using paperx

## Configuration

The paperx framework separates content from configuration:
- Content goes in `content.tex` 
- Document structure, preambles, and macros are handled by templates
- References are managed separately and selected per paper
"""
        
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        return project_path
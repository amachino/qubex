"""Example demonstrating the paperx LaTeX framework."""

from pathlib import Path
from tempfile import TemporaryDirectory

from qubex.paperx import DocumentGenerator, PaperConfig


def main():
    """Demonstrate paperx framework usage."""
    # Create a temporary directory for this example
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Running paperx example in: {temp_path}")
        
        # Set up configuration directory
        config_dir = temp_path / "config"
        config_dir.mkdir()
        
        # Create example configurations
        setup_example_configs(config_dir)
        
        # Create a sample project
        project_dir = temp_path / "sample_paper"
        create_sample_project(config_dir, project_dir)
        
        # Generate the document
        generate_sample_document(config_dir, project_dir)
        
        print("\nPaperx framework example completed successfully!")
        print(f"Check the generated files in: {project_dir}")


def setup_example_configs(config_dir: Path):
    """Set up example configuration files."""
    import yaml
    
    print("Setting up example configurations...")
    
    # Preamble configuration
    preamble_config = {
        "packages": [
            "amsmath",
            "amsfonts", 
            "amssymb",
            "graphicx",
            {"name": "geometry", "options": ["margin=1in"]},
            {"name": "hyperref", "options": ["colorlinks=true"]},
        ]
    }
    
    with open(config_dir / "preamble.yaml", "w") as f:
        yaml.dump(preamble_config, f)
    
    # Macros configuration
    macros_config = {
        "commands": {
            "qubex": "\\texttt{qubex}",
            "paperx": "\\texttt{paperx}",
            "ket": {"args": 1, "definition": "|#1\\rangle"},
            "bra": {"args": 1, "definition": "\\langle#1|"},
        },
        "environments": {
            "theorem": {
                "begin": "\\textbf{Theorem.} \\itshape",
                "end": "\\upshape"
            }
        },
        "direct_latex": [
            "\\newcommand{\\expect}[1]{\\langle#1\\rangle}",
        ]
    }
    
    with open(config_dir / "macros.yaml", "w") as f:
        yaml.dump(macros_config, f)
    
    # References configuration
    references_config = {
        "entries": {
            "paperx2024": {
                "type": "misc",
                "title": "Paperx: A LaTeX Framework for Academic Writing",
                "author": "Qubex Contributors",
                "year": "2024",
                "note": "LaTeX framework implementation"
            },
            "latex_guide": {
                "type": "book",
                "title": "The LaTeX Companion",
                "author": "Mittelbach, F. and Goossens, M.",
                "publisher": "Addison-Wesley",
                "year": "2004",
                "edition": "2nd"
            }
        }
    }
    
    with open(config_dir / "references.yaml", "w") as f:
        yaml.dump(references_config, f)
    
    # Template configuration
    template_config = {
        "documentclass": "article",
        "documentclass_options": ["11pt", "a4paper"],
        "structure": {
            "sections": ["Introduction", "Framework Design", "Example Usage", "Conclusion"],
            "include_abstract": True
        },
        "preamble_additions": [
            "% Paperx example template",
            "\\setcounter{secnumdepth}{3}"
        ]
    }
    
    with open(config_dir / "template_default.yaml", "w") as f:
        yaml.dump(template_config, f)
    
    print("✓ Configuration files created")


def create_sample_project(config_dir: Path, project_dir: Path):
    """Create a sample paperx project."""
    print("Creating sample project...")
    
    # Create document generator
    doc_gen = DocumentGenerator(config_dir=config_dir)
    
    # Create project structure
    doc_gen.create_project(project_dir)
    
    # Create custom content
    content = """\\section{Introduction}

The \\paperx framework provides a clean separation between content and 
configuration for \\LaTeX document generation. This example demonstrates
the key features of the framework.

\\section{Framework Design}

The \\paperx framework consists of several key components:

\\begin{itemize}
\\item Configuration management for templates and macros
\\item Reference management with per-paper selection
\\item Template-based document generation
\\item Command-line interface for project management
\\end{itemize}

\\subsection{Quantum Computing Integration}

Since this framework is part of the \\qubex project, it includes
quantum computing specific macros like \\ket{0}, \\ket{1}, and 
\\ket{\\psi} = \\alpha\\ket{0} + \\beta\\ket{1}.

\\begin{theorem}
The \\paperx framework enables reproducible document generation
by separating content from formatting concerns.
\\end{theorem}

\\section{Example Usage}

Using \\paperx is straightforward:

\\begin{enumerate}
\\item Create a project with configuration files
\\item Write content in \\LaTeX
\\item Generate the final document
\\end{enumerate}

The framework handles preambles, macros, and references automatically
based on the configuration files \\cite{paperx2024}.

\\section{Conclusion}

The \\paperx framework demonstrates how configuration-driven approaches
can simplify academic writing workflows. For more information on \\LaTeX
best practices, see \\cite{latex_guide}.
"""
    
    # Write custom content
    with open(project_dir / "content.tex", "w") as f:
        f.write(content)
    
    # Update project configuration
    import yaml
    paper_config = {
        "template": "default",
        "title": "Paperx Framework Example",
        "author": "Qubex Team",
        "abstract": """
This document demonstrates the paperx framework, a LaTeX-based system 
for academic writing that separates content from configuration. The 
framework enables authors to focus on content while maintaining 
consistent formatting and managing references efficiently.
        """.strip(),
        "references": ["paperx2024", "latex_guide"]
    }
    
    with open(project_dir / "paper.yaml", "w") as f:
        yaml.dump(paper_config, f)
    
    print("✓ Sample project created")


def generate_sample_document(config_dir: Path, project_dir: Path):
    """Generate the sample document."""
    print("Generating sample document...")
    
    # Load project configuration
    import yaml
    with open(project_dir / "paper.yaml", "r") as f:
        paper_config = yaml.safe_load(f)
    
    # Load content
    with open(project_dir / "content.tex", "r") as f:
        content = f.read()
    
    # Create document generator
    doc_gen = DocumentGenerator(config_dir=config_dir)
    
    # Generate document
    output_path = project_dir / "sample_paper.tex"
    result_path = doc_gen.generate_document(
        content=content,
        output_path=output_path,
        title=paper_config["title"],
        author=paper_config["author"],
        abstract=paper_config["abstract"],
        references=paper_config["references"],
    )
    
    print(f"✓ Generated LaTeX document: {result_path}")
    
    # Display the generated preamble (first 30 lines)
    with open(result_path, "r") as f:
        lines = f.readlines()
    
    print("\nGenerated document preview (first 30 lines):")
    print("-" * 50)
    for i, line in enumerate(lines[:30]):
        print(f"{i+1:2d}: {line.rstrip()}")
    if len(lines) > 30:
        print(f"... ({len(lines) - 30} more lines)")
    print("-" * 50)


if __name__ == "__main__":
    main()
# Paperx: LaTeX Paper Writing Framework

Paperx is a LaTeX-based framework for academic writing that separates content from configuration, allowing authors to focus on content while maintaining consistent formatting and managing references efficiently.

## Overview

The paperx framework provides:

- **Separation of Concerns**: Content, formatting, and configuration are cleanly separated
- **Template Management**: Reusable document templates with configurable preambles and macros
- **Reference Management**: Structured reference databases with per-paper selection
- **Configuration-Driven**: YAML-based configuration files for easy customization
- **CLI Interface**: Command-line tools for project management and document generation

## Quick Start

### Basic Usage

```python
import qubex.paperx as paperx

# Create a document generator
doc_gen = paperx.DocumentGenerator()

# Generate a simple document
doc_gen.generate_document(
    content="\\section{Introduction}\n\nThis is my paper content.",
    output_path="my_paper.tex",
    title="My Paper Title",
    author="Author Name",
)
```

### Project-Based Workflow

```python
# Create a new project
doc_gen = paperx.DocumentGenerator()
project_path = doc_gen.create_project("my_project")

# Edit content.tex and paper.yaml files
# Then build the document
# (This can also be done via CLI)
```

### CLI Usage

```bash
# Create a new project
python -m qubex.paperx.cli create my_project

# Build the document
python -m qubex.paperx.cli build my_project

# Validate configuration
python -m qubex.paperx.cli validate

# List available references
python -m qubex.paperx.cli list-refs
```

## Configuration

### Directory Structure

```
~/.paperx/config/          # Default configuration directory
├── preamble.yaml          # Package and preamble settings
├── macros.yaml           # Custom LaTeX commands and environments
├── references.yaml       # Reference database
├── template_default.yaml # Default template
└── template_ieee.yaml    # IEEE template (example)
```

### Configuration Files

#### preamble.yaml

```yaml
packages:
  - amsmath
  - amsfonts
  - name: geometry
    options:
      - margin=1in
  - name: hyperref
    options:
      - colorlinks=true
```

#### macros.yaml

```yaml
commands:
  qubex: "\\texttt{qubex}"
  ket:
    args: 1
    definition: "|#1\\rangle"

environments:
  theorem:
    begin: "\\textbf{Theorem.} \\itshape"
    end: "\\upshape"

direct_latex:
  - "\\newcommand{\\expect}[1]{\\langle#1\\rangle}"
```

#### references.yaml

```yaml
entries:
  paperx2024:
    type: "misc"
    title: "Paperx Framework"
    author: "Qubex Team"
    year: "2024"

collections:
  quantum_computing:
    - nielsen2010
    - preskill2018
```

#### template_default.yaml

```yaml
documentclass: "article"
documentclass_options:
  - "11pt"
  - "a4paper"

structure:
  sections:
    - "Introduction"
    - "Methods"
    - "Results"
    - "Conclusion"
  include_abstract: true

preamble_additions:
  - "% Custom template additions"
  - "\\setcounter{secnumdepth}{3}"
```

## Project Structure

A paperx project contains:

```
my_project/
├── content.tex           # Main paper content (LaTeX)
├── paper.yaml           # Paper-specific configuration
├── README.md            # Project documentation
└── my_project.tex       # Generated LaTeX document (after build)
└── my_project.bib       # Generated bibliography (if references used)
```

### paper.yaml Example

```yaml
template: "default"
title: "My Research Paper"
author: "First Author and Second Author"
abstract: |
  This paper presents novel research in quantum computing
  applications using the qubex framework.

references:
  - paperx2024
  - quantum_ref_1
  - quantum_ref_2

keywords:
  - quantum computing
  - qubex
  - paperx
```

## API Reference

### Core Classes

#### DocumentGenerator

Main class for generating LaTeX documents.

```python
class DocumentGenerator:
    def __init__(self, config=None, *, config_dir=None, template_name="default"):
        """Initialize document generator."""
    
    def generate_document(self, content, *, output_path, references=None, 
                         title=None, author=None, abstract=None, **kwargs):
        """Generate complete LaTeX document."""
    
    def create_project(self, project_dir, template_name="default"):
        """Create new paperx project directory."""
```

#### PaperConfig

Configuration management class.

```python
class PaperConfig:
    def __init__(self, *, config_dir=None, template_name="default", 
                 preamble_file="preamble.yaml", **kwargs):
        """Initialize configuration loader."""
    
    @property
    def preamble_config(self) -> dict:
        """Get preamble configuration."""
    
    @property
    def macros_config(self) -> dict:
        """Get macros configuration."""
    
    @property
    def references_config(self) -> dict:
        """Get references configuration."""
```

#### TemplateManager

Template and macro management.

```python
class TemplateManager:
    def generate_preamble(self, template_name=None) -> str:
        """Generate LaTeX preamble from configuration."""
    
    def generate_macros(self) -> str:
        """Generate LaTeX macros from configuration."""
    
    def list_available_templates(self) -> list[str]:
        """Get list of available templates."""
```

#### ReferenceManager

Reference database management.

```python
class ReferenceManager:
    def get_references_for_paper(self, paper_refs) -> dict:
        """Get references selected for a specific paper."""
    
    def generate_bibtex(self, paper_refs) -> str:
        """Generate BibTeX content for selected references."""
    
    def search_references(self, query) -> list[str]:
        """Search references by content."""
```

## Examples

### Academic Paper

```python
import qubex.paperx as paperx

# Set up configuration with quantum computing macros
config = paperx.PaperConfig(template_name="default")
doc_gen = paperx.DocumentGenerator(config)

content = """
\\section{Introduction}

Quantum computing represents a paradigm shift in computational capabilities.
Using the \\qubex framework, we can express quantum states like \\ket{0} 
and \\ket{1}.

\\section{Methodology}

Our approach utilizes quantum superposition: 
\\ket{\\psi} = \\alpha\\ket{0} + \\beta\\ket{1}

The expectation value is computed as \\expect{\\hat{H}}.
"""

doc_gen.generate_document(
    content=content,
    output_path="quantum_paper.tex",
    title="Quantum Computing with Qubex",
    author="Research Team",
    abstract="This paper demonstrates quantum computing applications...",
    references=["nielsen2010", "qubex2024"],
)
```

### Conference Paper (IEEE Format)

```python
# Use IEEE template
config = paperx.PaperConfig(template_name="ieee")
doc_gen = paperx.DocumentGenerator(config)

# Generate IEEE-formatted paper
doc_gen.generate_document(
    content=conference_content,
    output_path="conference_paper.tex",
    title="Quantum Error Correction Advances",
    author="A. Researcher\\thanks{University affiliation}",
    references=["quantum_refs"],
)
```

## Design Principles

The paperx framework follows several key principles from the qubex project:

1. **Separation of Concerns**: Content and configuration are cleanly separated
2. **Configuration-Driven**: YAML files define document structure and formatting
3. **Reproducibility**: Deterministic document generation from configuration
4. **Modularity**: Independent components for templates, references, and macros
5. **Extensibility**: Easy to add new templates and macros

## Integration with Qubex

Paperx is designed as part of the qubex ecosystem:

- Uses qubex coding standards and patterns
- Follows qubex configuration management approaches
- Includes quantum computing specific macros and templates
- Integrates with qubex documentation workflows

## Best Practices

### Content Organization

1. Keep content in separate `.tex` files
2. Use descriptive filenames and project names
3. Organize references in logical collections
4. Document custom macros and their purposes

### Configuration Management

1. Start with provided templates and customize as needed
2. Use version control for configuration files
3. Share template configurations across projects
4. Validate configurations before building documents

### Reference Management

1. Maintain a central reference database
2. Use descriptive reference keys
3. Include all necessary bibliography information
4. Organize references by topic or project

## Troubleshooting

### Common Issues

1. **Configuration file not found**: Ensure files exist in the correct directory
2. **Invalid YAML syntax**: Validate YAML files using online tools
3. **Missing references**: Check reference keys match database entries
4. **Template errors**: Validate template configuration with `paperx validate`

### Debugging

Use the CLI validation command to check configurations:

```bash
python -m qubex.paperx.cli validate --config-dir /path/to/config
```

## Contributing

The paperx framework follows qubex contribution guidelines:

1. Add tests for new functionality
2. Follow existing code style and patterns
3. Update documentation for API changes
4. Use minimal, focused changes

## License

Part of the qubex project. See project LICENSE file for details.
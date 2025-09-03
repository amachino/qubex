"""Command-line interface for paperx framework."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .config import PaperConfig
from .document import DocumentGenerator


def create_project_command(args: argparse.Namespace) -> int:
    """Create a new paperx project."""
    project_path = Path(args.project_dir)
    
    if project_path.exists() and any(project_path.iterdir()):
        print(f"Error: Directory {project_path} already exists and is not empty")
        return 1
    
    try:
        doc_gen = DocumentGenerator(
            config_dir=args.config_dir,
            template_name=args.template,
        )
        
        created_path = doc_gen.create_project(
            project_dir=project_path,
            template_name=args.template,
        )
        
        print(f"Created paperx project in: {created_path}")
        print("\nNext steps:")
        print(f"1. Edit {created_path / 'content.tex'} with your paper content")
        print(f"2. Update {created_path / 'paper.yaml'} with your paper metadata")
        print(f"3. Generate your document with: paperx build {created_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error creating project: {e}")
        return 1


def build_document_command(args: argparse.Namespace) -> int:
    """Build a LaTeX document from paperx project."""
    project_path = Path(args.project_dir)
    
    if not project_path.exists():
        print(f"Error: Project directory {project_path} does not exist")
        return 1
    
    content_file = project_path / "content.tex"
    config_file = project_path / "paper.yaml"
    
    if not content_file.exists():
        print(f"Error: Content file {content_file} not found")
        return 1
    
    try:
        # Load project configuration
        paper_config = {}
        if config_file.exists():
            with open(config_file, "r") as f:
                paper_config = yaml.safe_load(f) or {}
        
        # Load content
        with open(content_file, "r") as f:
            content = f.read()
        
        # Create document generator
        doc_gen = DocumentGenerator(
            config_dir=args.config_dir,
            template_name=paper_config.get("template", args.template),
        )
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = project_path / f"{project_path.name}.tex"
        
        # Generate document
        result_path = doc_gen.generate_document(
            content=content,
            output_path=output_path,
            title=paper_config.get("title"),
            author=paper_config.get("author"),
            date=paper_config.get("date"),
            abstract=paper_config.get("abstract"),
            references=paper_config.get("references", []),
            metadata=paper_config,
        )
        
        print(f"Generated LaTeX document: {result_path}")
        
        # Also generate .bib file if references exist
        if paper_config.get("references"):
            bib_path = result_path.with_suffix(".bib")
            print(f"Generated bibliography file: {bib_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error building document: {e}")
        return 1


def validate_config_command(args: argparse.Namespace) -> int:
    """Validate paperx configuration."""
    try:
        config = PaperConfig(config_dir=args.config_dir)
        
        print("Validating paperx configuration...")
        
        # Check if config files exist and are valid
        try:
            _ = config.preamble_config
            print("✓ Preamble configuration is valid")
        except Exception as e:
            print(f"✗ Preamble configuration error: {e}")
            return 1
        
        try:
            _ = config.macros_config
            print("✓ Macros configuration is valid")
        except Exception as e:
            print(f"✗ Macros configuration error: {e}")
            return 1
        
        try:
            _ = config.references_config
            print("✓ References configuration is valid")
        except Exception as e:
            print(f"✗ References configuration error: {e}")
            return 1
        
        # Validate templates
        from .templates import TemplateManager
        template_manager = TemplateManager(config)
        
        templates = template_manager.list_available_templates()
        if templates:
            print(f"✓ Found {len(templates)} template(s): {', '.join(templates)}")
            
            for template_name in templates:
                is_valid, errors = template_manager.validate_template(template_name)
                if is_valid:
                    print(f"  ✓ Template '{template_name}' is valid")
                else:
                    print(f"  ✗ Template '{template_name}' has errors:")
                    for error in errors:
                        print(f"    - {error}")
        else:
            print("⚠ No templates found")
        
        print("\nConfiguration validation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return 1


def list_references_command(args: argparse.Namespace) -> int:
    """List available references."""
    try:
        config = PaperConfig(config_dir=args.config_dir)
        from .references import ReferenceManager
        
        ref_manager = ReferenceManager(config)
        available_refs = ref_manager.list_available_references()
        
        if not available_refs:
            print("No references found in the database")
            return 0
        
        print(f"Found {len(available_refs)} references:")
        print()
        
        for ref_key in available_refs:
            ref_data = ref_manager._reference_database[ref_key]
            ref_type = ref_data.get("type", "unknown")
            title = ref_data.get("title", "No title")
            author = ref_data.get("author", "No author")
            
            print(f"• {ref_key} ({ref_type})")
            print(f"  Title: {title}")
            print(f"  Author: {author}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error listing references: {e}")
        return 1


def search_references_command(args: argparse.Namespace) -> int:
    """Search references."""
    try:
        config = PaperConfig(config_dir=args.config_dir)
        from .references import ReferenceManager
        
        ref_manager = ReferenceManager(config)
        results = ref_manager.search_references(args.query)
        
        if not results:
            print(f"No references found matching '{args.query}'")
            return 0
        
        print(f"Found {len(results)} reference(s) matching '{args.query}':")
        print()
        
        for ref_key in results:
            ref_data = ref_manager._reference_database[ref_key]
            ref_type = ref_data.get("type", "unknown")
            title = ref_data.get("title", "No title")
            author = ref_data.get("author", "No author")
            
            print(f"• {ref_key} ({ref_type})")
            print(f"  Title: {title}")
            print(f"  Author: {author}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error searching references: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="paperx",
        description="LaTeX paper writing framework with separated content and configuration",
    )
    
    # Global options
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Configuration directory (default: ~/.paperx/config)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create project command
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new paperx project",
    )
    create_parser.add_argument(
        "project_dir",
        type=Path,
        help="Directory to create the project in",
    )
    create_parser.add_argument(
        "--template",
        default="default",
        help="Template to use (default: default)",
    )
    create_parser.set_defaults(func=create_project_command)
    
    # Build document command
    build_parser = subparsers.add_parser(
        "build",
        help="Build LaTeX document from paperx project",
    )
    build_parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory to build",
    )
    build_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output LaTeX file path",
    )
    build_parser.add_argument(
        "--template",
        default="default",
        help="Template to use if not specified in project config",
    )
    build_parser.set_defaults(func=build_document_command)
    
    # Validate configuration command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate paperx configuration",
    )
    validate_parser.set_defaults(func=validate_config_command)
    
    # List references command
    list_refs_parser = subparsers.add_parser(
        "list-refs",
        help="List available references",
    )
    list_refs_parser.set_defaults(func=list_references_command)
    
    # Search references command
    search_refs_parser = subparsers.add_parser(
        "search-refs",
        help="Search references",
    )
    search_refs_parser.add_argument(
        "query",
        help="Search query",
    )
    search_refs_parser.set_defaults(func=search_references_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
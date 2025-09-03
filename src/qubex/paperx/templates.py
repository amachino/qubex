"""Template management for paperx framework."""

from __future__ import annotations

from typing import Any

from .config import PaperConfig


class TemplateManager:
    """Manages LaTeX templates and macros.
    
    Handles template loading, preamble generation, and macro management
    to separate document structure from content.
    """
    
    def __init__(self, config: PaperConfig) -> None:
        """Initialize template manager.
        
        Parameters
        ----------
        config : PaperConfig
            Configuration instance containing template settings.
        """
        self.config = config
        self._templates: dict[str, dict[str, Any]] = {}
    
    def load_template(self, template_name: str | None = None) -> dict[str, Any]:
        """Load a specific template configuration.
        
        Parameters
        ----------
        template_name : str, optional
            Name of the template to load. If None, uses default template.
            
        Returns
        -------
        dict[str, Any]
            Template configuration.
        """
        if template_name is None:
            template_name = self.config.template_name
            
        if template_name not in self._templates:
            self._templates[template_name] = self.config.get_template_config(template_name)
            
        return self._templates[template_name]
    
    def generate_preamble(self, template_name: str | None = None) -> str:
        """Generate LaTeX preamble from configuration.
        
        Parameters
        ----------
        template_name : str, optional
            Name of the template. If None, uses default template.
            
        Returns
        -------
        str
            Generated LaTeX preamble.
        """
        # Load template and preamble configurations
        template_config = self.load_template(template_name)
        preamble_config = self.config.preamble_config
        
        preamble_parts = []
        
        # Document class
        doc_class = template_config.get("documentclass", "article")
        doc_options = template_config.get("documentclass_options", [])
        if doc_options:
            options_str = ",".join(doc_options)
            preamble_parts.append(f"\\documentclass[{options_str}]{{{doc_class}}}")
        else:
            preamble_parts.append(f"\\documentclass{{{doc_class}}}")
        
        preamble_parts.append("")  # Empty line
        
        # Packages
        packages = preamble_config.get("packages", [])
        for package in packages:
            if isinstance(package, str):
                preamble_parts.append(f"\\usepackage{{{package}}}")
            elif isinstance(package, dict):
                pkg_name = package.get("name", "")
                options = package.get("options", [])
                if options:
                    options_str = ",".join(options)
                    preamble_parts.append(f"\\usepackage[{options_str}]{{{pkg_name}}}")
                else:
                    preamble_parts.append(f"\\usepackage{{{pkg_name}}}")
        
        if packages:
            preamble_parts.append("")  # Empty line after packages
        
        # Custom commands and macros
        macros = self.generate_macros()
        if macros:
            preamble_parts.append("% Custom macros")
            preamble_parts.append(macros)
            preamble_parts.append("")
        
        # Template-specific preamble additions
        if "preamble_additions" in template_config:
            preamble_parts.append("% Template-specific additions")
            preamble_parts.extend(template_config["preamble_additions"])
            preamble_parts.append("")
        
        return "\n".join(preamble_parts)
    
    def generate_macros(self) -> str:
        """Generate LaTeX macros from configuration.
        
        Returns
        -------
        str
            Generated LaTeX macro definitions.
        """
        macros_config = self.config.macros_config
        macro_parts = []
        
        # Standard command definitions
        commands = macros_config.get("commands", {})
        for cmd_name, cmd_def in commands.items():
            if isinstance(cmd_def, str):
                macro_parts.append(f"\\newcommand{{\\{cmd_name}}}{{{cmd_def}}}")
            elif isinstance(cmd_def, dict):
                args = cmd_def.get("args", 0)
                definition = cmd_def.get("definition", "")
                if args > 0:
                    macro_parts.append(f"\\newcommand{{\\{cmd_name}}}[{args}]{{{definition}}}")
                else:
                    macro_parts.append(f"\\newcommand{{\\{cmd_name}}}{{{definition}}}")
        
        # Environment definitions
        environments = macros_config.get("environments", {})
        for env_name, env_def in environments.items():
            if isinstance(env_def, dict):
                begin_def = env_def.get("begin", "")
                end_def = env_def.get("end", "")
                macro_parts.append(f"\\newenvironment{{{env_name}}}{{{begin_def}}}{{{end_def}}}")
        
        # Direct LaTeX code
        direct_latex = macros_config.get("direct_latex", [])
        if direct_latex:
            macro_parts.extend(direct_latex)
        
        return "\n".join(macro_parts)
    
    def get_document_structure(self, template_name: str | None = None) -> dict[str, Any]:
        """Get the document structure configuration.
        
        Parameters
        ----------
        template_name : str, optional
            Name of the template. If None, uses default template.
            
        Returns
        -------
        dict[str, Any]
            Document structure configuration.
        """
        template_config = self.load_template(template_name)
        return template_config.get("structure", {})
    
    def list_available_templates(self) -> list[str]:
        """Get list of available template names.
        
        Returns
        -------
        list[str]
            List of available template names.
        """
        template_files = list(self.config._config_dir.glob("template_*.yaml"))
        template_names = []
        
        for template_file in template_files:
            # Extract template name from filename
            name = template_file.stem.replace("template_", "")
            template_names.append(name)
            
        return sorted(template_names)
    
    def validate_template(self, template_name: str) -> tuple[bool, list[str]]:
        """Validate a template configuration.
        
        Parameters
        ----------
        template_name : str
            Name of the template to validate.
            
        Returns
        -------
        tuple[bool, list[str]]
            Tuple of (is_valid, list_of_errors).
        """
        errors = []
        
        try:
            template_config = self.load_template(template_name)
            
            # Check required fields
            if "documentclass" not in template_config:
                errors.append("Missing required field: documentclass")
            
            # Validate document class options
            if "documentclass_options" in template_config:
                options = template_config["documentclass_options"]
                if not isinstance(options, list):
                    errors.append("documentclass_options must be a list")
            
            # Validate structure if present
            if "structure" in template_config:
                structure = template_config["structure"]
                if not isinstance(structure, dict):
                    errors.append("structure must be a dictionary")
                    
        except Exception as e:
            errors.append(f"Error loading template: {e}")
        
        return len(errors) == 0, errors
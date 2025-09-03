"""Configuration management for paperx framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

PathLike = Union[str, Path]


class PaperConfig:
    """Configuration loader for paperx framework.
    
    Manages loading and validation of configuration files for templates,
    preambles, macros, and references following the qubex configuration patterns.
    """
    
    def __init__(
        self,
        *,
        config_dir: PathLike | None = None,
        template_name: str = "default",
        preamble_file: str = "preamble.yaml",
        macros_file: str = "macros.yaml", 
        references_file: str = "references.yaml",
    ) -> None:
        """Initialize configuration loader.
        
        Parameters
        ----------
        config_dir : PathLike, optional
            Directory containing configuration files. If None, uses default.
        template_name : str, optional
            Name of the template to use, by default "default".
        preamble_file : str, optional
            Name of the preamble configuration file, by default "preamble.yaml".
        macros_file : str, optional
            Name of the macros configuration file, by default "macros.yaml".
        references_file : str, optional
            Name of the references configuration file, by default "references.yaml".
        """
        self._config_dir = Path(config_dir) if config_dir else self._get_default_config_dir()
        self.template_name = template_name
        self.preamble_file = preamble_file
        self.macros_file = macros_file
        self.references_file = references_file
        
        # Load configurations
        self._preamble_config: dict[str, Any] | None = None
        self._macros_config: dict[str, Any] | None = None
        self._references_config: dict[str, Any] | None = None
    
    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory."""
        # Follow similar pattern to backend config_loader
        return Path.home() / ".paperx" / "config"
    
    def _load_config_file(self, file_name: str) -> dict[str, Any]:
        """Load a YAML configuration file.
        
        Parameters
        ----------
        file_name : str
            Name of the configuration file to load.
            
        Returns
        -------
        dict[str, Any]
            Loaded configuration data.
            
        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        yaml.YAMLError
            If there's an error parsing the YAML file.
        """
        path = self._config_dir / file_name
        try:
            with open(path, "r", encoding="utf-8") as file:
                result = yaml.safe_load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {path}") from e
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error loading configuration file: {path}") from e
        
        return result if result is not None else {}
    
    @property
    def preamble_config(self) -> dict[str, Any]:
        """Get preamble configuration."""
        if self._preamble_config is None:
            self._preamble_config = self._load_config_file(self.preamble_file)
        return self._preamble_config
    
    @property
    def macros_config(self) -> dict[str, Any]:
        """Get macros configuration."""
        if self._macros_config is None:
            self._macros_config = self._load_config_file(self.macros_file)
        return self._macros_config
    
    @property
    def references_config(self) -> dict[str, Any]:
        """Get references configuration."""
        if self._references_config is None:
            self._references_config = self._load_config_file(self.references_file)
        return self._references_config
    
    def get_template_config(self, template_name: str | None = None) -> dict[str, Any]:
        """Get configuration for a specific template.
        
        Parameters
        ----------
        template_name : str, optional
            Name of the template. If None, uses the default template name.
            
        Returns
        -------
        dict[str, Any]
            Template configuration.
        """
        if template_name is None:
            template_name = self.template_name
            
        template_file = f"template_{template_name}.yaml"
        return self._load_config_file(template_file)
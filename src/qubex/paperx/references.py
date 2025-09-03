"""Reference management for paperx framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PaperConfig


class ReferenceManager:
    """Manages structured references for papers.
    
    Provides functionality to manage BibTeX references with per-paper selection
    capabilities, allowing authors to maintain a central reference database
    while selecting specific references for each paper.
    """
    
    def __init__(self, config: PaperConfig) -> None:
        """Initialize reference manager.
        
        Parameters
        ----------
        config : PaperConfig
            Configuration instance containing reference settings.
        """
        self.config = config
        self._reference_database: dict[str, dict[str, Any]] = {}
        self._load_references()
    
    def _load_references(self) -> None:
        """Load references from configuration."""
        refs_config = self.config.references_config
        
        # Load central reference database
        if "database" in refs_config:
            database_path = self.config._config_dir / refs_config["database"]
            self._reference_database = self._load_bibtex_database(database_path)
        
        # Load individual reference entries
        if "entries" in refs_config:
            self._reference_database.update(refs_config["entries"])
    
    def _load_bibtex_database(self, path: Path) -> dict[str, dict[str, Any]]:
        """Load references from a BibTeX file.
        
        Parameters
        ----------
        path : Path
            Path to the BibTeX file.
            
        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary of reference entries keyed by citation key.
            
        Notes
        -----
        This is a simplified BibTeX parser. For production use, consider
        using a dedicated BibTeX parsing library like pybtex.
        """
        references = {}
        
        if not path.exists():
            return references
            
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                
            # Simple BibTeX parsing (simplified for demonstration)
            entries = self._parse_bibtex_simple(content)
            references.update(entries)
            
        except Exception as e:
            raise ValueError(f"Error parsing BibTeX file {path}: {e}") from e
            
        return references
    
    def _parse_bibtex_simple(self, content: str) -> dict[str, dict[str, Any]]:
        """Simple BibTeX parser for demonstration purposes.
        
        Parameters
        ----------
        content : str
            BibTeX file content.
            
        Returns
        -------
        dict[str, dict[str, Any]]
            Parsed reference entries.
            
        Notes
        -----
        This is a very basic parser for demonstration. A production
        implementation should use a proper BibTeX library.
        """
        entries = {}
        
        # Split into individual entries - improved pattern
        import re
        # Match entire entries from @ to closing }
        entry_pattern = r'@(\w+)\s*\{\s*([^,\s]+)\s*,\s*(.*?)\n\s*\}'
        matches = re.findall(entry_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for entry_type, key, fields_str in matches:
            fields = {}
            fields["type"] = entry_type.lower()
            
            # Parse fields - improved pattern for curly braces
            field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'
            field_matches = re.findall(field_pattern, fields_str)
            
            for field_name, field_value in field_matches:
                fields[field_name.lower()] = field_value.strip()
                
            entries[key.strip()] = fields
            
        return entries
    
    def get_references_for_paper(self, paper_refs: list[str]) -> dict[str, dict[str, Any]]:
        """Get references selected for a specific paper.
        
        Parameters
        ----------
        paper_refs : list[str]
            List of reference keys to include in the paper.
            
        Returns
        -------
        dict[str, dict[str, Any]]
            Selected references for the paper.
            
        Raises
        ------
        KeyError
            If a requested reference key is not found in the database.
        """
        selected_refs = {}
        
        for ref_key in paper_refs:
            if ref_key not in self._reference_database:
                raise KeyError(f"Reference '{ref_key}' not found in database")
            selected_refs[ref_key] = self._reference_database[ref_key]
            
        return selected_refs
    
    def generate_bibtex(self, paper_refs: list[str]) -> str:
        """Generate BibTeX content for selected references.
        
        Parameters
        ----------
        paper_refs : list[str]
            List of reference keys to include.
            
        Returns
        -------
        str
            BibTeX formatted reference content.
        """
        selected_refs = self.get_references_for_paper(paper_refs)
        bibtex_content = []
        
        for ref_key, ref_data in selected_refs.items():
            entry_type = ref_data.get("type", "article")
            bibtex_content.append(f"@{entry_type}{{{ref_key},")
            
            # Add fields
            for field, value in ref_data.items():
                if field != "type":
                    bibtex_content.append(f"  {field} = {{{value}}},")
            
            bibtex_content.append("}")
            bibtex_content.append("")  # Empty line between entries
            
        return "\n".join(bibtex_content)
    
    def list_available_references(self) -> list[str]:
        """Get list of all available reference keys.
        
        Returns
        -------
        list[str]
            List of available reference keys.
        """
        return list(self._reference_database.keys())
    
    def search_references(self, query: str) -> list[str]:
        """Search references by title, author, or other fields.
        
        Parameters
        ----------
        query : str
            Search query string.
            
        Returns
        -------
        list[str]
            List of matching reference keys.
        """
        matching_keys = []
        query_lower = query.lower()
        
        for ref_key, ref_data in self._reference_database.items():
            # Search in various fields
            searchable_text = " ".join([
                ref_data.get("title", ""),
                ref_data.get("author", ""),
                ref_data.get("journal", ""),
                ref_data.get("booktitle", ""),
                ref_key,
            ]).lower()
            
            if query_lower in searchable_text:
                matching_keys.append(ref_key)
                
        return matching_keys
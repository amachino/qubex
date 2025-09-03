"""Paperx: LaTeX paper writing framework.

A framework for writing academic papers with LaTeX that separates content from
configuration, allowing authors to focus on content while externalizing
preambles, macros, and reference management.
"""

from __future__ import annotations

from .config import PaperConfig
from .document import DocumentGenerator
from .references import ReferenceManager
from .templates import TemplateManager

__all__ = [
    "PaperConfig",
    "DocumentGenerator", 
    "ReferenceManager",
    "TemplateManager",
]
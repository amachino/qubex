"""QuEL-3 instrument configuration collections and YAML persistence."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from .instrument import InstrumentSpec


class InstrumentConfiguration(BaseModel):
    """Collect deployable instruments without cached runtime identity or state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruments: tuple[InstrumentSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_aliases(self) -> Self:
        """Require instrument aliases to be unique across all configured ports."""
        aliases: set[str] = set()
        for instrument in self.instruments:
            if instrument.alias in aliases:
                raise ValueError(f"Duplicate instrument alias `{instrument.alias}`.")
            aliases.add(instrument.alias)
        return self

    def save_yaml(self, path: str | Path) -> Path:
        """Save deployable settings as plain YAML, creating parent directories."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(
                self.model_dump(mode="json"),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        """Load and validate deployable settings without accessing hardware."""
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(document)

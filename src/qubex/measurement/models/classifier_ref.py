"""Serializable classifier reference model."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from pydantic import model_validator
from sklearn import __version__ as SKLEARN_VERSION

from qubex.core import DataModel
from qubex.measurement.classifiers.state_classifier import StateClassifier


@cache
def _load_classifier(
    path: str,
    sklearn_version: str,
    mtime_ns: int,
    file_size: int,
) -> StateClassifier:
    """Load and cache classifier instances by file identity and version."""
    _ = (mtime_ns, file_size)
    if sklearn_version != SKLEARN_VERSION:
        raise ValueError(
            "scikit-learn version mismatch for classifier load: "
            f"ref={sklearn_version}, runtime={SKLEARN_VERSION}."
        )
    return StateClassifier.load(path)


class ClassifierRef(DataModel):
    """Serializable classifier pointer."""

    path: str
    version: str = SKLEARN_VERSION

    @model_validator(mode="after")
    def _validate_fields(self) -> ClassifierRef:
        """Validate classifier reference metadata."""
        if self.path.strip() == "":
            raise ValueError("classifier_ref.path must not be empty.")
        if self.version.strip() == "":
            raise ValueError("classifier_ref.version must not be empty.")
        return self

    def load(self) -> StateClassifier:
        """Load classifier from serialized reference."""
        if self.version != SKLEARN_VERSION:
            raise ValueError(
                "scikit-learn version mismatch for classifier load: "
                f"ref={self.version}, runtime={SKLEARN_VERSION}."
            )
        resolved_path = Path(self.path).expanduser().resolve()
        file_info = resolved_path.stat()
        return _load_classifier(
            str(resolved_path),
            self.version,
            file_info.st_mtime_ns,
            file_info.st_size,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear process-wide classifier cache."""
        _load_classifier.cache_clear()

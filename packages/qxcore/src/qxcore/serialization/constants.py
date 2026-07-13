"""Shared constants for serialization."""

from __future__ import annotations

from typing import Final

FORMAT_NAME: Final[str] = "qxdata"
FORMAT_VERSION: Final[int] = 1

META_KEY: Final[str] = "__meta__"
META_FORMAT_KEY: Final[str] = "format"
META_VERSION_KEY: Final[str] = "version"

DATA_TYPE_KEY: Final[str] = "__type__"
DATA_NUMPY_PREFIX: Final[str] = "numpy."
DATA_TUNITS_PREFIX: Final[str] = "tunits."
DATA_PYTHON_PREFIX: Final[str] = "python."
DATA_COMPLEX_REAL_KEY: Final[str] = "real"
DATA_COMPLEX_IMAG_KEY: Final[str] = "imag"

TYPE_TUNITS_VALUE_ARRAY: Final[str] = "tunits.ValueArray"
TYPE_NUMPY_NDARRAY: Final[str] = "numpy.ndarray"

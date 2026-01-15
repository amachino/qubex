"""
Qubex core serialization (Python 3.10+).

Philosophy (qubex-oriented):
- Results contain numeric payloads: ndarray / complex are first-class.
- JSON is used as a structured, explicit transport format (not necessarily a bulk store).
- "Meaning-preserving" encoding:
    - complex scalar: {"__type__":"complex","re":...,"im":...}
    - ndarray:
        - dtype/shape/order are explicit
        - real arrays: flat "data": [...]
        - complex arrays: "data": {"__type__":"complex_array","re":[...],"im":[...]}
- Collision-resistant metadata:
    - internal: _meta_created_at, _meta_schema_version (non-user fields)
    - wire format: "__meta__": {...} (won't collide with user fields)
- Strict by default:
    - unknown fields -> error (can be relaxed per call)
- Safety guard:
    - inline array size limit to avoid JSON blow-ups (configurable)
- Extensible:
    - per-class codec configuration via @serializable_model(...)
    - lightweight registry for additional types if needed

Public API:
- SerializableModel: dataclass base that provides to_dict/to_json/from_dict/from_json
- serializable_model(...) decorator: per-class codec config
- dumps/loads: convenience for arbitrary objects (not typed model reconstruction)
"""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, TypeVar, get_args, get_origin
from uuid import UUID

__all__ = [
    "SerializationError",
    "UnknownFieldError",
    "MissingFieldError",
    "CodecError",
    "CodecRegistry",
    "DEFAULT_CODECS",
    "SerializableModel",
    "serializable_model",
    "dumps",
    "loads",
]


# -----------------------------
# Errors
# -----------------------------


class SerializationError(Exception):
    """Base error for serialization."""


class UnknownFieldError(SerializationError):
    """Raised when unknown fields are encountered in strict mode."""


class MissingFieldError(SerializationError):
    """Raised when required fields are missing during deserialization."""


class CodecError(SerializationError):
    """Raised when a codec cannot encode/decode a value."""


# -----------------------------
# datetime helpers (UTC ISO8601)
# -----------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def dt_to_iso_utc(dt: datetime) -> str:
    # Always serialize as UTC ISO8601 with "Z"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def dt_from_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# -----------------------------
# Typing helpers
# -----------------------------


def _is_optional(tp: Any) -> bool:
    origin = get_origin(tp)
    return origin is not None and type(None) in get_args(tp)


def _strip_optional(tp: Any) -> Any:
    if not _is_optional(tp):
        return tp
    args = tuple(a for a in get_args(tp) if a is not type(None))
    return args[0] if len(args) == 1 else tp


def _is_list(tp: Any) -> bool:
    return get_origin(tp) in (list,)


def _is_tuple(tp: Any) -> bool:
    return get_origin(tp) in (tuple,)


def _is_dict(tp: Any) -> bool:
    return get_origin(tp) in (dict,)


# -----------------------------
# Codec registry (minimal)
# -----------------------------


Jsonable = Any  # primitives/list/dict compatible with json.dumps


class CodecRegistry:
    """
    Minimal registry that encodes/decodes certain Python types into JSONable objects.

    Encoding uses:
      - tagged dict for special types: {"__type__": "...", ...payload...}
      - primitives and containers recursively encoded
      - dataclass instances encoded as a dict of fields (note: SerializableModel handles __meta__)

    Decoding:
      - tagged dict -> decoded by tag handler
      - dict/list recursively decoded
      - primitives pass through
    """

    TYPE_KEY: ClassVar[str] = "__type__"

    def __init__(self) -> None:
        self._encoders: dict[type, Callable[[Any, "CodecRegistry"], Jsonable]] = {}
        self._decoders: dict[
            str, Callable[[Mapping[str, Any], "CodecRegistry"], Any]
        ] = {}

    # ---- registration ----
    def register_type(
        self,
        py_type: type,
        tag: str,
        encode: Callable[[Any, "CodecRegistry"], Jsonable],
        decode: Callable[[Mapping[str, Any], "CodecRegistry"], Any],
    ) -> None:
        if tag in self._decoders:
            raise CodecError(f"Codec tag collision: {tag!r}")
        self._encoders[py_type] = encode
        self._decoders[tag] = decode

    def copy(self) -> "CodecRegistry":
        reg = CodecRegistry()
        reg._encoders.update(self._encoders)
        reg._decoders.update(self._decoders)
        return reg

    # ---- encode/decode ----
    def encode_any(self, obj: Any) -> Jsonable:
        # dataclass instance (NOT class)
        if is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: self.encode_any(getattr(obj, f.name)) for f in fields(obj)}

        # exact type encoder
        enc = self._encoders.get(type(obj))
        if enc is not None:
            return enc(obj, self)

        # Enum subclasses: encode via Enum base encoder if registered
        if isinstance(obj, Enum) and Enum in self._encoders:
            return self._encoders[Enum](obj, self)

        # Path subclasses (PosixPath/WindowsPath): encode via Path base encoder if registered
        if isinstance(obj, Path) and Path in self._encoders:
            return self._encoders[Path](obj, self)

        # containers
        if isinstance(obj, Mapping):
            return {str(k): self.encode_any(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.encode_any(v) for v in obj]
        if isinstance(obj, tuple):
            # JSON has no tuples -> represent as list (typed reconstruction handled by SerializableModel)
            return [self.encode_any(v) for v in obj]

        # primitives
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        raise CodecError(
            f"Object of type {type(obj).__name__} is not JSON-serializable (no codec)."
        )

    def decode_tagged(self, obj: Any) -> Any:
        if isinstance(obj, Mapping):
            if self.TYPE_KEY in obj:
                tag = obj[self.TYPE_KEY]
                dec = self._decoders.get(tag)
                if dec is None:
                    raise CodecError(f"Unknown codec tag: {tag!r}")
                return dec(obj, self)
            return {str(k): self.decode_tagged(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.decode_tagged(v) for v in obj]
        return obj


# -----------------------------
# Default codecs (safe + qubex numeric)
# -----------------------------


DEFAULT_CODECS = CodecRegistry()


def _enc_datetime(dt: datetime, reg: CodecRegistry) -> Jsonable:
    return {reg.TYPE_KEY: "datetime", "value": dt_to_iso_utc(dt)}


def _dec_datetime(obj: Mapping[str, Any], reg: CodecRegistry) -> datetime:
    v = obj.get("value")
    if not isinstance(v, str):
        raise CodecError("datetime codec expects a string 'value'.")
    return dt_from_iso(v)


DEFAULT_CODECS.register_type(datetime, "datetime", _enc_datetime, _dec_datetime)


def _enc_uuid(u: UUID, reg: CodecRegistry) -> Jsonable:
    return {reg.TYPE_KEY: "uuid", "value": str(u)}


def _dec_uuid(obj: Mapping[str, Any], reg: CodecRegistry) -> UUID:
    v = obj.get("value")
    if not isinstance(v, str):
        raise CodecError("uuid codec expects a string 'value'.")
    return UUID(v)


DEFAULT_CODECS.register_type(UUID, "uuid", _enc_uuid, _dec_uuid)


def _enc_path(p: Path, reg: CodecRegistry) -> Jsonable:
    return {reg.TYPE_KEY: "path", "value": str(p)}


def _dec_path(obj: Mapping[str, Any], reg: CodecRegistry) -> Path:
    v = obj.get("value")
    if not isinstance(v, str):
        raise CodecError("path codec expects a string 'value'.")
    return Path(v)


DEFAULT_CODECS.register_type(Path, "path", _enc_path, _dec_path)


def _enc_enum(e: Enum, reg: CodecRegistry) -> Jsonable:
    # Keep enum tag; actual Enum class reconstruction is field-type-driven in SerializableModel
    return {reg.TYPE_KEY: "enum", "value": reg.encode_any(e.value)}


def _dec_enum(obj: Mapping[str, Any], reg: CodecRegistry) -> Any:
    return reg.decode_tagged(obj.get("value"))


DEFAULT_CODECS.register_type(Enum, "enum", _enc_enum, _dec_enum)


def _enc_complex(z: complex, reg: CodecRegistry) -> Jsonable:
    return {reg.TYPE_KEY: "complex", "re": float(z.real), "im": float(z.imag)}


def _dec_complex(obj: Mapping[str, Any], reg: CodecRegistry) -> complex:
    re = obj.get("re")
    im = obj.get("im")
    if not isinstance(re, (int, float)) or not isinstance(im, (int, float)):
        raise CodecError("complex codec expects numeric 're' and 'im'.")
    return complex(float(re), float(im))


# qubex default: complex is first-class
DEFAULT_CODECS.register_type(complex, "complex", _enc_complex, _dec_complex)


def _enc_complex_array(
    re_list: list[float], im_list: list[float], reg: CodecRegistry
) -> Jsonable:
    return {reg.TYPE_KEY: "complex_array", "re": re_list, "im": im_list}


def _dec_complex_array(
    obj: Mapping[str, Any], reg: CodecRegistry
) -> tuple[list[float], list[float]]:
    re = obj.get("re")
    im = obj.get("im")
    if not isinstance(re, list) or not isinstance(im, list):
        raise CodecError("complex_array expects 're' and 'im' lists.")
    # values should be numbers; keep light checks
    for x in re:
        if not isinstance(x, (int, float)):
            raise CodecError("complex_array 're' must be numeric.")
    for x in im:
        if not isinstance(x, (int, float)):
            raise CodecError("complex_array 'im' must be numeric.")
    return ([float(x) for x in re], [float(x) for x in im])


# Tag is used only inside ndarray encoding (not for scalar fields)
DEFAULT_CODECS._decoders["complex_array"] = (
    _dec_complex_array  # intentional internal hook
)


# numpy ndarray support is default for qubex
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _require_numpy() -> Any:
    if np is None:
        raise CodecError("numpy is not available; cannot encode/decode ndarray.")
    return np


# -----------------------------
# SerializableModel (core)
# -----------------------------


M = TypeVar("M", bound="SerializableModel")


@dataclass(kw_only=True)
class SerializableModel:
    """
    Base class for qubex serializable dataclass models.

    Meta:
      - stored internally as _meta_created_at/_meta_schema_version (non-colliding)
      - serialized under "__meta__" dict

    Strictness:
      - unknown fields rejected by default (STRICT_UNKNOWN_FIELDS)

    Array policy:
      - ndarray inline item limit (MAX_INLINE_ARRAY_ITEMS)
      - dtype=object rejected
      - order is explicit ("C" or "F"), encode uses C by default
    """

    # collision-resistant internal meta
    _meta_created_at: datetime = field(default_factory=utc_now, init=False, repr=False)
    _meta_schema_version: int = field(default=1, init=False, repr=False)

    # constants
    META_KEY: ClassVar[str] = "__meta__"
    META_CREATED_AT: ClassVar[str] = "created_at"
    META_SCHEMA_VERSION: ClassVar[str] = "schema_version"

    STRICT_UNKNOWN_FIELDS: ClassVar[bool] = True
    DEFAULT_JSON_KWARGS: ClassVar[dict[str, Any]] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }

    # safety: JSON inline array size limit (elements)
    MAX_INLINE_ARRAY_ITEMS: ClassVar[int] = 200_000

    # per-class registry (can be overridden by @serializable_model)
    __codec_registry__: ClassVar[CodecRegistry] = DEFAULT_CODECS

    __serializable_numpy__: ClassVar[bool | None] = None
    __serializable_complex__: ClassVar[bool | None] = None

    # ---- meta API ----
    @property
    def created_at(self) -> datetime:
        return self._meta_created_at

    @property
    def schema_version(self) -> int:
        return self._meta_schema_version

    # ---- hook (schema migration) ----
    @classmethod
    def upgrade_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        return dict(data)

    # ---- ndarray encoding/decoding (qubex canonical) ----
    @classmethod
    def _encode_ndarray(cls, a: Any, reg: CodecRegistry) -> Jsonable:
        _np = _require_numpy()

        if not isinstance(a, _np.ndarray):
            raise CodecError("ndarray codec expected a numpy.ndarray.")
        if str(a.dtype) == "object":
            raise CodecError("ndarray codec does not support dtype=object.")

        # We fix encoding order to C for canonical representation.
        order = "C"
        flat_size = int(a.size)

        if flat_size > cls.MAX_INLINE_ARRAY_ITEMS:
            raise CodecError(
                f"ndarray too large to inline in JSON (size={flat_size} > {cls.MAX_INLINE_ARRAY_ITEMS}). "
                "Store payload in .npz/.npy/zarr and keep JSON as metadata."
            )

        dtype_str = str(a.dtype)
        shape = list(a.shape)

        # Complex arrays: store re/im separately (meaning-preserving, compact)
        if a.dtype.kind == "c":
            re = a.real.ravel(order=order).astype(float).tolist()
            im = a.imag.ravel(order=order).astype(float).tolist()
            data = {reg.TYPE_KEY: "complex_array", "re": re, "im": im}
        else:
            data = a.ravel(order=order).tolist()

        return {
            reg.TYPE_KEY: "ndarray",
            "dtype": dtype_str,
            "shape": shape,
            "order": order,
            "data": reg.encode_any(data),
        }

    @classmethod
    def _decode_ndarray(cls, obj: Mapping[str, Any], reg: CodecRegistry) -> Any:
        _np = _require_numpy()

        dtype = obj.get("dtype")
        shape = obj.get("shape")
        order = obj.get("order", "C")
        data = obj.get("data")

        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise CodecError("ndarray expects 'dtype'(str) and 'shape'(list).")
        if order not in ("C", "F"):
            raise CodecError("ndarray 'order' must be 'C' or 'F'.")

        decoded_data = reg.decode_tagged(data)

        # Complex array special payload
        if (
            isinstance(decoded_data, Mapping)
            and decoded_data.get(reg.TYPE_KEY) == "complex_array"
        ):
            # This branch happens only if someone manually wrapped; normally decode_tagged returns tuple
            re_list, im_list = _dec_complex_array(decoded_data, reg)
            re = _np.array(re_list, dtype=float).reshape(
                tuple(int(x) for x in shape), order=order
            )
            im = _np.array(im_list, dtype=float).reshape(
                tuple(int(x) for x in shape), order=order
            )
            return re.astype(dtype) + 1j * im.astype(dtype)

        if (
            isinstance(decoded_data, tuple)
            and len(decoded_data) == 2
            and all(isinstance(x, list) for x in decoded_data)
        ):
            # decode_tagged("complex_array") returns (re_list, im_list)
            re_list, im_list = decoded_data
            re = _np.array(re_list, dtype=float).reshape(
                tuple(int(x) for x in shape), order=order
            )
            im = _np.array(im_list, dtype=float).reshape(
                tuple(int(x) for x in shape), order=order
            )
            # Construct complex with requested dtype (complex64/128 etc.)
            return re.astype(dtype) + 1j * im.astype(dtype)

        # Real / general numeric array
        if not isinstance(decoded_data, list):
            raise CodecError(
                "ndarray 'data' must decode to a list (or complex_array payload)."
            )
        arr = _np.array(decoded_data, dtype=dtype)
        return arr.reshape(tuple(int(x) for x in shape), order=order)

    # ---- serialization ----
    def to_dict(self, *, include_meta: bool = True) -> dict[str, Any]:
        reg = self.__codec_registry__
        body: dict[str, Any] = {}

        # encode user fields only (exclude internal meta)
        for f in fields(self):
            if f.name.startswith("_meta_"):
                continue
            v = getattr(self, f.name)

            # ndarray special-case (canonical qubex encoding)
            if np is not None and isinstance(v, np.ndarray):
                body[f.name] = self._encode_ndarray(v, reg)
            else:
                body[f.name] = reg.encode_any(v)

        if not include_meta:
            return body

        meta = {
            self.META_CREATED_AT: reg.encode_any(self.created_at),
            self.META_SCHEMA_VERSION: int(self.schema_version),
        }
        return {self.META_KEY: meta, **body}

    def to_json(self, *, include_meta: bool = True, **json_kwargs: Any) -> str:
        kw = dict(self.DEFAULT_JSON_KWARGS)
        kw.update(json_kwargs)
        return json.dumps(self.to_dict(include_meta=include_meta), **kw)

    # ---- deserialization ----
    @classmethod
    def from_dict(
        cls: type[M],
        data: Mapping[str, Any],
        *,
        allow_unknown_fields: bool | None = None,
    ) -> M:
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        allow_unknown = (
            (not cls.STRICT_UNKNOWN_FIELDS)
            if allow_unknown_fields is None
            else allow_unknown_fields
        )
        reg = cls.__codec_registry__

        raw = cls.upgrade_dict(dict(data))

        # extract meta (non-colliding)
        meta = raw.pop(cls.META_KEY, None)
        created_at: datetime | None = None
        schema_version: int | None = None

        if meta is not None:
            if not isinstance(meta, Mapping):
                raise SerializationError(f"{cls.META_KEY} must be an object/dict.")
            if cls.META_CREATED_AT in meta:
                ca = reg.decode_tagged(meta[cls.META_CREATED_AT])
                if isinstance(ca, datetime):
                    created_at = ca
                elif isinstance(ca, str):
                    created_at = dt_from_iso(ca)
                else:
                    raise SerializationError(
                        "meta.created_at must decode to datetime or ISO string."
                    )
            if cls.META_SCHEMA_VERSION in meta:
                sv = meta[cls.META_SCHEMA_VERSION]
                if not isinstance(sv, int):
                    raise SerializationError("meta.schema_version must be an int.")
                schema_version = sv

        # dataclass fields excluding internal meta
        fdefs = {f.name: f for f in fields(cls) if not f.name.startswith("_meta_")}

        if not allow_unknown:
            unknown = set(raw.keys()) - set(fdefs.keys())
            if unknown:
                raise UnknownFieldError(
                    f"Unknown field(s) for {cls.__name__}: {sorted(unknown)}"
                )

        kwargs: dict[str, Any] = {}
        for name, f in fdefs.items():
            if name not in raw:
                continue
            val = reg.decode_tagged(raw[name])

            # ndarray tagged dict -> decode via canonical ndarray codec
            if isinstance(val, Mapping) and val.get(reg.TYPE_KEY) == "ndarray":
                val2 = cls._decode_ndarray(val, reg)
                kwargs[name] = _coerce_typed_value(f.type, val2, reg=reg)
            else:
                kwargs[name] = _coerce_typed_value(f.type, val, reg=reg)

        missing = _missing_required_fields(kwargs, fdefs)
        if missing:
            raise MissingFieldError(
                f"Missing required field(s) for {cls.__name__}: {missing}"
            )

        obj = cls(**kwargs)

        # apply meta after construction
        if created_at is not None:
            obj._meta_created_at = created_at
        if schema_version is not None:
            obj._meta_schema_version = schema_version
        return obj

    @classmethod
    def from_json(
        cls: type[M],
        s: str,
        *,
        allow_unknown_fields: bool | None = None,
        **json_kwargs: Any,
    ) -> M:
        obj = json.loads(s, **json_kwargs)
        if not isinstance(obj, Mapping):
            raise SerializationError(
                f"JSON must decode to an object/dict, got {type(obj).__name__}"
            )
        return cls.from_dict(obj, allow_unknown_fields=allow_unknown_fields)


def _missing_required_fields(
    kwargs: dict[str, Any], fdefs: dict[str, Any]
) -> list[str]:
    missing: list[str] = []
    for name, f in fdefs.items():
        if f.init is False:
            continue
        has_default = (f.default is not MISSING) or (f.default_factory is not MISSING)
        if not has_default and name not in kwargs:
            missing.append(name)
    return missing


def _coerce_typed_value(tp: Any, value: Any, *, reg: CodecRegistry) -> Any:
    """
    Coerce decoded value into declared field type (best-effort, strict where needed).

    Notes:
    - tuple is stored as JSON list; reconstructed only if field type is tuple[...]
    - Enum reconstruction is field-type-driven
    - nested SerializableModel handled
    - Path/UUID/datetime handled
    """
    if value is None:
        return None

    tp0 = _strip_optional(tp)

    # datetime
    if tp0 is datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return dt_from_iso(value)
        raise SerializationError(
            f"Expected datetime or ISO string, got {type(value).__name__}"
        )

    # Enum subclasses
    if isinstance(tp0, type) and issubclass(tp0, Enum):
        return tp0(value)

    # Path
    if tp0 is Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        raise SerializationError(f"Expected Path or str, got {type(value).__name__}")

    # UUID
    if tp0 is UUID:
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            return UUID(value)
        raise SerializationError(f"Expected UUID or str, got {type(value).__name__}")

    # nested SerializableModel
    if isinstance(tp0, type) and issubclass(tp0, SerializableModel):
        if isinstance(value, Mapping):
            return tp0.from_dict(value)
        raise SerializationError(
            f"Expected dict/object for {tp0.__name__}, got {type(value).__name__}"
        )

    # other dataclass
    if isinstance(tp0, type) and is_dataclass(tp0):
        if isinstance(value, Mapping):
            return tp0(**value)
        raise SerializationError(
            f"Expected dict/object for {tp0.__name__}, got {type(value).__name__}"
        )

    # list
    if _is_list(tp0):
        (elem_tp,) = get_args(tp0) if get_args(tp0) else (Any,)
        if not isinstance(value, list):
            raise SerializationError(
                f"Expected list for {tp0}, got {type(value).__name__}"
            )
        return [_coerce_typed_value(elem_tp, v, reg=reg) for v in value]

    # tuple (JSON stores as list)
    if _is_tuple(tp0):
        args = get_args(tp0)
        if not isinstance(value, list):
            raise SerializationError(
                f"Expected JSON array (list) for {tp0}, got {type(value).__name__}"
            )
        if len(args) == 2 and args[1] is Ellipsis:
            # tuple[T, ...]
            elem_tp = args[0]
            return tuple(_coerce_typed_value(elem_tp, v, reg=reg) for v in value)
        if args:
            if len(value) != len(args):
                raise SerializationError("Tuple arity mismatch.")
            return tuple(
                _coerce_typed_value(t, v, reg=reg) for t, v in zip(args, value)
            )
        return tuple(value)

    # dict
    if _is_dict(tp0):
        args = get_args(tp0)
        key_tp, val_tp = args if len(args) == 2 else (str, Any)
        if key_tp not in (str, Any):
            raise SerializationError("Only dict[str, T] is supported for JSON objects.")
        if not isinstance(value, Mapping):
            raise SerializationError(
                f"Expected dict/object for {tp0}, got {type(value).__name__}"
            )
        return {
            str(k): _coerce_typed_value(val_tp, v, reg=reg) for k, v in value.items()
        }

    return value


# -----------------------------
# Decorator: per-class codec config
# -----------------------------


def serializable_model(
    *,
    # knobs (currently informational; qubex defaults already include numpy/complex when numpy is available)
    numpy: bool | None = None,
    complex: bool | None = None,
    strict: bool | None = None,
    max_inline_array_items: int | None = None,
):
    """
    Configure per-class serialization policy.

    Typical usage:
        @serializable_model(strict=True, max_inline_array_items=50_000)
        @dataclass
        class MyResult(SerializableModel):
            ...

    Notes:
    - If numpy is not available, ndarray serialization will raise CodecError at runtime.
    - 'numpy' and 'complex' flags are kept for explicitness and future policy gating.
      Current qubex defaults already support complex always, and ndarray if numpy is installed.
    """

    def decorator(cls: type[SerializableModel]) -> type[SerializableModel]:
        if not issubclass(cls, SerializableModel):
            raise TypeError(
                "@serializable_model requires the class to inherit from SerializableModel"
            )

        # registry is copied so class-local changes don't affect others
        cls.__codec_registry__ = DEFAULT_CODECS.copy()

        if strict is not None:
            cls.STRICT_UNKNOWN_FIELDS = bool(strict)
        if max_inline_array_items is not None:
            cls.MAX_INLINE_ARRAY_ITEMS = int(max_inline_array_items)

        # Keep flags as metadata (for introspection / future gating)
        cls.__serializable_numpy__ = numpy
        cls.__serializable_complex__ = complex

        return cls

    return decorator


# -----------------------------
# Convenience JSON helpers
# -----------------------------


def dumps(obj: Any, **json_kwargs: Any) -> str:
    """
    Dump arbitrary objects to JSON using DEFAULT_CODECS.
    - SerializableModel uses its own to_json (includes __meta__ by default)
    - Other objects use DEFAULT_CODECS.encode_any
    """
    if isinstance(obj, SerializableModel):
        return obj.to_json(**json_kwargs)
    kw: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    kw.update(json_kwargs)
    return json.dumps(DEFAULT_CODECS.encode_any(obj), **kw)


def loads(s: str, **json_kwargs: Any) -> Any:
    """
    Load JSON to Python values, decoding tagged objects via DEFAULT_CODECS.
    For typed model reconstruction, use Model.from_json instead.
    """
    obj = json.loads(s, **json_kwargs)
    return DEFAULT_CODECS.decode_tagged(obj)

# API compatibility guidelines

This page defines compatibility rules for public or user-visible Python APIs.

## Deprecation style

- Use `typing_extensions.deprecated` for deprecated functions, methods, classes, and properties.
- Use `warnings.warn(..., DeprecationWarning)` only for conditional compatibility paths, such as deprecated keyword aliases that warn only when the old spelling is used.
- Keep deprecated aliases thin. They should delegate to the new API and avoid duplicating implementation logic.
- Add or update compatibility tests for the deprecated path until the path is removed.

## Deprecation messages

Write deprecation messages with three pieces of information:

1. What is deprecated.
2. What to use instead.
3. The earliest removal version.

Use a version-based removal statement, not a calendar deadline:

```python
@deprecated(
    "`old_api` is deprecated; use `new_api` instead. "
    "Deprecated in v1.5.0; will be removed no earlier than v1.6.0."
)
```

For stable public APIs, choose the removal version according to the compatibility promise of the package. If removal is a breaking change, the earliest removal version should be the next breaking-release line. For experimental or not-yet-released APIs, a shorter removal window is acceptable, but the message should still name the earliest removal version.

Do not write "will be removed soon" or a date unless the release date is already fixed in the release plan.

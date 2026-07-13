# Pulse ABC Migration Plan

This note documents the migration path from a concrete `Pulse` class to an abstract base class.

## Goals

- Keep type-based behavior checks on `Pulse` (`isinstance(x, Pulse)` stays valid).
- Move explicit arbitrary I/Q waveform construction to `Arbitrary`.
- Require `_sample_values` implementation for shape classes (`Gaussian`, `FlatTop`, `Drag`, and others).

## Current State

- `Pulse` direct instantiation is deprecated.
- `Arbitrary` is provided for explicit I/Q waveform usage.
- Built-in pulse shape classes define `_sample_values`.
- Subclassing `Pulse` without overriding `_sample_values` is deprecated.

## Future State

- `Pulse` becomes an abstract base class (ABC).
- `_sample_values` becomes abstract on `Pulse`.
- Any concrete `Pulse` subclass must implement `_sample_values`.
- `Arbitrary` remains a concrete `Pulse` subclass for explicit sampled waveforms.

## Compatibility Policy

- Use `isinstance(waveform, Pulse)` for pulse-type dispatch.
- Do not use `Pulse(...)` for arbitrary samples; use `Arbitrary(...)`.
- Existing call sites that rely on `Pulse` as a base type remain valid.

## Migration Guidance

1. Replace `Pulse(values)` with `Arbitrary(values)`.
2. For custom pulse classes inheriting `Pulse`, implement `_sample_values`.
3. Keep downstream type checks on `Pulse`, not on `Arbitrary`.

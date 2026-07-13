# v1.5.0 Pulse Lazy Sampling Notes

## Status

- State: `IMPLEMENTED`
- Documented on: 2026-02-19

## Purpose

Record the pulse sampling behavior introduced in the v1.5.0 development line.

## Summary

- Pulse sampling is now lazy by default.
- `Pulse` stores sampled values only when explicit `values` are provided, or when sampling is materialized later.
- Shape pulses (`Blank`, `Gaussian`, `FlatTop`, `Drag`, etc.) now implement `_sample_values()` and defer array creation until first materialization.

## Current `Pulse` behavior

`Pulse.__init__(..., values=None, duration=None, lazy=True)`:

- If `values` is provided:
  - values are normalized to `np.complex128`
  - `_values` and `_length` are set immediately
- If `values` is `None`:
  - only metadata/state is initialized
  - sampled values are generated when `_materialize_values()` is called (typically via `values` access)

`lazy` flag:

- Default is `True`.
- Subclasses can call `_finalize_initialization()` to eagerly materialize when `lazy=False` is requested.

## Materialization trigger points

Sampling runs when code path reaches waveform values, including:

- `Pulse.values`
- `PulseArray.get_values(...)`
- `PulseSchedule.get_sampled_sequence(...)`
- `PulseSchedule.get_sampled_sequences(...)`
- `PulseSchedule.values`
- plotting paths that call sequence/pulse values

Operations that do not require sampled arrays can remain lazy (for example structure-only schedule composition).

## Why this change

- Reduce memory pressure for long blank-heavy sequences.
- Keep pulse construction cheap until actual sampled arrays are needed.
- Prepare for backend-specific execution paths where sampled arrays are not always the optimal transport format.

## Known limitations

- If downstream flow requests sampled arrays (`values`), full materialization still occurs.
- Transport optimization (for example blank-elision event streams) must be handled in backend execution adapters and is not solved by lazy sampling alone.

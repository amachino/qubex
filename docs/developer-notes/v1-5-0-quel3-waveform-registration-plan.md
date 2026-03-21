# v1.5.0 QuEL-3 Waveform Registration Plan

## Status

- State: `PROPOSED`
- Documented on: 2026-02-19

## Purpose

Define the plan for QuEL-3 execution using `quelware-client` `Sequencer` with waveform reuse by shape.

## Background

- QuEL-3 uses `quelware-client` `Sequencer`.
- `register_waveform` supports waveform registration once and reuse with per-event:
  - `gain`
  - `phase_offset_deg`
- In Qubex waveform model, these correspond to:
  - `scale`
  - `phase`

This means pulses with the same shape but different scale/phase should reuse one registered waveform.

## Target behavior

### 1) Do not send blank regions as explicit sampled arrays

- Build timeline by `Sequencer.add_event(...)` only for non-blank regions.
- Represent blank as time gaps between events.

### 2) Register waveforms by shape key

- Same shape -> one `register_waveform(...)`
- Per-event amplitude/phase variation -> `gain` and `phase_offset_deg`

### 3) QuEL-3 sampling period policy

- QuEL-3 default sampling period in Qubex path: `0.4 ns`.
- Readout path downsampling behavior is delegated to `Sequencer`/quelware contract.

### 4) Temporary mixed-`dt` workaround in current implementation

Current Qubex execution still assumes one backend-level sampling period in the
QuEL-3 path. It does not yet model per-channel `dt` in `PulseSchedule` or in
backend execution contracts.

Until that design is implemented:

- control waveforms stay on the shared QuEL-3 control grid (`0.4 ns`)
- readout waveforms are normalized in the QuEL-3 measurement adapter to the
  readout grid (`0.8 ns`) before quelware registration
- waveform registration deduplicates by normalized registered waveform hash plus
  sampling period, not by original pulse-library hash alone

This workaround is intentionally local to the QuEL-3 adapter and should be
removed once per-channel sampling-period support becomes a first-class feature.

## Shape equivalence policy

Two pulses are considered equivalent for shared registration when:

- sampled shape is identical within tolerance after factoring out one complex scalar
- differences are only global amplitude/phase (complex scalar)

## Planned factoring algorithm

Given sampled pulse values `v`:

1. If all samples are effectively zero (blank), do not register waveform.
2. Find first sample with `abs(v[i]) > eps`.
3. Define complex scalar `c = v[i]`.
4. Normalize shape `s = v / c`.
5. Quantize/round `s` with deterministic tolerance and hash bytes as shape key.
6. Use `abs(c)` as `gain`, `angle(c)` as `phase_offset_deg`.

Notes:

- Prefer parametric fast-path (class + parameters + sampling period + length) when available.
- Use sampled fallback only when parametric identity is unavailable.

## Integration points

- QuEL-3 backend adapter/controller execution path:
  - schedule flattening
  - event extraction
  - waveform registry/cache
  - `Sequencer.register_waveform(...)` and `Sequencer.add_event(...)`

## Expected effects

- Fewer waveform registrations in repetitive sequences (for example randomized benchmarking).
- Lower payload size by avoiding long blank sample arrays.
- Better execution throughput on QuEL-3 path.

## Open questions

- Exact numeric tolerance for shape key stability.
- Preferred key source priority: parametric identity vs sampled hash.
- Cache lifetime boundary: per execution, per session, or longer.

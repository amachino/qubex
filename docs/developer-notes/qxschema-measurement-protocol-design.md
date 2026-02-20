# qxschema Measurement Protocol Design

## Status

- State: `PROPOSED`
- Last updated: `2026-02-20`
- Scope: policy and interface design only

## Goal

Publish the following four interfaces in `qxschema` as external shared protocol contracts:

- `MeasurementConfig`
- `MeasurementResult`
- `SweepMeasurementConfig`
- `SweepMeasurementResult`

This document defines the architecture and migration policy to achieve that goal
without forcing immediate one-to-one model equality with internal `qubex` models.

## Decision Summary

1. `qxschema` is the external contract source of truth.
2. `qubex` internal runtime models can diverge from `qxschema`.
3. Compatibility is guaranteed by explicit converters, not by model identity.
4. Contract stability is versioned and additive-first.
5. Existing sweep protocol drafts are respected, but tightened before "public stable" declaration.

## Why this direction

- Internal execution models need freedom for backend/runtime evolution.
- External users need stable, language-agnostic contracts.
- Forcing immediate model unification increases regression risk in active runtime paths.
- Converter boundaries make protocol changes explicit and testable.

## Architecture

### 1) Contract layer (`qxschema`)

- Owns protocol DTOs only.
- Must not depend on `qubex` runtime internals.
- Includes schema version fields and compatibility notes.

### 2) Runtime layer (`qubex`)

- Keeps current internal execution/data models as needed for performance and backend constraints.
- Internal field names and structures are allowed to differ from protocol DTOs.

### 3) Conversion layer (`qubex`)

- Provides `protocol -> runtime` and `runtime -> protocol` converters for all 4 interfaces.
- Converter behavior must be deterministic and documented.
- Lossy paths must surface explicit warnings or strict-mode errors.

## Protocol design principles

### P-001 Versioned contract

- Every public protocol model includes `protocol_version`.
- Breaking changes require a new version (for example `v2`), not silent mutation.

### P-002 Explicit units

- Use unit-explicit field naming for transport payloads (for example `interval_ns`, `frequency_hz`).
- Avoid relying on Python-specific wrapper types at the boundary.

### P-003 Typed over free-form metadata

- Prefer typed metadata fields over unbounded `dict` when the field is part of the contract.
- Keep an `extensions` map only for forward-compatible optional data.

### P-004 Stable array semantics

- For result payloads, axis meaning and ordering must be explicitly specified.
- Complex-valued and large-array serialization format must be defined in protocol docs.

### P-005 Additive-first evolution

- In-version updates are additive only (`optional` fields, new enum variants with fallback policy).
- Required-field deletion/rename is a version bump.

## Recommendations per interface

### MeasurementConfig

- Target state: close to current internal model shape.
- Keep core fields: mode/shots/interval + DSP options + frequency overrides.
- Clarify which fields are execution-effective vs advisory.

### MeasurementResult

- Keep as canonical transport result for schedule execution APIs.
- Standardize:
  - target/capture data layout
  - sampling period metadata semantics
  - config snapshots (`request_config` vs `runtime_config`) separation

### SweepMeasurementConfig

- Keep external proposal structure as baseline.
- Clarify currently ambiguous semantics:
  - axis ordering
  - multiple sweep keys in same axis
  - expression evaluation scope
  - `data_acquisition` fields that are mandatory for execution vs reserved metadata

### SweepMeasurementResult

- Current draft is too generic for stable contract publication.
- Before publish, define:
  - typed sweep-axis descriptors
  - strict mapping between sweep axes and result tensor shape
  - data key semantics and payload layout

## Converter contract requirements

### C-001 Round-trip expectation

- For stable fields, `protocol -> runtime -> protocol` should be lossless.

### C-002 Loss visibility

- If runtime has data not representable in current protocol version:
  - strict mode: raise explicit conversion error
  - permissive mode: drop with warning + structured conversion report

### C-003 Validation boundary

- Protocol validation errors are raised before runtime execution starts.
- Runtime-only constraints are validated separately and reported distinctly.

## Migration plan

1. Define `qxschema` v1 for the 4 interfaces with explicit field-level semantics.
2. Implement converter modules in `qubex` for all 4 interfaces.
3. Add contract tests:
   - schema validation
   - round-trip conversion
   - backward compatibility fixtures
4. Mark legacy/internal-only APIs as non-protocol in docs.
5. Publish protocol usage guide for external integrators.

## Open decisions

- Exact encoding strategy for complex arrays in JSON-oriented transports.
- Whether to split `MeasurementResult` config snapshot into typed submodels.
- How to handle optional legacy fields from external draft sweep payloads.
- Strict policy for unknown fields (`reject` vs `allow in extensions`).

## v1.5.0 relationship

- This protocol work is design-first and converter-first.
- Runtime model replacement/unification is not required for v1.5.0.
- Existing runtime behavior remains authoritative for execution until converter-backed protocol paths are completed.

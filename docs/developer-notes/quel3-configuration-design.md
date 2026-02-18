# QuEL-3 Configuration Design

## Purpose

Define a stable configuration contract for QuEL-3 that fits the current
`MeasurementClient` compatibility boundary and supports future backend growth.

Related policy:

- `quel3-wiring-binding-policy.md`
- `system-configuration-schema-draft.md`

## Current implementation snapshot

- Backend family selection is session-scoped:
  - `backend_kind="quel1" | "quel3"`
- QuEL-3 measurement execution path exists in:
  - `src/qubex/backend/quel3/quel3_backend_controller.py`
- Current runtime defaults are controller built-ins:
  - endpoint: `localhost`
  - port: `50051`
  - trigger wait: `1000000`
- Target-to-instrument resolution is performed by runtime-side logic.

## Decision log

Status legend:

- `DECIDED`: finalized
- `PENDING`: requires decision

### D1. Configuration source of truth

- Status: `IN_PROGRESS`
- Question: Where should QuEL-3 runtime configuration be defined primarily?
- Current behavior:
  - Backend family is selected by config file (`system.yaml` / `chip.yaml` fallback).
  - QuEL-3 runtime endpoint/port/trigger values are controller defaults.
- Target:
  - Config-file first (`system.yaml`) with optional explicit runtime override.

### D2. Config scope split

- Status: `DECIDED`
- Question: How should we split static vs runtime settings?
- Decision:
  - static:
    - chip metadata and topology (`chip.yaml`)
    - physical wiring (`wiring.v2.yaml`: `control/readout` with `qubit_id/mux_id -> port_id`)
  - runtime:
    - backend selection and backend-specific runtime settings (`system.yaml`)
    - endpoint, port, wait, ttl_ms, tentative_ttl_ms

### D2.1 System/chip cardinality

- Status: `DECIDED`
- Decision:
  - one `system.yaml` maps to one `chip_id`.

### D3. Alias and resource mapping policy

- Status: `DECIDED`
- Question: What is the canonical mapping path?
- Decision:
  - No dedicated target-binding config file in v1.5.0.
  - Mapping is resolved automatically by Qubex at runtime.
  - Physical source of truth is port-first wiring, not instrument-first mapping.

### D4. Session resource selection

- Status: `PENDING`
- Question: Which resources should be opened in one session?
- Candidate options:
  1. Only resources needed by the request payload
  2. All configured QuEL-3 resources
  3. Configurable policy (default: needed-only)

### D5. Timing and trigger policy

- Status: `PENDING`
- Question: How should trigger wait and timing policy be configured?
- Candidate options:
  1. `system.yaml` default (`wait`) + optional per-session override
  2. Per-target configurable
  3. Auto-tuned by observed hardware response

### D6. Result semantics for `single` and `avg`

- Status: `PENDING`
- Question: Which averaging semantics should be canonical for QuEL-3?
- Candidate options:
  1. Controller returns already averaged data for `avg`
  2. Controller returns per-shot, upper layer averages
  3. Configurable mode

### D7. Failure and fallback policy

- Status: `PENDING`
- Question: What should happen when an alias/resource is missing?
- Candidate options:
  1. Fail-fast with explicit error (recommended)
  2. Skip target and continue
  3. Fallback to target label guessing

### D8. Physical identifier base and label layer

- Status: `DECIDED`
- Question: How should physical identifiers and experiment labels be separated?
- Decision:
  - Physical identifiers (`qubit_id`, `resonator_id`, `mux_id`) are integer and zero-based.
  - Human-facing labels (`Qxxx`, `RQxxx`, `Mxx`) belong to the experiment layer.
  - For v1.5.0 scope, a standalone labels YAML is not required.
  - Runtime target resolution should prefer registry metadata, not label string parsing.

## Proposed minimum beta contract

- Single source policy for endpoint/port/wait and session TTL is documented.
- Alias mapping policy is deterministic and testable.
- Session resource selection policy is deterministic.
- Missing alias/resource behavior is fail-fast.
- `single`/`avg` result semantics are explicitly documented.

## Test implications

- Unit tests:
  - config resolution precedence
  - alias/resource mapping resolution
  - error paths (missing mapping, invalid config)
- Integration tests:
  - one minimal QuEL-3 measurement scenario with explicit config
  - one negative scenario (missing alias/resource)

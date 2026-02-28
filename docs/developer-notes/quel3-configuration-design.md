# QuEL-3 Configuration Design

## Purpose

Define a stable configuration contract for QuEL-3 that fits the current
`Measurement` compatibility boundary and supports future backend growth.

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
- Session defaults (provisional, aligned with `quelware-client`):
  - `ttl_ms=4000`
  - `tentative_ttl_ms=1000`
- Target-to-instrument resolution is performed by runtime-side logic.
- QuEL-3 system synchronizer is currently a no-op:
  - no backend-settings snapshot pull path is implemented yet
- `quelware-client` exposes `PORT`/`INSTRUMENT` resources and currently
  represents readout paths as transceiver-style resources in examples
  (`...:p0p1trx`).

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

- Status: `DECIDED`
- Decision:
  - Open only resources required by the request payload.
  - The selected resource set may span multiple units, and cross-unit synchronized trigger is required for beta gate scenarios.

### D5. Timing and trigger policy

- Status: `IN_PROGRESS`
- Current policy:
  - Use `quelware-client` defaults as provisional runtime values for beta:
    - endpoint=`localhost`
    - port=`50051`
    - trigger wait=`1000000`
    - `ttl_ms=4000`, `tentative_ttl_ms=1000`
- Target policy:
  - Move to config-file-first (`system.yaml`) with optional overrides in a later update.

### D6. Result semantics for `single` and `avg`

- Status: `DECIDED`
- Decision:
  - Canonical mode mapping follows quelware capture modes:
    - `single` -> `CaptureMode.VALUES_PER_ITER`
    - `avg` -> `CaptureMode.AVERAGED_VALUE`
    - waveform inspection flows (for example `check_waveform`) -> `CaptureMode.AVERAGED_WAVEFORM`

### D7. Failure and fallback policy

- Status: `DECIDED`
- Decision:
  - Unresolved or ambiguous alias/resource resolution must fail fast with explicit error.
  - Skip-and-continue behavior and target-label guessing are not allowed in beta contract.

### D8. Physical identifier base and label layer

- Status: `DECIDED`
- Question: How should physical identifiers and experiment labels be separated?
- Decision:
  - Physical identifiers (`qubit_id`, `resonator_id`, `mux_id`) are integer and zero-based.
  - Human-facing labels (`Qxxx`, `RQxxx`, `Mxx`) belong to the experiment layer.
  - For v1.5.0 scope, a standalone labels YAML is not required.
  - Runtime target resolution should prefer registry metadata, not label string parsing.

### D9. QuEL-3 readout `tx/rx/trx` handling in ExperimentSystem

- Status: `DECIDED`
- Question: How should Qubex handle quelware transceiver-style readout resources?
- Decision:
  - Keep `ExperimentSystem` logical model unchanged:
    - readout output (`read_out`) and input (`read_in`) remain explicit wiring roles.
  - In QuEL-3 runtime resolution, allow both roles to resolve to one transceiver
    resource/alias when wiring and instrument metadata are consistent.
  - For measurement execution payload, one readout alias may carry both:
    - waveform events (`tx` side)
    - capture windows (`rx` side)
  - Fail-fast rules:
    - unresolved alias/resource: fail
    - ambiguous candidates: fail
    - resolved resource role incompatible with requested operation: fail

### D10. `dump_box`-equivalent backend settings visibility

- Status: `PENDING`
- Question: Is there a QuEL-3 API equivalent to QuEL-1 `dump_box` for LO/NCO-like runtime settings?
- Current state:
  - QuEL-1 has `dump_box`-based synchronization and cache update.
  - QuEL-3 path does not expose equivalent settings in Qubex today.
  - Current `quelware-client` surface visible from this workspace includes:
    - `list_resource_infos`
    - `get_port_info`
    - `get_instrument_info`
    - execution/result APIs
  - No confirmed API currently returns QuEL-1-style per-port runtime settings
    (LO/CNCO/FNCO/VATT/FSC) as a pull snapshot.
- Interim policy for beta:
  - Treat QuEL-3 backend settings pull/sync as unsupported capability.
  - Ensure QuEL-1-only introspection utilities fail clearly on QuEL-3.

### D11. `system` package common vs backend-specific boundary

- Status: `DECIDED`
- Question: Which parts should stay shared, and which must split by backend?
- Decision:
  - Shared (`system` common):
    - quantum/chip topology and target registry
    - wiring loading and normalization
    - session-level backend-kind selection and orchestration entrypoint
  - Backend-specific:
    - hardware synchronization implementation
    - backend-settings snapshot schema and application logic
    - low-level runtime configuration/introspection semantics
- Reference:
  - `system-package-quel1-quel3-boundary.md`

### D12. `CharacterizationService` frequency-sweep semantics on QuEL-3

- Status: `PENDING`
- Question: How should qubit/resonator frequency scans work on QuEL-3 where
  QuEL-1-style LO/CNCO cache operations are unavailable?
- Current state:
  - `CharacterizationService.scan_qubit_frequencies()` and
    `scan_resonator_frequencies()` currently call
    `SystemManager.modified_backend_settings(...)` for subrange retuning.
  - `CharacterizationService.measure_electrical_delay()` also relies on the
    same backend-settings path for far-detuned starts.
  - QuEL-3 path currently treats backend-settings pull/sync and AWG/CAP reset
    as unsupported capabilities.
- Required beta policy:
  - QuEL-3 path must not rely on QuEL-1-only backend-settings cache operations.
  - Frequency sweep contract must be explicit:
    - either use an official quelware coarse-tuning API, or
    - constrain sweeps to a fixed coarse setting and sweep only supported fine
      range.
  - When requested range exceeds supported range, fail fast with a clear error
    and suggested valid range.
  - Capability and behavior differences must be visible in docs and tests.

## Proposed minimum beta contract

- Single source policy for endpoint/port/wait and session TTL is documented.
- Alias mapping policy is deterministic and testable.
- Session resource selection policy is deterministic.
- Missing alias/resource behavior is fail-fast.
- `single`/`avg` result semantics are explicitly documented.
- Cross-unit synchronized trigger behavior is required and validated.
- `tx/rx/trx` handling is deterministic:
  - logical readout `read_out`/`read_in` may converge to one transceiver alias in QuEL-3 runtime.
- Unsupported QuEL-3 settings introspection capability is explicit and non-silent.

## Test implications

- Unit tests:
  - config resolution precedence
  - alias/resource mapping resolution
  - readout `tx/rx/trx` convergence rules
  - error paths (missing mapping, invalid config)
- Integration tests:
  - one minimal QuEL-3 measurement scenario with explicit config
  - one negative scenario (missing/ambiguous alias/resource)
  - one scenario where readout out/in resolve to one transceiver alias

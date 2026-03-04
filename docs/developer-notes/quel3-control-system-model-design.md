# QuEL-3 Control-System Model Design

## Purpose

Define a concrete model boundary for QuEL-3 control semantics and clarify why it
must not be represented by the current QuEL-1-oriented `ControlSystem` class.

## Scope

- In scope:
  - model/class boundary between QuEL-1 and QuEL-3 control layers
  - terminology boundary (`box/port` vs `unit/instrument`)
  - deploy-time frequency planning responsibility
  - test and migration implications
- Out of scope:
  - full GA API shape for every manager method
  - quelware-internal implementation details

## Problem Statement

Current `src/qubex/system/control_system.py` is effectively a QuEL-1 model:

- topology unit is `Box` and `Port`
- low-level knobs (`lo_freq`, `cnco_freq`, `fnco_freq`, `vatt`, `fsc`) are stored
  directly in model objects
- behavior assumes direct port/channel tuning from Qubex-side model mutations

QuEL-3 differs at the control abstraction level:

- topology unit is `Unit`
- runtime resource is `Instrument` (not only `Port`)
- one instrument owns a frequency range, and deploy-time configuration uses that
  range to determine concrete `CNCO/FNCO` settings in quelware

Therefore, reusing the same concrete `ControlSystem` class for both backends
couples common models to QuEL-1 assumptions and causes abstraction leakage.

## Decision

Status legend:

- `DECIDED`: finalized for implementation direction
- `PENDING`: open for follow-up

### D1. Concrete control-system classes are backend-specific

- Status: `DECIDED`
- Decision:
  - Keep QuEL-1 concrete model as `box/port/channel` centric.
  - Introduce QuEL-3 concrete model as `unit/instrument` centric.
  - Do not force QuEL-3 to inherit QuEL-1 low-level tuning fields.

### D2. Common layer keeps logical experiment vocabulary only

- Status: `DECIDED`
- Decision:
  - Keep common `ExperimentSystem` vocabulary (`ctrl`, `read_out`, `read_in`,
    `pump`) and target registry behavior.
  - Keep wiring source of truth as physical port mapping (`wiring.v2.yaml`).
  - Allow QuEL-3 runtime to resolve one logical role pair (`read_out` +
    `read_in`) to one transceiver instrument (`trx`) when consistent.

### D3. Deploy-time frequency planning is backend responsibility on QuEL-3

- Status: `DECIDED`
- Decision:
  - QuEL-3 runtime resolves target to instrument and checks requested frequency
    against instrument range.
  - Concrete `CNCO/FNCO` values are set by quelware deploy flow.
  - Qubex common model must not require QuEL-1-style mutable NCO caches for
    QuEL-3.

### D4. QuEL-3 settings pull remains an explicit unsupported capability

- Status: `DECIDED`
- Decision:
  - QuEL-1-style `dump_box`/backend-settings pull remains unsupported on QuEL-3
    unless a dedicated quelware API is confirmed.
  - Unsupported paths must fail fast with explicit error messages.

## Proposed Model Boundary

### Common layer (`src/qubex/system`, backend-neutral)

- `QuantumSystem`, `TargetRegistry`, wiring normalization
- experiment assembly flow and logical wiring roles
- backend selection and orchestration entrypoints in `SystemManager`

### QuEL-1 concrete layer

- `Box -> Port -> Channel` model
- mutable low-level per-port/per-channel settings (`LO/CNCO/FNCO/...`)
- synchronizer and settings pull/push behaviors tied to QuEL-1 APIs

### QuEL-3 concrete layer

- `Unit -> Instrument` model
- instrument metadata contains:
  - stable identifier/alias
  - resource role (`tx`, `rx`, `trx`)
  - bound physical port identity
  - frequency range (`min_hz`, `max_hz`, optional `step_hz`)
- deploy planner validates requested frequencies and delegates actual NCO setup
  to quelware

## Runtime Flow (QuEL-3)

1. Load physical wiring (`qubit/mux -> port`) from `wiring.v2.yaml`.
2. Build measurement payload from logical targets and schedules.
3. Resolve target bindings to instrument aliases/resources at runtime.
4. Validate compatibility:
   - port consistency
   - role compatibility (`tx/rx/trx`)
   - requested frequency inside instrument range
5. Deploy via quelware; quelware determines and applies concrete `CNCO/FNCO`.
6. Trigger synchronized execution and fetch results.

Fail-fast rules:

- unresolved alias/resource: fail
- ambiguous alias candidates: fail
- out-of-range frequency request: fail with valid range hint

## Configuration Impact

- Keep `wiring.v2.yaml` as the physical source of truth.
- Do not add mandatory instrument-binding config for v1.5.0.
- Optional future extension:
  - instrument profile metadata in `system.yaml` for static range hints
  - still validated against live runtime resolver information

## Implementation Plan

1. Extract a minimal backend-neutral control capability contract:
   - query/validate operations only
   - no shared mutable LO/NCO knobs
2. Isolate QuEL-1-only constants and traits from common model paths.
3. Introduce QuEL-3 control model entities (`Unit`, `Instrument`, range model)
   under backend-specific package paths.
4. Wire QuEL-3 deploy-time frequency validation into execution manager path.
5. Add explicit capability gates and unsupported errors for QuEL-1-only utility
   flows on QuEL-3.

## Test Implications

- Unit tests:
  - port-to-instrument resolution with `tx/rx/trx` compatibility checks
  - range validation behavior (in-range, out-of-range, boundary)
  - ambiguity and unresolved binding fail-fast behavior
- Integration tests:
  - one multi-unit synchronized run with resolved instrument aliases
  - one transceiver convergence case (`read_out` and `read_in` to one alias)
  - one negative out-of-range deployment scenario

## Related Notes

- `system-package-quel1-quel3-boundary.md`
- `quel3-configuration-design.md`
- `quel3-wiring-binding-policy.md`

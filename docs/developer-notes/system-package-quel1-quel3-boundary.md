# System Package QuEL-1/QuEL-3 Boundary

## Purpose

Define how far `src/qubex/system` can be shared across QuEL-1 and QuEL-3,
and where backend-specific split is required for v1.5.0 and GA hardening.

## Current observation

- Shared entrypoint (`SystemManager`) already selects backend family per session.
- QuEL-1 has full synchronizer and backend-settings pull path (`dump_box`-based).
- QuEL-3 synchronizer is currently no-op; backend-settings pull path is absent.
- Core model classes still include QuEL-1-oriented constants/traits in
  `control_system.py` and `experiment_system.py`.
- QuEL-1 rebuilds controller-side runtime model from `ExperimentSystem`.
- QuEL-3 does not require controller-side logical-model rebuild before `push()`;
  it can consume `ExperimentSystem` directly at hardware-sync time.

## Boundary decision

### Must remain common

- chip/topology model (`QuantumSystem`, `TargetRegistry`)
- wiring loading/normalization (`ConfigLoader`, `wiring.v2` normalization)
- experiment-system assembly flow
- session-level backend selection and orchestration in `SystemManager`
- fail-fast policy for unresolved/ambiguous runtime resolution
- backend-facing planning inputs derived from shared logical models
  (`ExperimentSystem`, `Target`, `ControlParams`)

### Must remain backend-specific

- hardware synchronization implementation (`Quel1SystemSynchronizer`, `Quel3SystemSynchronizer`)
- backend-settings snapshot acquisition and schema
- low-level hardware introspection API mapping (QuEL-1 `dump_box` style)
- coarse tuning strategy for spectroscopy (`LO/CNCO` retune semantics)
- clock/trigger/device-control semantics not representable in shared contract
- operational tools that depend on QuEL-1-only APIs (`experiment_tool.dump_box`, etc.)
- QuEL-3 quelware session ownership and `deploy_instruments(...)` execution
- runtime deployment cache ownership (`target_alias_map`, deployed instrument infos)

## v1.5.0 required actions

1. Introduce explicit capability labels in docs and API behavior:
   - `backend_settings_pull`: supported on QuEL-1, unsupported on QuEL-3 (current)
   - `hardware_push_configure`: supported on QuEL-1 and QuEL-3
2. Gate QuEL-1-only utility paths with clear unsupported errors on QuEL-3.
3. Keep `ExperimentSystem` logical readout split (`read_out`, `read_in`) as common
   vocabulary, but allow QuEL-3 runtime convergence to one `trx` alias/resource.
4. Avoid adding QuEL-3-specific hardware details into common model classes unless
   they are backend-agnostic abstractions.
5. Refactor `CharacterizationService` frequency-sweep path to use backend
   capabilities/strategy; do not hard-require QuEL-1-only LO/CNCO cache APIs on
   QuEL-3.
6. Implement QuEL-3 `push()` deploy flow with an explicit split:
   - system-side planner converts logical targets to deploy requests
   - backend-side configuration manager owns quelware deploy execution and cache
   while keeping backend-settings pull unsupported.
7. Keep `sync_experiment_system_to_backend_controller(...)` meaningful for
   QuEL-1 rebuilds, but avoid using it as a fake pre-push cache step on
   QuEL-3.

## Post-beta refactor candidates

1. Isolate QuEL-1 constants/traits behind backend profile objects.
2. Move low-level frequency/device knobs out of common `system` models where possible.
3. Define a backend-neutral settings snapshot interface only if multiple backends
   can provide comparable semantics.

## Risks if boundary is not enforced

- QuEL-1 assumptions leak into QuEL-3 code path and cause runtime errors.
- Common models become tightly coupled to one backend and block future extensions.
- Operator utilities fail unpredictably instead of failing fast with clear errors.

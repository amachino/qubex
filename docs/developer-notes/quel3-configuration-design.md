# QuEL-3 configuration design

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
- Current runtime defaults are controller and runtime built-ins:
  - endpoint: `localhost`
  - port: `50051`
  - trigger `wait_ms=None` (use `quelware-client` default handling)
- Session TTLs are explicit Qubex runtime values:
  - `ttl_ms=30000`
  - `tentative_ttl_ms=5000`
- Target names are instrument aliases. The controller owns one `InstrumentCache`
  containing complete hardware information, including resource IDs.
- QuEL-3 `push()` plans an `InstrumentConfiguration` from `TargetRegistry`, deploys through the
  configuration manager, then reads touched ports back into the shared cache.
- QuEL-3 pull collects an independent backend-settings snapshot. Runtime
  instrument information is acquired during connect, deploy, or explicit refresh.
- QuEL-3 backend runtime supports only the `server` client mode.
- Current runtime still uses process-global `SystemManager` singleton state:
  - one active experiment/measurement session per process is assumed
  - `Experiment(..., backend_controller=...)` is treated as session-start
    configuration, not per-reload override state
- `quelware-client` exposes `PORT`/`INSTRUMENT` resources and currently
  represents readout paths as transceiver-style resources in examples
  (`...:p0p1trx`).

## Instrument configuration and YAML

`InstrumentSpec` contains exactly five configuration fields:

```yaml
instruments:
  - port_id: unit-a:tx_p01
    alias: Q00
    role: TRANSMITTER
    frequency_range_min_hz: 4000000000.0
    frequency_range_max_hz: 4200000000.0
```

`InstrumentConfiguration.instruments` stores the specifications. The controller
provides the public API for acquiring, inspecting, saving, loading, and deploying
them:

```python
controller.refresh_instrument_cache(unit_labels=["unit-a"])
configuration = controller.get_instrument_configuration()
path = controller.save_instrument_configuration("instruments.yaml")
loaded = controller.load_instrument_configuration(path)
controller.deploy_instruments(configuration=loaded)
```

Get and save use cached hardware information without another hardware read.
Load is pure configuration parsing: it neither deploys nor changes the cache.
YAML contains specifications only. Resource IDs and driver configuration are
runtime information held in the controller's private `InstrumentCache`.

Deployment replaces the instruments on touched ports and reads hardware back
into that cache. Connect loads all existing instruments. Explicit refresh
replaces all or selected units. An empty
configuration or an explicitly empty refresh selection performs no work.

`Quel3HardwareState` is an independent diagnostic snapshot. It can contain partial
data and read issues, and is never used as executable instrument cache input.
SystemManager pull, hardware-state display, and `is_synced()` leave the
instrument cache unchanged.

## Decision log

Status legend:

- `DECIDED`: finalized
- `PENDING`: requires decision

### D1. Configuration source of truth

- Status: `IN_PROGRESS`
- Question: Where should QuEL-3 runtime configuration be defined primarily?
- Current behavior:
  - Backend family is selected by config file (`system.yaml` selected entry `backend`).
  - QuEL-3 runtime endpoint/port/trigger values are controller defaults.
- Target:
  - Config-file first (`system.yaml` keyed by `system_id`) with optional
    explicit runtime override.

### D2. Config scope split

- Status: `DECIDED`
- Question: How should we split static vs runtime settings?
- Decision:
  - static:
    - chip metadata and topology (`chip.yaml`)
    - reusable hardware inventory (`box.yaml`)
      - QuEL-3 `address` / `adapter` fields are optional
    - physical wiring (`wiring.yaml`: `system_id -> control/readout` with
      `qubit_id/mux_id -> port_id`)
  - runtime:
    - deployment selection and backend-specific runtime settings
      (`system.yaml`: `system_id -> chip_id, backend, backend runtime`)
    - endpoint, port, trigger `wait_ms`, ttl_ms, tentative_ttl_ms

### D2.1 System/chip cardinality

- Status: `DECIDED`
- Decision:
  - one `system.yaml` entry maps to one `chip_id`
  - many system entries may reference the same `chip_id`

### D2.2 Box selection policy

- Status: `DECIDED`
- Decision:
  - Do not require a `boxes:` field in `system.yaml` for the baseline schema.
  - Derive the active box set from `wiring.yaml[system_id]`.
  - Validate that every referenced box id exists in `box.yaml`.

### D3. Alias and resource mapping policy

- Status: `DECIDED`
- Question: What is the canonical mapping path?
- Decision:
  - No dedicated target-binding config file in v1.5.0.
  - The target name is the instrument alias. Execution reads its resource ID
    and runtime configuration from the shared cache.
  - Physical source of truth is port-first wiring, not instrument-first mapping.

### D4. Session resource selection

- Status: `DECIDED`
- Decision:
  - Open only resources required by the request payload.
  - The selected resource set may span multiple units, and cross-unit synchronized trigger is required for beta gate scenarios.

### D5. Timing and trigger policy

- Status: `IN_PROGRESS`
- Current policy:
  - Use `quelware-client` 0.4.1-compatible runtime values for beta:
    - endpoint=`localhost`
    - port=`50051`
    - trigger `wait_ms=None`
    - `ttl_ms=30000`, `tentative_ttl_ms=5000`
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
  - Missing cached instruments and duplicate aliases must fail with explicit errors.
  - Skip-and-continue behavior and target-label guessing are not allowed in beta contract.

### D8. Physical identifier base and label layer

- Status: `DECIDED`
- Question: How should physical identifiers and experiment labels be separated?
- Decision:
  - Physical identifiers (`qubit_id`, `resonator_id`, `mux_id`) are integer and zero-based.
  - Human-facing labels (`Qxxx`, `RQxxx`, `Mxx`) belong to the experiment layer.
  - Do not introduce a separate labels YAML for v1.5.0; resolve labels from
    chip/graph or registry metadata instead.
  - Runtime target resolution should prefer registry metadata, not label string parsing.

### D9. QuEL-3 readout `tx/rx/trx` handling in ExperimentSystem

- Status: `DECIDED`
- Question: How should Qubex handle quelware transceiver-style readout resources?
- Decision:
  - Keep `ExperimentSystem` logical model unchanged:
    - readout output (`read_out`) and input (`read_in`) remain explicit wiring roles.
  - The planner combines the paired ports into one transceiver deployment for
    each readout target; its timeline carries output and capture operations.
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
  - QuEL-3 exposes instrument snapshots containing identity, port, role, and
    definition information; these are not QuEL-1 per-port tuning settings.
  - Current `quelware-client` surface visible from this workspace includes:
    - `list_resource_infos`
    - `get_port_info`
    - `get_instrument_info`
    - execution/result APIs
  - No confirmed API currently returns QuEL-1-style per-port runtime settings
    (LO/CNCO/FNCO/VATT/FSC) as a pull snapshot.
- Interim policy for beta:
  - Support instrument snapshot pull. Keep saved-settings application separate
    from acquisition of executable instrument information.
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

- Status: `IN_PROGRESS`
- Question: How should qubit/resonator frequency scans work on QuEL-3 where
  QuEL-1-style LO/CNCO cache operations are unavailable?
- Current state:
  - `CharacterizationService.scan_qubit_frequencies()` and
    `scan_resonator_frequencies()` currently call
    `SystemManager.modified_backend_settings(...)` for subrange retuning.
  - `CharacterizationService.measure_electrical_delay()` also relies on the
    same backend-settings path for far-detuned starts.
  - QuEL-3 path currently treats unsupported backend-settings override and
    AWG/CAP reset requests as compatibility no-op instead of raising.
  - This fallback keeps the software path alive, but it does not yet define the
    valid coarse/fine sweep contract for QuEL-3 hardware.
- Required beta policy:
  - QuEL-3 path must not rely on QuEL-1-only backend-settings cache operations.
  - Frequency sweep contract must be explicit:
    - either use an official quelware coarse-tuning API, or
    - constrain sweeps to a fixed coarse setting and sweep only supported fine
      range.
  - When requested range exceeds supported range, fail fast with a clear error
    and suggested valid range.
  - Capability and behavior differences must be visible in docs and tests.

### D13. QuEL-3 `push()` semantics and manager ownership

- Status: `DECIDED`
- Question: Where should QuEL-3 instrument deployment be implemented?
- Decision:
  - Treat instrument deployment as configuration-stage behavior triggered by
    `SystemManager.push(...)`.
  - Keep execution manager focused on run-time sequencing and result retrieval.
  - Do not require controller-side logical-model rebuild before QuEL-3
    `push()`.
  - Split QuEL-3 push responsibilities explicitly:
    - system-side planner converts active targets into
      `InstrumentConfiguration` of five-field `InstrumentSpec` values
    - backend-side `Quel3ConfigurationManager` owns
      `session.deploy_instruments(...)`
    - `Quel3BackendController` owns the shared `InstrumentCache` and delegates
      deployment and readback to `Quel3ConfigurationManager`
  - Planner responsibilities:
    - active-target (`ExperimentContext.targets`) -> deploy definition conversion
    - one-instrument-per-target range planning using
      `target.frequency ± frequency_margin`
    - port and role derivation from logical target metadata
  - Backend configuration-manager responsibilities:
    - quelware client/session lifecycle for deploy
    - readback through `Quel3HardwareStateReader` and updates to the supplied
      controller-owned cache
  - Shared-port deployment policy:
    - one port may host multiple instruments
    - configuration manager must not collapse same-port specifications to one
      definition or overwrite earlier deploys accidentally
    - send all selected definitions for each port together with `append=False`
    - replace cache entries on those ports after successful hardware readback
    - preserve unrelated ports; an empty configuration does not change the cache
    - if deployment or readback fails, leave touched ports uncached
  - Session/runtime assumption for current scope:
    - `SystemManager` remains singleton-managed
    - QuEL-3 server runtime settings are assumed stable for the active session
    - interleaving multiple active `Experiment` / `Measurement` objects with
      different backend runtime settings in one process is not supported
  - Future direction:
    - move `SystemManager` from process-global singleton state to
      session/experiment-owned state
    - remove remaining ambient runtime assumptions at that point
  - Execution uses cached instrument information. Operators acquire it through
    connection, deployment, or explicit controller refresh.
  - Pull, pure fetch, and `is_synced()` collect diagnostic snapshots independently
    of executable instrument state.

## Proposed minimum beta contract

- Single source policy for endpoint/port/wait and session TTL is documented.
- Alias mapping policy is deterministic and testable.
- Session resource selection policy is deterministic.
- Missing alias/resource behavior is fail-fast.
- `single`/`avg` result semantics are explicitly documented.
- Cross-unit synchronized trigger behavior is required and validated.
- `tx/rx/trx` handling is deterministic:
  - logical readout `read_out`/`read_in` may converge to one transceiver alias in QuEL-3 runtime.
- Instrument snapshot pull is supported; QuEL-1 tuning-cache operations remain
  backend-specific.

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

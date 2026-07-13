# QuEL-3 Wiring And Binding Policy

## Purpose

Define the configuration split for QuEL-3 so that:

- experimenters can continue physical rewiring workflows,
- measurement execution remains stable when quelware resource exposure varies,
- target-label parsing is removed from execution-time logic.

## Decision

### Primary source: `qubit (mux) -> port`

- The QuEL-3 equivalent of legacy `wiring.yaml` should remain a physical wiring map.
- The primary mapping unit is `qubit/mux` to `port_id` (not instrument).
- This keeps the configuration aligned with how operators actually rewire systems.

### Physical identifier policy

- Physical IDs should be integer-based and zero-based for compatibility:
  - `qubit_id`: `0..`
  - `resonator_id`: `0..`
  - `mux_id`: `0..`
- Human-facing labels (for example `Q001`, `RQ01`) belong to the experiment layer
  and should not be embedded in physical wiring keys.
- For v1.5.0 scope, a dedicated labels YAML is not required; labels can be
  derived from chip/graph metadata at runtime.

### Target-to-instrument mapping policy

- A dedicated target-binding configuration file is not required for v1.5.0.
- Qubex resolves target-to-instrument mapping automatically at runtime.
- Manual override config can be reconsidered later only if field requirements appear.

## Rationale

- Existing wiring design is physical-port based (`<box>:<port>` preferred, legacy `<box>-<port>`), and current system assembly also starts from ports.
- In quelware, instruments are attached to a port (`InstrumentInfo.port_id`) and can be deployed as multiple instruments per port.
- Alias-based lookup is convenient for execution but unstable as a physical source of truth.
- Some systems may expose only combined ports (for example `p0p1trx`) rather than decomposed `p0` and `p1`.
- Therefore, port-first mapping is the most robust baseline across hardware variants.

## Recommended configuration split

### 1. Physical wiring file (port-first)

Store physical correspondence only.

- File name: `wiring.v2.yaml`
- `qubit_id -> control port_id`
- `mux_id -> readout out/in/pump port_id`

Example:

```yaml
schema_version: 2
chip_id: 64Q

control:
  0: unit-a:p2tx
  1: unit-a:p3tx

readout:
  0:
    out: unit-a:p0p1trx
    in: unit-a:p0p1trx
    pump: unit-a:p4tx
```

## Runtime resolution policy

1. Resolve target properties from `TargetRegistry` (no label parsing dependency).
2. Resolve physical port from wiring map.
3. Convert runtime bindings to `<unit>:<port>` using physical port metadata:
   - `port.box_id`
   - integer `port.number`
   - `experiment_system.get_box(box_id).name`
4. Resolve instrument alias automatically in runtime from deployed alias map when available.
5. Validate consistency (`instrument.port_id` matches expected port when port constraints are required).
6. Fail fast on unresolved or inconsistent mapping.

## Runtime contract

- QuEL-3 execution does not fall back to logical `port.id` strings such as `QT1.CTRL1`.
- Missing physical port metadata is a configuration error and must raise immediately.
- Measurement execution should prefer deployed `target_alias_map` over port-based alias inference.
- Port-based inference remains a validation path, not the primary happy path.
- Deploy-time instrument alias is identical to `target_label`.

## Deployed instrument `port_id` contract

- Runtime resolver accepts current deploy-time resource identifiers only:
  - `unit-a:tx_p04`
  - `unit-a:rx_p00`
  - `unit-a:trx_p00p01`
- Legacy formats are intentionally unsupported in the current unreleased implementation.

## Impact on v1.5.0 scope

- This policy is a design baseline for QuEL-3 configuration work.
- It does not require changing QuEL-1 wiring behavior.
- It should be reflected in:
  - QuEL-3 configuration loader design,
  - `TargetRegistry` introduction plan,
  - integration tests for missing/ambiguous binding cases.

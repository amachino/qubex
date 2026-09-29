# QuEL-3 wiring and instrument identity policy

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
- Do not introduce a dedicated labels YAML for v1.5.0; derive labels from
  chip/graph or registry metadata at runtime.

### QuEL-3 unit identity

Use the quelware unit label as the `box.yaml` key and `name`, and use the same
label in `wiring.yaml` box references. No box-to-unit translation is performed.

```yaml
quel3-02-a01:
  name: quel3-02-a01
  type: quel3
```

### Target-to-instrument mapping policy

- A dedicated target-binding configuration file is not required for v1.5.0.
- The planner uses physical wiring to select a deployment port.
- The deployed instrument alias is exactly the target name; execution uses that
  name to read the shared `InstrumentCache`. No target-to-alias map is stored.

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
  0: unit-a:tx_p02
  1: unit-a:tx_p03

readout:
  0:
    out: unit-a:trx_p00p01
    in: unit-a:trx_p00p01
    pump: unit-a:tx_p04
```

## Deployment and execution policy

1. Resolve target metadata from `TargetRegistry`.
2. During planning, use `port.box_id` directly as the quelware unit label and
   derive the unit-qualified port from it and integer `port.number`. Pair readout
   output/input wiring into the transceiver port.
3. Build an `InstrumentConfiguration` containing one five-field `InstrumentSpec`
   per target, using the target name as alias. Deploy the configuration through
   the controller; definitions for each port form one call with `append=False`.
4. Read complete instrument information back from hardware into the
   controller-owned `InstrumentCache`, replacing only the touched ports.
5. Execute timelines keyed by target name using the cached resource ID and
   driver configuration. A missing target requires deploy or explicit refresh.

## Runtime contract

- Physical metadata is required by deployment planning; execution does not
  reconstruct a port or infer an alias from wiring.
- Local target aliases are unique across the controller's instrument cache.
  Duplicate aliases fail explicitly, including duplicates on different units.
- Unit decoration belongs to resource/port IDs. When quelware returns a
  unit-decorated alias, cache insertion strips only the matching unit prefix.
- A partial deploy replaces every instrument on each touched port. Include all
  targets that must remain on a shared port. Other cached ports are preserved.
- Connect acquires all existing instruments, and explicit refresh acquires all
  or selected units. Pull and inspection collect
  diagnostic snapshots independently of the execution cache.
- Applications use the controller's get/save/load instrument configuration APIs.
  Get and save export cached specifications; load reads YAML without changing
  hardware or runtime state.

## Deployed instrument `port_id` contract

Current deployment identifiers are unit-qualified:

- `unit-a:tx_p04`
- `unit-a:rx_p00`
- `unit-a:trx_p00p01`

`InstrumentSpec` carries this full port ID, alias, role, and minimum/maximum
frequency in Hz. `InstrumentConfiguration` holds the selected specifications.
Its YAML representation contains these fields only; resource IDs and driver
configuration are acquired from hardware during connect, deploy, or explicit refresh.

## Impact on v1.5.0 scope

- This policy is a design baseline for QuEL-3 configuration work.
- It does not require changing QuEL-1 wiring behavior.
- It should be reflected in:
  - QuEL-3 configuration loader design,
  - `TargetRegistry` introduction plan,
  - integration tests for missing instruments, duplicate aliases, and scoped cache replacement.

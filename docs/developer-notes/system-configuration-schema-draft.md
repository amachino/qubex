# System Configuration Schema

## Purpose

Define a clean split between reusable catalogs (`chip`, `box`) and
system-specific deployment manifests (`system`, `wiring`).

## Scope and status

- Status: target schema for the pre-release configuration reorganization.
- Current implementation supports:
  - `system.yaml` keyed by `system_id`
  - `wiring.yaml` keyed by `system_id`
  - deprecated `chip_id` lookup as a compatibility path when it resolves to
    exactly one system entry
- Target relation:
  - `system.yaml` is a catalog keyed by `system_id`
  - one system entry maps to one `chip_id`
  - many system entries may reference the same `chip_id`
  - `wiring.yaml` is keyed by `system_id`

## Design decisions

### D1. File split

- `chip.yaml`: chip catalog
  - chip-intrinsic metadata
  - `name`, `n_qubits`, `topology`
- `box.yaml`: box catalog
  - reusable hardware inventory
  - `type`, optional backend-specific connection metadata
  - `address` and `adapter` remain required for QuEL-1-family boxes
  - `address` and `adapter` are optional for QuEL-3 boxes
- `system.yaml`: system catalog
  - `system_id -> chip_id, backend, backend-specific runtime settings`
- `wiring.yaml`: system wiring
  - `system_id -> mux/qubit/resonator to box-port or backend-resource mapping`

### D2. System/chip cardinality

- One system entry maps to one `chip_id`.
- Many system entries may reference the same `chip_id`.
- Example:
  - `64Q-HF-Q1 -> chip_id: 64Q-HF`
  - `144Q-LF-Q1 -> chip_id: 144Q-LF`
  - `144Q-LF-Q3 -> chip_id: 144Q-LF`

### D3. Wiring scope

- Wiring belongs to the system layer, not the chip layer.
- `wiring.yaml` is keyed by `system_id`, because wiring describes how one chip
  deployment is attached to one concrete set of boxes/resources.
- Box usage is derived from wiring references.
- `system.yaml` does not need an explicit `boxes:` field by default.
- Preferred port specifier format is `<box_id>:<port>`.
- Legacy `<box_id>-<port>` remains accepted for compatibility.

### D4. Backend section style

- No `common:` section for now.
- Keep only backend-specific sections.
- Active section is selected by top-level `backend` inside each system entry.

### D5. Box selection policy

- Loader resolves the active box set from wiring references.
- Every box id referenced by wiring must exist in `box.yaml`.
- If a future use case requires declaring unwired but system-owned boxes,
  `boxes:` may be added later as an optional extension. It is not part of the
  baseline schema.

### D6. Parameter key style

- Qubit-scoped parameter files should use qubit indices as keys.
- Example:
  - preferred: `0: 4.123`
  - compatibility: `Q000: 4.123`
- Loader normalizes integer indices to the canonical qubit labels for the
  selected chip.

## Proposed schema

### chip.yaml

```yaml
64Q-HF:
  name: "64-qubit high-frequency example chip"
  n_qubits: 64
  topology:
    type: square_lattice
    mux_size: 4

144Q-LF:
  name: "144-qubit low-frequency example chip"
  n_qubits: 144
  topology:
    type: square_lattice
    mux_size: 4
```

### box.yaml

```yaml
BOX1:
  name: "Control Rack A"
  type: quel1-a
  address: 10.0.0.2
  adapter: dummy

QT1:
  name: "QuEL-3 Unit 1"
  type: quel3
```

### system.yaml

```yaml
64Q-HF-Q1:
  chip_id: 64Q-HF
  backend: quel1
  quel1:
    clock_master: 10.0.0.10

144Q-LF-Q3:
  chip_id: 144Q-LF
  backend: quel3
  quel3:
    endpoint: localhost
    port: 50051
```

### wiring.yaml

```yaml
64Q-HF-Q1:
  - mux: 0
    ctrl: [BOX1:2, BOX1:4, BOX1:9, BOX1:11]
    read_out: BOX1:1
    read_in: BOX1:0

144Q-LF-Q3:
  - mux: 0
    ctrl: [QT1:4, QT1:2, QT1:11, QT1:9]
    read_out: QT1:1
    read_in: QT1:0
```

## Validation rules

- `system_id` must exist in `system.yaml`.
- `system_id` should exist in `wiring.yaml`.
- `chip_id` referenced by a system entry must exist in `chip.yaml`.
- `backend` must be `quel1` or `quel3`.
- Selected backend section is optional in the baseline schema.
- Unselected backend sections are ignored.
- `topology.type` is required when `topology` is provided.
- Every box id referenced in `wiring.yaml[system_id]` must exist in `box.yaml`.

## Loader resolution model

Target load order:

1. Select one `system_id`
2. Read `system.yaml[system_id]`
3. Resolve `chip_id` from the selected system entry
4. Read `chip.yaml[chip_id]`
5. Read `wiring.yaml[system_id]`
6. Derive used box ids from wiring references
7. Read only the referenced box definitions from `box.yaml`
8. Read parameter files from the selected `params_dir`

## Compatibility policy

- `system_id` is the canonical loader input.
- `chip_id` may remain as an optional compatibility input during migration.
- When `chip_id` is used as loader input:
  - emit a deprecation warning
  - resolve the corresponding system entry from `system.yaml`
- `chip_id` compatibility lookup is allowed only when it identifies exactly one
  system entry.
- If no system entry matches the given `chip_id`, loading fails.
- If multiple system entries match the given `chip_id`, loading fails and the
  caller must specify `system_id` explicitly.

Example:

- allowed:
  - `chip_id=64Q-HF` and only one system entry references `64Q-HF`
- error:
  - `144Q-LF-Q1 -> chip_id: 144Q-LF`
  - `144Q-LF-Q3 -> chip_id: 144Q-LF`
  - loader input is only `chip_id=144Q-LF`

## Migration note

- Current implementation resolves backend from the selected system entry in
  `system.yaml` and resolves wiring from `system_id` first, then `chip_id`
  only as a compatibility fallback.
- During migration, `chip_id` may be accepted as a deprecated compatibility
  input, but it must not silently choose one system when multiple systems share
  the same chip.
- This design intentionally avoids an explicit `boxes:` field in `system.yaml`
  unless later requirements justify it.

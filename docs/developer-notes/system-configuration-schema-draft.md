# System Configuration Schema Draft

## Purpose

Define a clean split between chip metadata and runtime/backend settings.

## Scope and status

- Status: draft for upcoming implementation work.
- Target relation: one `system.yaml` describes one `chip_id`.
- This document defines desired schema, not current loader behavior.

## Design decisions

### D1. File split

- `chip.yaml`: chip-intrinsic metadata
  - `name`, `n_qubits`, `topology`
- `system.yaml`: runtime/backend settings
  - `backend`
  - backend-specific settings (`quel1`, `quel3`)

### D2. Topology key style

- Use `topology.type` (not `topology.kind`) in YAML.
- Internal code may normalize to `*_kind` naming if needed.

### D3. Backend section style

- No `common:` section for now.
- Keep only backend-specific sections.
- Active section is selected by top-level `backend`.

### D4. Clock-master policy

- `quel1` section may define `clock_master`.
- `quel3` section does not require `clock_master` in current quelware-client API.

## Proposed schema

### chip.yaml

```yaml
64Q:
  name: "2023-1st-64Q-No14-run3 chip (1,0)"
  n_qubits: 64
  topology:
    type: square_lattice
    mux_size: 4
```

### system.yaml

```yaml
schema_version: 1
chip_id: 64Q
backend: quel3

quel1:
  clock_master: 10.0.0.10

quel3:
  endpoint: 10.0.0.20
  port: 50051
  trigger_wait: 1000000
  session_ttl_ms: 4000
  tentative_ttl_ms: 1000
```

## Validation rules (draft)

- `chip_id` must exist in `chip.yaml`.
- `backend` must be `quel1` or `quel3`.
- Selected backend section must exist.
- Unselected backend sections are ignored.
- `topology.type` is required when `topology` is provided.

## Migration note

- During transition, `chip.yaml` `backend` may be accepted as a compatibility fallback.
- After `system.yaml` adoption, runtime/backend selection should move to `system.yaml`.

# System Configuration Schema

## Purpose

Define a clean split between chip metadata and runtime/backend settings.

## Scope and status

- Status: baseline for v1.5.0 (implemented for backend selection and QuEL-1 clock-master resolution).
- Target relation: one `system.yaml` describes one `chip_id`.
- QuEL-3 runtime section fields are currently staged work; runtime endpoint/port/trigger values use controller defaults.

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

quel3: {}
```

## Validation rules

- `chip_id` must exist in `chip.yaml`.
- `backend` must be `quel1` or `quel3`.
- Selected backend section is optional for v1.5.0.
- Unselected backend sections are ignored.
- `topology.type` is required when `topology` is provided.

## Migration note

- Backend selection is resolved from `system.yaml` top-level `backend` only
  (or explicit runtime override). `chip.yaml` does not carry backend selection.
- For v1.5.0, QuEL-3 runtime endpoint/port/trigger values use controller defaults.
